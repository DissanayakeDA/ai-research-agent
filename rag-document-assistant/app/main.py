"""FastAPI backend: the HTTP interface to ingestion and question answering.

    uvicorn app.main:app        then open http://127.0.0.1:8000/docs to try every endpoint

    GET  /health      is the service up; which models; how much is indexed
    GET  /documents   the documents in the index
    POST /documents   upload PDFs (multipart form field "files"); one status per file
    POST /ask         {"question": "...", "top_k": 4, "min_score": null} -> answer + sources

The embedding model, the vector store and the LLM client are created ONCE at startup and
shared by every request. Loading the model per request would add seconds to each one.
"""

import logging
import re
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import Settings, configure_logging, load_settings
from app.embeddings import Embedder, EmbeddingError
from app.generator import Generator
from app.ingestion import ingest_pdf
from app.llm_client import LLMError
from app.pipeline import RagPipeline, build_llm_client
from app.retriever import MAX_QUESTION_CHARS, Retriever
from app.schemas import Answer, IngestResult, StoredDocument
from app.vector_store import VectorStore, VectorStoreError, open_store

logger = logging.getLogger(__name__)

MAX_FILES_PER_UPLOAD = 50


@dataclass
class Services:
    """Everything the endpoints need, created once when the server starts."""

    settings: Settings
    embedder: Embedder
    store: VectorStore
    pipeline: RagPipeline
    ingest_lock: threading.Lock = field(default_factory=threading.Lock)  # index one upload at a time


def build_services(settings: Settings) -> Services:
    embedder = Embedder(settings.embedding_model)
    store = open_store(settings)
    pipeline = RagPipeline(Retriever(embedder, store), Generator(build_llm_client(settings)))
    return Services(settings=settings, embedder=embedder, store=store, pipeline=pipeline)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=20)  # None -> TOP_K from .env
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class Health(BaseModel):
    status: str
    documents: int
    chunks: int
    collection: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_mb: int
    llm_model: str
    llm_base_url: str
    llm_is_local: bool  # False -> retrieved passages are sent to a third-party provider


def create_app(services_factory: Callable[[], Services] | None = None) -> FastAPI:
    """`services_factory` lets tests run the real endpoints with fake models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services_factory is not None:
            app.state.services = services_factory()
        else:
            settings = load_settings()
            configure_logging(settings.log_level)
            app.state.services = build_services(settings)
        logger.info("Ready: %d chunks in '%s'", app.state.services.store.count(), app.state.services.store.collection_name)
        yield

    app = FastAPI(title="RAG Document Assistant", lifespan=lifespan)

    def get_services(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health", response_model=Health)
    def health(services: Services = Depends(get_services)) -> Health:
        settings = services.settings
        return Health(
            status="ok",
            documents=len(services.store.list_documents()),
            chunks=services.store.count(),
            collection=services.store.collection_name,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
            max_upload_mb=settings.max_upload_mb,
            llm_model=settings.llm_model,
            llm_base_url=settings.llm_base_url,  # never the API key
            llm_is_local=settings.llm_is_local,
        )

    @app.get("/documents", response_model=list[StoredDocument])
    def list_documents(services: Services = Depends(get_services)) -> list[StoredDocument]:
        return services.store.list_documents()

    # Plain `def` (not `async def`): FastAPI runs it in a worker thread, so slow embedding
    # work doesn't freeze the server for other requests.
    @app.post("/documents", response_model=list[IngestResult])
    def upload_documents(files: list[UploadFile], services: Services = Depends(get_services)) -> list[IngestResult]:
        if len(files) > MAX_FILES_PER_UPLOAD:
            raise HTTPException(400, f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.")
        settings = services.settings
        results = []
        for upload in files:
            filename = Path(upload.filename or "upload.pdf").name  # never trust a client-supplied path
            if not filename.lower().endswith(".pdf"):
                results.append(IngestResult(filename=filename, status="failed", message="Only PDF files are accepted."))
                continue
            # Read at most one byte past the limit: enough to know a file is too big without
            # loading all of it into memory.
            data = upload.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                results.append(IngestResult(
                    filename=filename, status="failed",
                    message=f"'{filename}' is larger than the {settings.max_upload_mb} MB limit.",
                ))
                continue
            with services.ingest_lock:
                result = ingest_pdf(data, filename, settings, services.embedder, services.store)
            if result.status == "indexed":
                _keep_copy(settings.uploads_dir, result.doc_id, filename, data)
            logger.info("Upload '%s': %s (%s)", filename, result.status, result.message)
            results.append(result)
        return results

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest, services: Services = Depends(get_services)) -> Answer:
        top_k = request.top_k or services.settings.top_k
        try:
            return services.pipeline.ask(request.question, top_k, request.min_score)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, f"The language model request failed: {exc}") from exc
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(500, str(exc)) from exc

    return app


def _keep_copy(folder: Path, doc_id: str, filename: str, data: bytes) -> None:
    """Save the uploaded PDF, so the index can be rebuilt later with different settings."""
    safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", filename)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{doc_id[:12]}_{safe_name}").write_bytes(data)


app = create_app()
