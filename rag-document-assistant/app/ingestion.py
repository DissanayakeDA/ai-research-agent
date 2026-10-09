"""The ingestion pipeline: PDF bytes -> pages -> chunks -> vectors -> ChromaDB.

scripts/ingest.py uses it now, and the FastAPI upload endpoint will use it in Phase 6,
so both paths behave identically.

Repeat uploads are safe. The document ID is a hash of the file's bytes, so a file that is
already indexed - under any name - is recognised and skipped before any embedding work.
"""

import logging
import time

from app.chunker import chunk_document
from app.config import Settings
from app.document_loader import DocumentLoadError, load_pdf_bytes
from app.embeddings import Embedder, EmbeddingError
from app.schemas import IngestResult
from app.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


def ingest_pdf(data: bytes, filename: str, settings: Settings, embedder: Embedder, store: VectorStore) -> IngestResult:
    """Index one PDF. Never raises for a bad file: the result's status says what happened."""
    started = time.perf_counter()

    try:
        document = load_pdf_bytes(data, filename, settings.max_upload_bytes)
    except DocumentLoadError as exc:
        return IngestResult(filename=filename, status="failed", message=str(exc))

    try:
        existing = store.find_document(document.doc_id)
    except VectorStoreError as exc:
        return IngestResult(filename=filename, status="failed", message=str(exc), doc_id=document.doc_id)
    if existing is not None:
        reason = "already indexed" if existing.filename == filename else f"same content as '{existing.filename}', already indexed"
        return IngestResult(
            filename=filename, status="skipped", message=reason,
            doc_id=document.doc_id, pages=document.page_count, chunks=existing.chunk_count,
        )

    chunks = chunk_document(document, settings.chunk_size, settings.chunk_overlap)
    texts = [chunk.text for chunk in chunks]
    truncated = sum(count > embedder.max_tokens for count in embedder.count_tokens(texts))
    if truncated:
        logger.warning(
            "'%s': %d chunk(s) exceed the model's %d-token limit; text past that point is ignored",
            filename, truncated, embedder.max_tokens,
        )

    try:
        embeddings = embedder.embed_documents(texts)  # all vectors first: a failure here stores nothing
        store.add_chunks(chunks, embeddings)
    except (EmbeddingError, VectorStoreError) as exc:
        _remove_partial_document(store, document.doc_id)
        return IngestResult(filename=filename, status="failed", message=str(exc), doc_id=document.doc_id)

    seconds = time.perf_counter() - started
    logger.info("Indexed '%s': %d pages, %d chunks in %.1fs", filename, document.page_count, len(chunks), seconds)
    return IngestResult(
        filename=filename, status="indexed", message=f"indexed {len(chunks)} chunks",
        doc_id=document.doc_id, pages=document.page_count, chunks=len(chunks),
        truncated_chunks=truncated, seconds=round(seconds, 2),
    )


def _remove_partial_document(store: VectorStore, doc_id: str) -> None:
    """If a write failed halfway, delete what was written - otherwise the next upload would
    find the document 'already indexed' and skip it, leaving it incomplete forever."""
    try:
        store.delete_document(doc_id)
    except VectorStoreError:
        logger.exception("Could not clean up partially indexed document %s", doc_id[:12])
