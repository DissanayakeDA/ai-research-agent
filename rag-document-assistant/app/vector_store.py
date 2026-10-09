"""Store chunk vectors in ChromaDB on disk, together with their text and metadata.

A vector database answers one question quickly: "which stored vectors are closest to this
one?" ChromaDB builds an HNSW index - a graph linking each vector to its near neighbours -
so a search visits a small part of the collection instead of comparing against every vector.

Each chunk becomes one record:
    id         "<doc_id>:<chunk_index>"  - writing the same id again overwrites, never duplicates
    embedding  384 numbers
    document   the chunk text            - what the LLM and the user will read
    metadata   filename, pages, ...      - what citations are built from

One collection holds one *configuration* (embedding model + chunk size + overlap). Vectors
from different models live in different spaces, and mixing chunk sizes would muddy
experiments, so changing any of these settings switches to a different collection.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import chromadb

from app.config import Settings
from app.schemas import Chunk, RetrievedChunk, StoredDocument

logger = logging.getLogger(__name__)

WRITE_BATCH_SIZE = 500


class VectorStoreError(Exception):
    """ChromaDB could not be opened, read or written."""


def collection_name_for(embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    """e.g. "BAAI-bge-small-en-v1.5_c1000_o150" (Chroma allows letters, digits, . _ -)."""
    model = re.sub(r"[^A-Za-z0-9._-]", "-", embedding_model)
    return f"{model}_c{chunk_size}_o{chunk_overlap}"


def open_store(settings: Settings) -> "VectorStore":
    """The collection for the configuration in .env (created on first use)."""
    return VectorStore(
        settings.chroma_dir,
        collection_name_for(settings.embedding_model, settings.chunk_size, settings.chunk_overlap),
        metadata={
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )


class VectorStore:
    def __init__(self, path: Path, collection_name: str, metadata: dict[str, str | int] | None = None):
        self.path = path
        self.collection_name = collection_name
        try:
            self._client = chromadb.PersistentClient(
                path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                collection_name,
                configuration={"hnsw": {"space": "cosine"}},  # distance = 1 - cosine similarity
                metadata=metadata,  # only saved when the collection is first created
                embedding_function=None,  # we always supply our own vectors
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open ChromaDB collection '{collection_name}' in {path}: {exc}") from exc

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        try:
            for i in range(0, len(chunks), WRITE_BATCH_SIZE):
                batch = chunks[i : i + WRITE_BATCH_SIZE]
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings[i : i + WRITE_BATCH_SIZE],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[_metadata(chunk) for chunk in batch],
                )
        except Exception as exc:
            raise VectorStoreError(f"Writing chunks to ChromaDB failed: {exc}") from exc

    def query(self, embedding: list[float], n_results: int) -> list[RetrievedChunk]:
        """The `n_results` stored chunks whose vectors are closest to `embedding`, closest first."""
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results,  # an empty or smaller collection simply returns fewer
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Searching ChromaDB failed: {exc}") from exc

        rows = zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        return [
            RetrievedChunk(
                rank=rank,
                score=1.0 - distance,  # the collection uses cosine distance = 1 - cosine similarity
                chunk_id=chunk_id,
                doc_id=metadata["doc_id"],
                filename=metadata["filename"],
                chunk_index=metadata["chunk_index"],
                page_start=metadata["page_start"],
                page_end=metadata["page_end"],
                text=text,
            )
            for rank, (chunk_id, text, metadata, distance) in enumerate(rows, start=1)
        ]

    def find_document(self, doc_id: str) -> StoredDocument | None:
        """The stored document with this content hash, or None if it isn't indexed."""
        records = self._get(where={"doc_id": doc_id})
        if not records:
            return None
        return StoredDocument(doc_id=doc_id, filename=records[0]["filename"], chunk_count=len(records))

    def list_documents(self) -> list[StoredDocument]:
        records = self._get()
        counts = Counter(record["doc_id"] for record in records)
        names = {record["doc_id"]: record["filename"] for record in records}
        documents = [StoredDocument(doc_id=doc_id, filename=names[doc_id], chunk_count=n) for doc_id, n in counts.items()]
        return sorted(documents, key=lambda document: document.filename.lower())

    def delete_document(self, doc_id: str) -> None:
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"Deleting document {doc_id[:12]}... failed: {exc}") from exc

    def count(self) -> int:
        """Number of chunks in this collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete every chunk in this collection (other configurations are untouched)."""
        metadata = self._collection.metadata
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata=metadata,
                embedding_function=None,
            )
        except Exception as exc:
            raise VectorStoreError(f"Resetting collection '{self.collection_name}' failed: {exc}") from exc

    def _get(self, where: dict | None = None) -> list[dict]:
        """Metadata of matching chunks (no vectors or text - they aren't needed here)."""
        try:
            return self._collection.get(where=where, include=["metadatas"])["metadatas"]
        except Exception as exc:
            raise VectorStoreError(f"Reading from ChromaDB failed: {exc}") from exc


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }
