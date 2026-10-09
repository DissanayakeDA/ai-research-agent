"""Test doubles and builders shared by several test files."""

import math
import zlib

import pymupdf

from app.embeddings import EmbeddingError
from app.llm_client import LLMReply


def make_pdf(*page_texts: str, **save_options) -> bytes:
    """A PDF with one page per string ("" = a page with no text, like a scanned image)."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)  # "\n" starts a new line on the page
    data = pdf.tobytes(**save_options)
    pdf.close()
    return data


class FakeEmbedder:
    """Stands in for the real model: instant, no download, deterministic.

    Each word is hashed into one of 16 slots, so texts sharing words get similar vectors.
    That is enough to test storage and ingestion logic - it says nothing about the
    semantic quality of the real model (tests/test_embeddings.py covers that).
    """

    model_name = "fake-model"
    dimension = 16

    def __init__(self, fail: bool = False, max_tokens: int = 512):
        self.fail = fail
        self.max_tokens = max_tokens
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise EmbeddingError("Embedding failed: simulated failure")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % self.dimension] += 1.0
        length = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / length for x in vector]


class FakeLLM:
    """Stands in for the LLM API: returns a fixed reply (or raises) and records every request."""

    model = "fake-llm"

    def __init__(self, reply: str = "The CP-SAT solver was faster [1].", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return LLMReply(content=self.reply, prompt_tokens=100, completion_tokens=20, finish_reason="stop")


def make_services(data_dir, llm: FakeLLM):
    """The API's services wired to fakes: real ChromaDB in `data_dir`, fake embedder and LLM."""
    from app.config import load_settings
    from app.generator import Generator
    from app.main import Services
    from app.pipeline import RagPipeline
    from app.retriever import Retriever
    from app.vector_store import VectorStore

    settings = load_settings({
        "DATA_DIR": str(data_dir), "CHUNK_SIZE": "200", "CHUNK_OVERLAP": "20",
        "MAX_UPLOAD_MB": "1", "LLM_API_KEY": "sk-secret-test",
    })
    embedder = FakeEmbedder()
    store = VectorStore(settings.chroma_dir, "test-collection")
    pipeline = RagPipeline(Retriever(embedder, store), Generator(llm))
    return Services(settings=settings, embedder=embedder, store=store, pipeline=pipeline)
