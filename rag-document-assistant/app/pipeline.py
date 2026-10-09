"""The question-answering pipeline: question -> retrieve passages -> generate a cited answer.

scripts/ask.py uses it now; the FastAPI app will hold one instance in Phase 6.
"""

import time

from app.config import Settings
from app.embeddings import Embedder
from app.generator import Generator
from app.llm_client import LLMClient
from app.retriever import Retriever
from app.schemas import Answer
from app.vector_store import open_store


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, top_k: int, min_score: float | None = None) -> Answer:
        started = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k, min_score)
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)
        answer = self.generator.answer(question, chunks)
        return answer.model_copy(update={"retrieval_ms": retrieval_ms})


def build_llm_client(settings: Settings) -> LLMClient:
    return LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


def build_pipeline(settings: Settings) -> RagPipeline:
    """Create the real components from settings. Loads the embedding model (a few seconds)."""
    retriever = Retriever(Embedder(settings.embedding_model), open_store(settings))
    return RagPipeline(retriever, Generator(build_llm_client(settings)))
