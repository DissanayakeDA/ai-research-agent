"""Question time, step 1: find the chunks most similar to the question (the "R" in RAG).

    1. Embed the question with the SAME model used for the chunks, so both live in one
       vector space and can be compared.
    2. Ask ChromaDB for the nearest chunk vectors (cosine similarity).
    3. Drop duplicate passages and keep the best `top_k`.

Retrieval always returns the k *closest* chunks - even when none of them is relevant.
Ask about the capital of France and you still get k chunks about timetabling, just with
lower scores. That is why the generator (Phase 5) must be allowed to answer "the documents
don't say", and why `min_score` exists.
"""

import logging
import time

from app.embeddings import Embedder
from app.schemas import RetrievedChunk
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000  # the model reads at most 512 tokens anyway


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int, min_score: float | None = None) -> list[RetrievedChunk]:
        """The `top_k` most similar distinct chunks, best first.

        `min_score` (optional) drops chunks whose similarity is below it, so a question the
        documents can't answer may return fewer chunks - or none.
        """
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"The question is longer than {MAX_QUESTION_CHARS} characters.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        started = time.perf_counter()
        vector = self._embedder.embed_query(question)
        # Ask for extra candidates, so that removing duplicates still leaves top_k results.
        candidates = self._store.query(vector, n_results=top_k * 2)

        results: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for candidate in candidates:  # sorted by score, best first
            if min_score is not None and candidate.score < min_score:
                break  # every remaining candidate scores even lower
            text_key = " ".join(candidate.text.split()).lower()
            if text_key in seen_texts:  # same passage stored twice, e.g. in two near-identical files
                continue
            seen_texts.add(text_key)
            results.append(candidate.model_copy(update={"rank": len(results) + 1}))
            if len(results) == top_k:
                break

        logger.info(
            "Retrieved %d chunk(s) in %.0f ms (top score %s)",
            len(results), (time.perf_counter() - started) * 1000,
            f"{results[0].score:.3f}" if results else "-",
        )
        return results
