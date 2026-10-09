"""Tests for app/retriever.py and VectorStore.query, with a real ChromaDB and the fake embedder.

FakeEmbedder gives texts that share words similar vectors, so "tabu search" is closest to
the chunk about tabu search - enough to test ranking, metadata, de-duplication and limits.
"""

import math

import pytest

from app.retriever import Retriever
from app.schemas import Chunk
from app.vector_store import VectorStore
from helpers import FakeEmbedder

TEXTS = [
    "genetic algorithm with crossover and mutation for exam timetabling",
    "tabu search keeps a tabu list in memory to avoid cycling",
    "integer programming assigns each exam to exactly one time slot",
    "the library opens at nine on weekdays",
]


def chunk(doc_id: str, index: int, text: str, filename: str = "paper.pdf", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{index}", doc_id=doc_id, filename=filename, chunk_index=index,
        page_start=page, page_end=page + 1, char_start=0, char_end=len(text), text=text,
    )


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", "test-collection")


@pytest.fixture
def retriever(store):
    chunks = [chunk("doc1", i, text, page=i + 1) for i, text in enumerate(TEXTS)]
    embedder = FakeEmbedder()
    store.add_chunks(chunks, embedder.embed_documents(TEXTS))
    return Retriever(embedder, store)


def test_most_similar_chunk_comes_first(retriever):
    results = retriever.retrieve("tabu search memory", top_k=3)

    assert "tabu search" in results[0].text
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_returns_at_most_top_k(retriever):
    assert len(retriever.retrieve("exam", top_k=2)) == 2


def test_identical_text_scores_about_one(retriever):
    best = retriever.retrieve(TEXTS[2], top_k=1)[0]

    assert math.isclose(best.score, 1.0, abs_tol=1e-4)


def test_results_keep_filename_and_pages(retriever):
    best = retriever.retrieve("integer programming time slot", top_k=1)[0]

    assert (best.filename, best.page_start, best.page_end, best.chunk_index) == ("paper.pdf", 3, 4, 2)


def test_duplicate_passages_appear_once(store):
    embedder = FakeEmbedder()
    same = "tabu search keeps a tabu list in memory"
    chunks = [chunk("doc1", 0, same, "a.pdf"), chunk("doc2", 0, same, "copy.pdf"), chunk("doc1", 1, TEXTS[0], "a.pdf")]
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))

    results = Retriever(embedder, store).retrieve("tabu search", top_k=2)

    assert [r.text for r in results] == [same, TEXTS[0]]  # still 2 results after dropping the copy


def test_min_score_drops_weak_matches(retriever):
    results = retriever.retrieve(TEXTS[1], top_k=4, min_score=0.99)

    assert [r.text for r in results] == [TEXTS[1]]


def test_top_k_larger_than_the_index_returns_everything(retriever):
    assert len(retriever.retrieve("exam", top_k=50)) == len(TEXTS)


def test_empty_index_returns_nothing(store):
    assert Retriever(FakeEmbedder(), store).retrieve("anything", top_k=5) == []


@pytest.mark.parametrize("question, top_k", [("", 5), ("   ", 5), ("x" * 2001, 5), ("valid question", 0)])
def test_invalid_requests_are_rejected(retriever, question, top_k):
    with pytest.raises(ValueError):
        retriever.retrieve(question, top_k)
