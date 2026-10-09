"""Tests for app/embeddings.py with the REAL model (slower; downloads ~130 MB on first run).

Skipped by default. Run them with:  pytest -m model
"""

import math

import pytest

from app.config import load_settings
from app.embeddings import Embedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder():
    return Embedder(load_settings({}).embedding_model)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_vectors_have_the_model_dimension_and_unit_length(embedder):
    vectors = embedder.embed_documents(["Exam timetabling", "Genetic algorithms"])

    assert [len(v) for v in vectors] == [embedder.dimension, embedder.dimension]
    assert all(math.isclose(math.sqrt(dot(v, v)), 1.0, abs_tol=1e-4) for v in vectors)


def test_a_paraphrase_is_closer_than_unrelated_text(embedder):
    question = embedder.embed_query("How long do I have to postpone an exam?")
    paraphrase, unrelated = embedder.embed_documents(
        ["Students may defer an examination within 7 days.", "The library opens at 9 am on weekdays."]
    )

    assert dot(question, paraphrase) > dot(question, unrelated)


def test_long_text_exceeds_the_token_limit(embedder):
    assert embedder.count_tokens(["timetable " * 1000])[0] > embedder.max_tokens


def test_empty_input_gives_no_vectors(embedder):
    assert embedder.embed_documents([]) == []
