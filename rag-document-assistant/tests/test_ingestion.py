"""Tests for app/ingestion.py: the full load -> chunk -> embed -> store path, with a fake embedder."""

import pytest

from app.config import load_settings
from app.ingestion import ingest_pdf
from app.vector_store import VectorStore
from helpers import FakeEmbedder, make_pdf

PAGE = (
    "Exam timetabling assigns every exam to a room and a time slot.\n"
    "Hard constraints must never be violated, for example no student\n"
    "may sit two exams at the same time. Soft constraints, such as\n"
    "spreading a student's exams out, should be satisfied where possible."
)


@pytest.fixture
def settings():
    return load_settings({"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "20"})  # small, so a page gives several chunks


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", "test-collection")


def test_new_pdf_is_indexed(settings, store):
    result = ingest_pdf(make_pdf(PAGE, PAGE), "paper.pdf", settings, FakeEmbedder(), store)

    assert result.status == "indexed"
    assert result.pages == 2
    assert result.chunks > 2
    assert store.count() == result.chunks
    assert store.find_document(result.doc_id).filename == "paper.pdf"


def test_same_file_again_is_skipped_before_any_embedding(settings, store):
    data = make_pdf(PAGE)
    embedder = FakeEmbedder()

    first = ingest_pdf(data, "paper.pdf", settings, embedder, store)
    second = ingest_pdf(data, "paper.pdf", settings, embedder, store)

    assert second.status == "skipped"
    assert second.message == "already indexed"
    assert embedder.calls == 1
    assert store.count() == first.chunks


def test_same_content_under_another_name_is_recognised(settings, store):
    data = make_pdf(PAGE)
    ingest_pdf(data, "paper.pdf", settings, FakeEmbedder(), store)

    result = ingest_pdf(data, "copy of paper.pdf", settings, FakeEmbedder(), store)

    assert result.status == "skipped"
    assert "paper.pdf" in result.message


def test_invalid_file_fails_and_stores_nothing(settings, store):
    result = ingest_pdf(b"not a pdf at all", "fake.pdf", settings, FakeEmbedder(), store)

    assert result.status == "failed"
    assert "not a PDF" in result.message
    assert store.count() == 0


def test_embedding_failure_stores_nothing_and_can_be_retried(settings, store):
    data = make_pdf(PAGE)

    failed = ingest_pdf(data, "paper.pdf", settings, FakeEmbedder(fail=True), store)
    retried = ingest_pdf(data, "paper.pdf", settings, FakeEmbedder(), store)

    assert failed.status == "failed"
    assert "simulated failure" in failed.message
    assert retried.status == "indexed"  # not wrongly "skipped" because of a half-written first attempt


def test_chunks_longer_than_the_token_limit_are_counted(settings, store):
    result = ingest_pdf(make_pdf(PAGE), "paper.pdf", settings, FakeEmbedder(max_tokens=5), store)

    assert result.status == "indexed"
    assert result.truncated_chunks == result.chunks  # every chunk has more than 5 words
