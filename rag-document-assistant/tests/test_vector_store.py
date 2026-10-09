"""Tests for app/vector_store.py, using a real ChromaDB in a temporary folder and fake vectors."""

import pytest

from app.schemas import Chunk
from app.vector_store import VectorStore, collection_name_for
from helpers import FakeEmbedder


def make_chunks(doc_id: str = "doc1", filename: str = "paper.pdf", count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}:{i}", doc_id=doc_id, filename=filename, chunk_index=i,
            page_start=i + 1, page_end=i + 1, char_start=i * 100, char_end=i * 100 + 90,
            text=f"chunk {i} of {filename} about exam timetabling",
        )
        for i in range(count)
    ]


def vectors_for(chunks: list[Chunk]) -> list[list[float]]:
    return FakeEmbedder().embed_documents([chunk.text for chunk in chunks])


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", "test-collection")


def test_added_chunks_are_counted(store):
    chunks = make_chunks(count=3)
    store.add_chunks(chunks, vectors_for(chunks))

    assert store.count() == 3


def test_adding_the_same_chunks_again_does_not_duplicate_them(store):
    chunks = make_chunks(count=3)
    store.add_chunks(chunks, vectors_for(chunks))
    store.add_chunks(chunks, vectors_for(chunks))

    assert store.count() == 3


def test_find_document_by_content_hash(store):
    chunks = make_chunks("doc1", "paper.pdf", count=2)
    store.add_chunks(chunks, vectors_for(chunks))

    found = store.find_document("doc1")

    assert found is not None
    assert (found.filename, found.chunk_count) == ("paper.pdf", 2)
    assert store.find_document("unknown") is None


def test_list_documents_groups_chunks_by_document(store):
    for doc_id, filename, count in [("doc1", "b.pdf", 2), ("doc2", "a.pdf", 3)]:
        chunks = make_chunks(doc_id, filename, count)
        store.add_chunks(chunks, vectors_for(chunks))

    documents = store.list_documents()

    assert [(d.filename, d.chunk_count) for d in documents] == [("a.pdf", 3), ("b.pdf", 2)]


def test_delete_document_removes_only_that_document(store):
    for doc_id in ("doc1", "doc2"):
        chunks = make_chunks(doc_id, f"{doc_id}.pdf", 2)
        store.add_chunks(chunks, vectors_for(chunks))

    store.delete_document("doc1")

    assert store.find_document("doc1") is None
    assert store.find_document("doc2") is not None


def test_chunks_persist_on_disk(tmp_path):
    chunks = make_chunks(count=2)
    VectorStore(tmp_path / "chroma", "test-collection").add_chunks(chunks, vectors_for(chunks))

    reopened = VectorStore(tmp_path / "chroma", "test-collection")

    assert reopened.count() == 2


def test_reset_empties_the_collection_and_keeps_its_metadata(tmp_path):
    store = VectorStore(tmp_path / "chroma", "test-collection", metadata={"chunk_size": 1000})
    chunks = make_chunks(count=2)
    store.add_chunks(chunks, vectors_for(chunks))

    store.reset()

    assert store.count() == 0
    assert VectorStore(tmp_path / "chroma", "test-collection")._collection.metadata == {"chunk_size": 1000}


def test_each_configuration_gets_its_own_collection_name():
    name = collection_name_for("BAAI/bge-small-en-v1.5", 1000, 150)

    assert name == "BAAI-bge-small-en-v1.5_c1000_o150"
    assert name != collection_name_for("BAAI/bge-small-en-v1.5", 500, 50)
    assert name != collection_name_for("sentence-transformers/all-MiniLM-L6-v2", 1000, 150)


def test_chunks_and_vectors_must_match_in_number(store):
    with pytest.raises(ValueError):
        store.add_chunks(make_chunks(count=2), [[0.1] * 16])
