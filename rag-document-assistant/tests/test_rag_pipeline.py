"""End-to-end test of the question-answering pipeline: real ChromaDB, fake embedder and fake LLM.

It checks the wiring - the passages the retriever finds are exactly the ones numbered in the
prompt and mapped back to sources - without a model download or an API call.
"""

import pytest

from app.config import load_settings
from app.generator import Generator
from app.ingestion import ingest_pdf
from app.pipeline import RagPipeline
from app.retriever import Retriever
from app.vector_store import VectorStore
from helpers import FakeEmbedder, FakeLLM, make_pdf

PAGE_1 = "Tabu search keeps a tabu list of recent moves\nso the search does not cycle back."
PAGE_2 = "Integer programming assigns every exam\nto exactly one time slot."


@pytest.fixture
def store(tmp_path):
    store = VectorStore(tmp_path / "chroma", "test-collection")
    settings = load_settings({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "0"})  # one chunk per page here
    result = ingest_pdf(make_pdf(PAGE_1, PAGE_2), "methods.pdf", settings, FakeEmbedder(), store)
    assert result.status == "indexed"
    return store


def test_question_flows_from_retrieval_to_a_cited_answer(store):
    llm = FakeLLM("Tabu search keeps a list of recent moves [1].")
    pipeline = RagPipeline(Retriever(FakeEmbedder(), store), Generator(llm))

    answer = pipeline.ask("tabu search recent moves", top_k=2)

    assert answer.found
    assert answer.sources[0].filename == "methods.pdf"
    assert answer.sources[0].page_start == 1
    assert "tabu list" in answer.sources[0].text
    prompt = llm.calls[0][1]["content"]
    assert '<passage id="1" source="methods.pdf" pages="1">' in prompt
    assert answer.retrieval_ms >= 0


def test_nothing_relevant_means_no_llm_call(store):
    llm = FakeLLM()
    pipeline = RagPipeline(Retriever(FakeEmbedder(), store), Generator(llm))

    answer = pipeline.ask("capital of France", top_k=2, min_score=0.99)

    assert not answer.found
    assert llm.calls == []
