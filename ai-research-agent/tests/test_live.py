"""Live tests against the real LLM and search APIs.

Skipped by default. They cost tokens and search credits, and they are NON-deterministic:
the model decides whether to search, so an occasional different decision is expected.
Think of them as a tiny behaviour check, not a guarantee.

    RUN_LIVE_TESTS=1 pytest -v -m live          (bash)
    $env:RUN_LIVE_TESTS=1; pytest -v -m live    (PowerShell)
"""

import os

import pytest

from app.agent import ResearchAgent
from app.config import load_settings
from app.llm import LLMClient
from app.tools import build_default_registry

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call real APIs"),
]


@pytest.fixture(scope="module")
def agent():
    settings = load_settings()
    return ResearchAgent(
        llm=LLMClient(settings.openai_api_key, settings.llm_model, settings.llm_base_url),
        tools=build_default_registry(settings),
        max_iterations=settings.max_iterations,
    )


def test_live_stable_concept_is_answered_without_search(agent):
    result = agent.run("In one short paragraph: what does the acronym RAG stand for in LLM applications?")
    assert not result.used_search, "model chose to search for a stable concept"
    assert "retrieval" in result.answer.lower()


def test_live_recent_question_triggers_search_with_real_sources(agent):
    result = agent.run("What are the latest developments in RAG evaluation in 2026?")
    assert result.used_search, "model answered a 'latest in 2026' question without searching"
    assert result.cited_sources, f"no sources cited; warnings: {result.warnings}"
    retrieved_urls = {s.url for s in result.retrieved_sources}
    assert all(s.url in retrieved_urls for s in result.cited_sources)
