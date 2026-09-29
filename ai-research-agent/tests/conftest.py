"""Test doubles: a scripted LLM and a fake search provider.

Real LLMs are non-deterministic and cost money, so the offline tests replace them with a
fake that returns pre-written responses in order - letting us test the ORCHESTRATION logic
(the loop, tool dispatch, error handling, sources) exactly and for free.
"""

import copy
import json
from datetime import date

import pytest

from app.agent import ResearchAgent
from app.models import AgentEvent, LLMResponse, SearchResult, ToolCall
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class FakeLLM:
    """Returns scripted responses in order and records every request it receives."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": tools, "tool_choice": tool_choice}
        )
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


class FakeSearch:
    """Search provider that returns canned results, or raises a given error."""

    def __init__(self, results: list[SearchResult] | None = None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.results[:max_results]


def text(content: str) -> LLMResponse:
    """A scripted LLM response that is a final text answer."""
    return LLMResponse(content=content, finish_reason="stop")


def tool_call(name: str, arguments, call_id: str = "call_1") -> LLMResponse:
    """A scripted LLM response requesting one tool. `arguments` may be a dict or a raw string."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=raw)], finish_reason="tool_calls"
    )


SAMPLE_RESULTS = [
    SearchResult(title="RAG survey", url="https://example.org/rag-survey", snippet="RAG grounds answers in retrieved documents."),
    SearchResult(title="Hallucination study", url="https://example.org/hallucination", snippet="Retrieval reduces unsupported claims."),
    SearchResult(title="Eval frameworks", url="https://example.org/evals", snippet="RAGAS and others measure faithfulness."),
]


@pytest.fixture
def build_agent():
    """Factory: build_agent(llm, search, max_iterations=5) -> (agent, recorded_events)."""

    def _build(llm: FakeLLM, search: FakeSearch, max_iterations: int = 5):
        events: list[AgentEvent] = []
        registry = ToolRegistry()
        registry.register(WebSearchTool(provider=search, max_results=5))
        agent = ResearchAgent(
            llm=llm,
            tools=registry,
            max_iterations=max_iterations,
            on_event=events.append,
            today=date(2026, 9, 29),
        )
        return agent, events

    return _build
