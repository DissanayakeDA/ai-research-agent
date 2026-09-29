"""Offline tests for the agent loop and its components. Run with: pytest -v"""

import json

import httpx
import pytest

from app.config import ConfigError, load_settings
from app.llm import LLMClient, LLMError
from app.models import SearchResult
from app.response import SourceTracker, build_final_response
from app.tools.web_search import SearchError, TavilySearch
from tests.conftest import SAMPLE_RESULTS, FakeLLM, FakeSearch, text, tool_call


def tool_messages(llm_call: dict) -> list[dict]:
    return [m for m in llm_call["messages"] if m["role"] == "tool"]


# --- 1. A question that does not require search ----------------------------------------

def test_answers_directly_without_search(build_agent):
    llm = FakeLLM([text("RAG stands for retrieval-augmented generation.")])
    search = FakeSearch(SAMPLE_RESULTS)
    agent, events = build_agent(llm, search)

    result = agent.run("What is RAG?")

    assert result.answer == "RAG stands for retrieval-augmented generation."
    assert not result.used_search
    assert search.queries == []                    # the tool was never executed
    assert result.cited_sources == []
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"][0]["function"]["name"] == "web_search"  # tool WAS offered
    assert [e.type for e in events] == ["user_question", "llm_request", "llm_response", "final_answer"]


# --- 2. A question that requires search --------------------------------------------------

def test_searches_then_answers_with_verified_sources(build_agent):
    llm = FakeLLM([
        tool_call("web_search", {"query": "RAG hallucination reduction"}),
        text("RAG grounds answers in retrieved text [1], which reduces unsupported claims [2]."),
    ])
    search = FakeSearch(SAMPLE_RESULTS)
    agent, _ = build_agent(llm, search)

    result = agent.run("How does RAG reduce hallucinations?")

    assert search.queries == ["RAG hallucination reduction"]
    assert result.used_search
    assert [s.url for s in result.cited_sources] == [
        "https://example.org/rag-survey",
        "https://example.org/hallucination",
    ]
    assert len(result.retrieved_sources) == 3
    assert result.warnings == []

    # The protocol: 2nd LLM call must contain the assistant's tool request, then the tool result
    # linked to it by tool_call_id, with numbered sources.
    second_call = llm.calls[1]["messages"]
    assert second_call[-2]["role"] == "assistant"
    assert second_call[-2]["tool_calls"][0]["id"] == "call_1"
    assert second_call[-1]["role"] == "tool"
    assert second_call[-1]["tool_call_id"] == "call_1"
    payload = json.loads(second_call[-1]["content"])
    assert [r["source_id"] for r in payload["results"]] == [1, 2, 3]


# --- 3. Search API failure ---------------------------------------------------------------

def test_search_failure_is_reported_to_llm_not_crash(build_agent):
    llm = FakeLLM([
        tool_call("web_search", {"query": "RAG evaluation 2026"}),
        text("The web search failed, so this is not verified. From general knowledge: ..."),
    ])
    search = FakeSearch(error=SearchError("search API returned HTTP 503"))
    agent, events = build_agent(llm, search)

    result = agent.run("Latest RAG evaluation developments in 2026?")

    error_payload = json.loads(tool_messages(llm.calls[1])[0]["content"])
    assert "HTTP 503" in error_payload["error"]
    assert result.cited_sources == []
    assert result.used_search
    failed = [e for e in events if e.type == "tool_result"][0]
    assert failed.data["ok"] is False


def test_tavily_http_errors_become_search_errors():
    def failing(request):
        return httpx.Response(500, json={"detail": "boom"})

    provider = TavilySearch(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(failing)))
    with pytest.raises(SearchError, match="HTTP 500"):
        provider.search("anything", 5)


def test_tavily_network_errors_become_search_errors():
    def unreachable(request):
        raise httpx.ConnectError("connection refused")

    provider = TavilySearch(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(unreachable)))
    with pytest.raises(SearchError, match="could not reach"):
        provider.search("anything", 5)


def test_tavily_parses_results_into_search_results():
    def ok(request):
        assert request.headers["Authorization"] == "Bearer test"
        return httpx.Response(200, json={"results": [
            {"title": "A", "url": "https://a.example", "content": "snippet  with\n spaces"},
            {"title": "No URL", "content": "skipped"},
        ]})

    provider = TavilySearch(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(ok)))
    assert provider.search("q", 5) == [SearchResult(title="A", url="https://a.example", snippet="snippet with spaces")]


# --- 4. Invalid tool calls ----------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_arguments, expected_error",
    [
        ("{not json", "not valid JSON"),
        ({"q": "wrong field name"}, "query: Field required"),
        ({"query": ""}, "query: String should have at least 2 characters"),
    ],
)
def test_invalid_tool_arguments_are_reported_and_model_can_retry(build_agent, bad_arguments, expected_error):
    llm = FakeLLM([
        tool_call("web_search", bad_arguments, call_id="call_bad"),
        tool_call("web_search", {"query": "RAG hallucinations"}, call_id="call_good"),
        text("Retrieval grounds the answer [1]."),
    ])
    search = FakeSearch(SAMPLE_RESULTS)
    agent, _ = build_agent(llm, search)

    result = agent.run("How does RAG reduce hallucinations?")

    first_error = json.loads(tool_messages(llm.calls[1])[0]["content"])
    assert expected_error in first_error["error"]
    assert search.queries == ["RAG hallucinations"]   # only the valid call reached the provider
    assert [s.id for s in result.cited_sources] == [1]


def test_unknown_tool_is_reported_to_llm(build_agent):
    llm = FakeLLM([
        tool_call("read_url", {"url": "https://example.org"}),
        text("I can't read URLs yet."),
    ])
    agent, _ = build_agent(llm, FakeSearch(SAMPLE_RESULTS))

    agent.run("Summarise https://example.org")

    error = json.loads(tool_messages(llm.calls[1])[0]["content"])["error"]
    assert "Unknown tool 'read_url'" in error
    assert "web_search" in error  # tells the model what it CAN use


# --- 5. Empty search results --------------------------------------------------------------

def test_empty_search_results(build_agent):
    llm = FakeLLM([
        tool_call("web_search", {"query": "obscure topic xyz"}),
        text("The search found nothing. From general knowledge: ..."),
    ])
    agent, events = build_agent(llm, FakeSearch(results=[]))

    result = agent.run("Tell me about obscure topic xyz")

    payload = json.loads(tool_messages(llm.calls[1])[0]["content"])
    assert payload["results"] == []
    assert "No results found" in payload["note"]
    assert result.retrieved_sources == [] and result.cited_sources == []
    assert [e.data["result_count"] for e in events if e.type == "tool_result"] == [0]


# --- Iteration limit ----------------------------------------------------------------------

def test_iteration_limit_disables_tools_and_forces_an_answer(build_agent):
    llm = FakeLLM([
        tool_call("web_search", {"query": "first"}, call_id="c1"),
        tool_call("web_search", {"query": "second"}, call_id="c2"),
        text("Best answer with what I found [1]."),
    ])
    agent, events = build_agent(llm, FakeSearch(SAMPLE_RESULTS), max_iterations=3)

    result = agent.run("Compare the latest RAG evaluation frameworks")

    assert [c["tool_choice"] for c in llm.calls] == ["auto", "auto", "none"]
    assert "Tool-use limit reached" in llm.calls[2]["messages"][-1]["content"]
    assert result.stopped_reason == "max_iterations"
    assert result.iterations == 3
    assert any("iteration limit" in w for w in result.warnings)
    assert any(e.type == "iteration_limit" for e in events)


# --- Source handling ----------------------------------------------------------------------

def test_invented_citations_and_urls_are_flagged_not_listed():
    tracker = SourceTracker()
    tracker.add(SAMPLE_RESULTS[:2])

    result = build_final_response(
        "q",
        "Claim A [1]. Claim B [1, 7]. See https://made-up.example/paper.",
        tracker,
        iterations=2,
        tool_calls_made=1,
    )

    assert [s.id for s in result.cited_sources] == [1]
    assert any("[7]" in w for w in result.warnings)
    assert any("made-up.example" in w for w in result.warnings)


def test_sources_are_numbered_globally_and_deduplicated_across_searches():
    tracker = SourceTracker()
    first = tracker.add(SAMPLE_RESULTS[:2])
    second = tracker.add([SAMPLE_RESULTS[1], SAMPLE_RESULTS[2]])

    assert [s.id for s in first] == [1, 2]
    assert [s.id for s in second] == [2, 3]   # same URL keeps its number
    assert len(tracker.all()) == 3


# --- Configuration & LLM API failures -----------------------------------------------------

def test_missing_api_keys_raise_clear_config_error():
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env={"OPENAI_API_KEY": "sk-test"})
    assert "SEARCH_API_KEY" in str(exc_info.value)
    assert "OPENAI_API_KEY" not in str(exc_info.value)


def test_settings_parse_optional_values():
    settings = load_settings(env={"OPENAI_API_KEY": "a", "SEARCH_API_KEY": "b", "MAX_ITERATIONS": "3", "DEBUG": "true"})
    assert settings.max_iterations == 3 and settings.debug is True


def test_llm_auth_error_raises_llm_error_with_provider_message():
    def unauthorized(request):
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    llm = LLMClient("bad-key", "test-model", http_client=httpx.Client(transport=httpx.MockTransport(unauthorized)))
    with pytest.raises(LLMError, match="HTTP 401: Incorrect API key provided"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_llm_retries_transient_errors_then_parses_tool_call(monkeypatch):
    monkeypatch.setattr("app.llm.time.sleep", lambda seconds: None)
    responses = iter([
        httpx.Response(503),
        httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_9", "type": "function",
                            "function": {"name": "web_search", "arguments": "{\"query\": \"rag\"}"}}],
        }}]}),
    ])
    llm = LLMClient("key", "test-model", http_client=httpx.Client(transport=httpx.MockTransport(lambda r: next(responses))))

    response = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert response.tool_calls[0].name == "web_search"
    assert response.tool_calls[0].arguments == '{"query": "rag"}'


def test_llm_failure_propagates_out_of_agent(build_agent):
    class BrokenLLM:
        def chat(self, *args, **kwargs):
            raise LLMError("LLM API returned HTTP 500")

    agent, _ = build_agent(BrokenLLM(), FakeSearch(SAMPLE_RESULTS))
    with pytest.raises(LLMError):
        agent.run("anything")
