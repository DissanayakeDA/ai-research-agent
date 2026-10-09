"""Tests for app/llm_client.py using httpx.MockTransport: a fake HTTP server, no network, no cost."""

import httpx
import pytest

from app.llm_client import LLMClient, LLMError

OK_BODY = {
    "choices": [{"message": {"content": "Answer [1].", "reasoning": "thinking..."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
}


def make_client(handler, api_key: str = "sk-test-secret", **kwargs) -> tuple[LLMClient, list[float]]:
    waits: list[float] = []
    client = LLMClient(
        base_url="https://llm.example/v1", model="test-model", api_key=api_key,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=waits.append, **kwargs,
    )
    return client, waits


def test_sends_the_request_and_parses_the_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(200, json=OK_BODY)

    client, _ = make_client(handler, temperature=0.0, max_tokens=500)
    reply = client.chat([{"role": "user", "content": "hi"}])

    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-secret"
    assert b'"max_tokens":500' in seen["body"] and b'"temperature":0.0' in seen["body"]
    assert (reply.content, reply.reasoning, reply.prompt_tokens, reply.completion_tokens) == ("Answer [1].", "thinking...", 120, 30)


def test_no_authorization_header_without_a_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=OK_BODY)

    client, _ = make_client(handler, api_key="")
    assert client.chat([{"role": "user", "content": "hi"}]).content == "Answer [1]."


def test_rate_limit_waits_as_told_then_retries():
    responses = iter([httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}),
                      httpx.Response(200, json=OK_BODY)])
    client, waits = make_client(lambda request: next(responses))

    assert client.chat([{"role": "user", "content": "hi"}]).content == "Answer [1]."
    assert waits == [7.0]


def test_rejected_key_gives_a_clear_error_without_the_key():
    client, _ = make_client(lambda request: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))

    with pytest.raises(LLMError, match="LLM_API_KEY") as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert "sk-test-secret" not in str(excinfo.value)


def test_unreachable_server_fails_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, waits = make_client(handler, max_retries=2)

    with pytest.raises(LLMError, match="Could not reach"):
        client.chat([{"role": "user", "content": "hi"}])
    assert waits == [1, 2]  # backed off before each retry


def test_empty_answer_cut_off_by_token_limit_is_explained():
    body = {"choices": [{"message": {"content": "", "reasoning": "long thinking"}, "finish_reason": "length"}]}
    client, _ = make_client(lambda request: httpx.Response(200, json=body))

    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        client.chat([{"role": "user", "content": "hi"}])


def test_provider_error_message_is_passed_on():
    client, _ = make_client(lambda request: httpx.Response(400, json={"error": {"message": "model not found"}}))

    with pytest.raises(LLMError, match="model not found"):
        client.chat([{"role": "user", "content": "hi"}])
