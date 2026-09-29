"""A deliberately thin client for any OpenAI-compatible Chat Completions API.

We call the HTTP endpoint directly with httpx instead of using an SDK, so the exact
request/response JSON for tool calling is visible:

Request  (POST {base_url}/chat/completions)
    {
      "model": "...",
      "messages": [{"role": "system", ...}, {"role": "user", ...}, ...],
      "tools": [{"type": "function", "function": {"name", "description", "parameters"}}],
      "tool_choice": "auto"          # "auto" = model decides, "none" = must answer in text
    }

Response
    {"choices": [{"message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [{"id": "call_abc", "type": "function",
                        "function": {"name": "web_search", "arguments": "{\"query\": \"...\"}"}}]
     }, "finish_reason": "tool_calls"}]}

The LLM never runs anything. It only returns a *request* to run a tool; our Python code
decides whether and how to execute it.
"""

import time
from typing import Any

import httpx

from app.models import LLMResponse, ToolCall


class LLMError(Exception):
    """The LLM API could not be reached or returned an unusable response."""


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,  # injectable for tests
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_retries = max_retries
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http_client or httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Send the conversation (and tool definitions) and return the model's next move."""
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        data = self._post_with_retries(payload)
        return self._parse(data)

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            is_last_attempt = attempt == self.max_retries
            try:
                response = self._http.post(self.url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:  # DNS failure, refused connection, timeout...
                if is_last_attempt:
                    raise LLMError(f"Could not reach the LLM API at {self.url}: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code in RETRYABLE_STATUS and not is_last_attempt:
                time.sleep(2**attempt)  # rate limit or server hiccup: back off 1s, 2s, ...
                continue
            if response.status_code >= 400:
                raise LLMError(
                    f"LLM API returned HTTP {response.status_code}: {_error_message(response)}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise LLMError("LLM API returned a response that is not valid JSON") from exc

        raise LLMError("LLM API request failed after retries")  # unreachable, keeps type-checkers happy

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            tool_calls = [
                ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=call["function"].get("arguments") or "{}",
                )
                for call in message.get("tool_calls") or []
            ]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape from LLM API: {str(data)[:300]}") from exc

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
        )


def _error_message(response: httpx.Response) -> str:
    """Pull the human-readable error out of an OpenAI-style error body, if there is one."""
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:300] or response.reason_phrase
