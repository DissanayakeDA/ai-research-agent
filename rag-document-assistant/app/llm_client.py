"""A thin client for any OpenAI-compatible Chat Completions API (Groq, OpenAI, Ollama...).

We call the HTTP endpoint directly with httpx, as in the research-agent project, so the
exact request and response are visible:

    POST {base_url}/chat/completions
    {"model": "...", "messages": [{"role": "system", ...}, {"role": "user", ...}],
     "temperature": 0.0, "max_tokens": 1024}

    -> {"choices": [{"message": {"content": "answer...", "reasoning": "..."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 250}}

"reasoning" only exists for reasoning models such as gpt-oss: their hidden thinking,
which also counts towards completion tokens.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_WAIT_SECONDS = 60.0


class LLMError(Exception):
    """The LLM could not be reached or returned an unusable answer. The message is safe to show."""


class LLMReply(BaseModel):
    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,  # injectable for tests
        sleep: Callable[[float], None] = time.sleep,  # injectable for tests
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # A local server (Ollama) needs no key, so only send the header when there is one.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http_client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    def chat(self, messages: list[dict[str, str]]) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        started = time.perf_counter()
        reply = self._parse(self._post_with_retries(payload))
        logger.info(
            "LLM %s answered in %.1fs (%d prompt + %d completion tokens)",
            self.model, time.perf_counter() - started, reply.prompt_tokens, reply.completion_tokens,
        )
        return reply

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            last_attempt = attempt == self.max_retries
            try:
                response = self._http.post(self.url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:  # refused connection, DNS failure, timeout...
                if last_attempt:
                    raise LLMError(
                        f"Could not reach the LLM at {self.url} ({exc.__class__.__name__}). "
                        "Check your internet connection, LLM_BASE_URL, or that the local server is running."
                    ) from exc
                self._sleep(2**attempt)
                continue

            if response.status_code in RETRYABLE_STATUS and not last_attempt:
                wait = _retry_after(response) or 2**attempt
                logger.warning("LLM returned HTTP %d; retrying in %.1fs", response.status_code, wait)
                self._sleep(min(wait, MAX_WAIT_SECONDS))
                continue
            if response.status_code in (401, 403):
                raise LLMError(f"The LLM provider rejected the API key (HTTP {response.status_code}). Check LLM_API_KEY.")
            if response.status_code >= 400:
                raise LLMError(f"LLM request failed with HTTP {response.status_code}: {_error_message(response)}")
            try:
                return response.json()
            except ValueError as exc:
                raise LLMError("The LLM returned a response that is not valid JSON.") from exc
        raise LLMError("LLM request failed after retries.")  # unreachable; keeps type checkers happy

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMReply:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response from the LLM: {str(data)[:300]}") from exc

        content = (message.get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        if not content:
            hint = " The token limit was reached first - raise LLM_MAX_TOKENS." if finish_reason == "length" else ""
            raise LLMError(f"The LLM returned an empty answer (finish_reason={finish_reason}).{hint}")

        usage = data.get("usage") or {}
        return LLMReply(
            content=content,
            reasoning=message.get("reasoning"),
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds the provider asked us to wait (rate limits send a Retry-After header)."""
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _error_message(response: httpx.Response) -> str:
    """The provider's human-readable error, from an OpenAI-style error body if there is one."""
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:300] or response.reason_phrase
