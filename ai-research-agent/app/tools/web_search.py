"""The `web_search` tool and the search provider behind it.

Two layers, so the provider can be swapped without touching the agent:

    WebSearchTool   - what the LLM sees: name, description, argument schema.
                      Knows nothing about Tavily.
    TavilySearch    - one concrete provider. To switch to Brave, Serper, SearXNG, etc.,
                      write another class with the same `search(query, max_results)` method
                      and pass it to WebSearchTool instead.
"""

from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from app.models import SearchResult, ToolResult
from app.tools.registry import ToolError

SNIPPET_MAX_CHARS = 500  # keep tool output small: every character is re-sent to the LLM each turn


class SearchError(ToolError):
    """The search provider failed (network error, bad key, quota exceeded, ...)."""


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


# --- The provider ----------------------------------------------------------------------

class TavilySearch:
    """Search provider backed by the Tavily API (https://docs.tavily.com)."""

    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 20.0, http_client: httpx.Client | None = None):
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http_client or httpx.Client(timeout=timeout)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        payload = {"query": query, "max_results": max_results, "search_depth": "basic"}
        try:
            response = self._http.post(self.URL, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise SearchError(f"could not reach the search API: {exc}") from exc

        if response.status_code in (401, 403):
            raise SearchError("search API rejected the API key (check SEARCH_API_KEY)")
        if response.status_code == 429:
            raise SearchError("search API rate limit or quota exceeded")
        if response.status_code >= 400:
            raise SearchError(f"search API returned HTTP {response.status_code}")

        try:
            items = response.json().get("results") or []
        except ValueError as exc:
            raise SearchError("search API returned invalid JSON") from exc

        return [
            SearchResult(
                title=(item.get("title") or "Untitled").strip(),
                url=item["url"],
                snippet=_shorten(item.get("content") or ""),
            )
            for item in items
            if item.get("url")
        ]


def _shorten(text: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# --- The tool (what the LLM sees) ----------------------------------------------------------

class WebSearchArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="A focused web search query, e.g. 'RAG evaluation frameworks 2026'.",
    )


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web for up-to-date information. Use it for recent events, current or "
        "'latest' information, specific facts, statistics, versions, or comparisons you are not "
        "certain about. Returns numbered results with title, URL and a short snippet. "
        "Do not use it for stable, well-known concepts you can explain confidently."
    )
    args_model = WebSearchArgs

    def __init__(self, provider: SearchProvider, max_results: int = 5):
        self.provider = provider
        self.max_results = max_results

    def run(self, args: WebSearchArgs) -> ToolResult:
        results = self.provider.search(args.query, self.max_results)  # SearchError propagates
        if not results:
            return ToolResult(
                ok=True,
                data={
                    "query": args.query,
                    "results": [],
                    "note": "No results found. Try a different query, or answer from general "
                    "knowledge and say that the search found nothing.",
                },
            )
        return ToolResult(ok=True, data={"query": args.query}, sources=results)
