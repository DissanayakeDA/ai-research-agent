"""Tools the agent can use. Add new tools here to make them available to the LLM."""

from app.config import Settings
from app.tools.registry import Tool, ToolError, ToolRegistry
from app.tools.web_search import TavilySearch, WebSearchTool

__all__ = ["Tool", "ToolError", "ToolRegistry", "build_default_registry"]


def build_default_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        WebSearchTool(
            provider=TavilySearch(api_key=settings.search_api_key),
            max_results=settings.max_search_results,
        )
    )
    return registry
