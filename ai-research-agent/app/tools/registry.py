"""The Tool Registry: the single place that knows which tools exist and how to run them.

It has two jobs, one for each direction of the conversation:

1. Describe tools TO the LLM  -> `schemas()` produces the JSON Schema list sent as `tools`.
2. Execute tools FOR the LLM  -> `execute()` takes the model's raw request (a name and a JSON
   string), validates it, runs the tool, and always returns a ToolResult - never raises for
   problems the model caused, because the model can often fix them if we tell it what went wrong.
"""

import json
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from app.models import ToolResult


class ToolError(Exception):
    """A tool failed in an expected way (e.g. the search API is down). Reported to the LLM."""


class Tool(Protocol):
    """What every tool must provide. Any class with these attributes qualifies."""

    name: str
    description: str        # The LLM reads this to decide WHEN to use the tool.
    args_model: type[BaseModel]  # Pydantic model = argument validation + JSON Schema.

    def run(self, args: BaseModel) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions in the OpenAI `tools` format. This is all the LLM knows about a tool."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        """Validate and run one tool call. Every failure becomes an error ToolResult."""
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"Unknown tool '{name}'. Available tools: {', '.join(self.names)}.")

        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return _error(f"Arguments for '{name}' are not valid JSON ({exc.msg}): {raw_arguments!r}")

        try:
            args = tool.args_model.model_validate(parsed)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            return _error(f"Invalid arguments for '{name}': {problems}")

        try:
            return tool.run(args)
        except ToolError as exc:
            return _error(f"Tool '{name}' failed: {exc}")


def _error(message: str) -> ToolResult:
    return ToolResult(ok=False, data={"error": message}, error=message)
