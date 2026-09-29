"""The Agent Orchestrator: the loop that turns an LLM into an agent.

    messages = [system prompt, user question]
    repeat (at most max_iterations times):
        response = llm(messages, tools)            # 1-2. send question + available tools
        messages.append(response)                  #      remember what the model said
        if response has no tool calls:             # 7.   plain text = the final answer
            return final answer
        for each tool call:                        # 3-4. the model requested a tool
            result = registry.execute(call)        # 5.   WE run it in Python
            messages.append(tool result)           # 6.   send the result back to the model

The LLM makes every decision (search or not, what to search, when to stop); this file
only carries out those decisions, enforces limits, and reports what happened.
"""

import json
from datetime import date
from typing import Any, Callable

from app.llm import LLMClient
from app.models import AgentEvent, AgentResult, ToolCall, ToolResult
from app.response import SourceTracker, build_final_response
from app.tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are a research assistant. Today's date is {today}.

You can call a web_search tool.

Deciding whether to search:
- Search when the question needs recent or current information ("latest", a specific year, \
new releases), specific facts or figures, comparisons of current tools or products, or \
anything you are not confident about.
- Answer directly, without searching, for stable and well-established concepts you can \
explain confidently.
- If the first results are not enough, you may search again with a better query.

Writing the answer:
- Search results have a numeric source_id. When a sentence uses information from a result, \
cite it inline like [1] or [2][3].
- Only cite source_ids that appear in the tool results. Never invent sources or URLs.
- Do not write a Sources or References section; the application adds it automatically.
- If you add information that did not come from the search results, put it in a separate \
final paragraph that starts with "From general knowledge:".
- If a search fails or finds nothing useful, say so briefly, then answer from general \
knowledge if you can, clearly labelled as such.
- Be concise: a few short paragraphs or a short bulleted list."""

FORCE_ANSWER_NOTE = (
    "Tool-use limit reached. Do not call any more tools. Write your final answer now, using the "
    "information gathered so far, and say if anything could not be verified."
)

EventHandler = Callable[[AgentEvent], None]


class ResearchAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        max_iterations: int = 5,
        on_event: EventHandler | None = None,
        today: date | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.on_event = on_event or (lambda event: None)
        self.today = today or date.today()

    def run(self, question: str) -> AgentResult:
        """Answer one question. Raises LLMError if the LLM API itself fails."""
        sources = SourceTracker()
        tool_calls_made = 0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(today=self.today.isoformat())},
            {"role": "user", "content": question},
        ]
        self._emit("user_question", question=question)

        for iteration in range(1, self.max_iterations + 1):
            # On the last allowed iteration, tools stay visible but the model may not call
            # them ("none"), so the loop is guaranteed to end with an answer.
            last_iteration = iteration == self.max_iterations
            if last_iteration and tool_calls_made:
                messages.append({"role": "user", "content": FORCE_ANSWER_NOTE})
                self._emit("iteration_limit", limit=self.max_iterations)

            tool_choice = "none" if last_iteration else "auto"
            self._emit(
                "llm_request",
                iteration=iteration,
                message_count=len(messages),
                tools=self.tools.names,
                tool_choice=tool_choice,
            )

            # Steps 1-2: send the conversation + tool definitions to the LLM.
            response = self.llm.chat(messages, tools=self.tools.schemas(), tool_choice=tool_choice)
            messages.append(response.to_message())
            self._emit(
                "llm_response",
                iteration=iteration,
                decision="tool_calls" if response.tool_calls else "final_answer",
                tool_calls=[call.model_dump() for call in response.tool_calls],
                content=response.content,
                usage=response.usage,
            )

            # Step 7: no tool requested -> this text is the final answer.
            if not response.tool_calls:
                result = build_final_response(
                    question,
                    response.content,
                    sources,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    stopped_reason="max_iterations" if last_iteration and tool_calls_made else "completed",
                )
                self._emit("final_answer", answer=result.answer)
                return result

            # Steps 3-6: the model requested one or more tools. Run each, feed results back.
            for call in response.tool_calls:
                tool_calls_made += 1
                self._emit("tool_call", name=call.name, arguments=call.arguments)
                outcome = self.tools.execute(call.name, call.arguments)
                messages.append(self._tool_message(call, outcome, sources))

        # Only reachable if the provider ignored tool_choice="none" on the last iteration.
        return build_final_response(
            question,
            "I could not finish researching this question within the allowed number of steps.",
            sources,
            iterations=self.max_iterations,
            tool_calls_made=tool_calls_made,
            stopped_reason="max_iterations",
        )

    def _tool_message(self, call: ToolCall, outcome: ToolResult, sources: SourceTracker) -> dict[str, Any]:
        """Turn a ToolResult into the `tool` message the LLM reads on its next turn."""
        payload = dict(outcome.data)
        if outcome.sources:
            numbered = sources.add(outcome.sources)
            payload["results"] = [
                {"source_id": s.id, "title": s.title, "url": s.url, "snippet": s.snippet}
                for s in numbered
            ]

        self._emit(
            "tool_result",
            name=call.name,
            ok=outcome.ok,
            error=outcome.error,
            result_count=len(outcome.sources),
            payload=payload,
        )
        # tool_call_id links this result to the exact request it answers.
        return {"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)}

    def _emit(self, event_type: str, **data: Any) -> None:
        self.on_event(AgentEvent(type=event_type, data=data))
