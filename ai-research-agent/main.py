"""Command-line interface for the AI Research Assistant.

    python main.py                      interactive mode
    python main.py --debug              show every step of the agent loop
    python main.py "your question"      answer one question and exit
"""

import argparse
import json
import sys
import textwrap

from app.agent import ResearchAgent
from app.config import ConfigError, load_settings
from app.llm import LLMClient, LLMError
from app.models import AgentEvent, AgentResult
from app.tools import build_default_registry

DIVIDER = "-" * 60


# --- Event renderers: the agent emits events, these decide what the user sees -----------

def progress_printer(event: AgentEvent) -> None:
    """Normal mode: short, friendly progress lines."""
    data = event.data
    if event.type == "tool_call" and data["name"] == "web_search":
        query = _parse_query(data["arguments"])
        print(f"Searching the web for: {query!r} ..." if query else "Searching the web...")
    elif event.type == "tool_result":
        if not data["ok"]:
            print(f"Tool problem: {data['error']}")
        elif data["name"] == "web_search":
            print(f"Found {data['result_count']} results.")
    elif event.type == "llm_request":
        print("Generating answer..." if data["iteration"] > 1 else "Deciding whether a web search is needed...")
    elif event.type == "iteration_limit":
        print(f"Reached the limit of {data['limit']} iterations - asking for a final answer.")


def debug_printer(event: AgentEvent) -> None:
    """Debug mode: every observable step - messages, decisions, tool calls, results.

    Only observable actions are shown. We never ask the model to reveal its private reasoning.
    """
    data = event.data
    if event.type == "user_question":
        _block("USER", data["question"])
    elif event.type == "llm_request":
        _block(
            f"LLM CALL {data['iteration']}",
            f"sending {data['message_count']} messages | tools={data['tools']} | "
            f"tool_choice={data['tool_choice']!r}",
        )
    elif event.type == "llm_response":
        usage = data.get("usage") or {}
        tokens = f" (tokens: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')})" if usage else ""
        if data["decision"] == "tool_calls":
            calls = ", ".join(c["name"] for c in data["tool_calls"])
            _block("AGENT DECISION", f"Needs more information -> requested tool(s): {calls}{tokens}")
            if data.get("content"):
                _block("AGENT MESSAGE", data["content"])
        else:
            _block("AGENT DECISION", f"Write the final answer (no tool call){tokens}")
    elif event.type == "tool_call":
        _block("TOOL", f"{data['name']}({data['arguments']})")
    elif event.type == "tool_result":
        body = json.dumps(data["payload"], indent=2, ensure_ascii=False)
        if len(body) > 2500:
            body = body[:2500] + "\n... (truncated for display)"
        _block("TOOL RESULT" if data["ok"] else "TOOL ERROR", body)
    elif event.type == "iteration_limit":
        _block("LIMIT", f"Iteration limit ({data['limit']}) reached - tools disabled, forcing an answer.")
    elif event.type == "final_answer":
        _block("FINAL", "Final answer received - citations checked, formatted answer below.")


def _block(label: str, body: str) -> None:
    print(f"\n[{label}]\n{body}")


def _parse_query(raw_arguments: str) -> str | None:
    try:
        return json.loads(raw_arguments).get("query")
    except (ValueError, AttributeError):
        return None


# --- Output -------------------------------------------------------------------------

def print_result(result: AgentResult) -> None:
    print(f"\n{DIVIDER}\nAnswer:\n")
    print(result.answer)

    if result.cited_sources:
        print("\nSources:")
        for source in result.cited_sources:
            print(f"  [{source.id}] {source.title}\n      {source.url}")
        basis = f"web search ({len(result.cited_sources)} of {len(result.retrieved_sources)} retrieved sources cited)"
    elif result.used_search:
        basis = "general knowledge - a web search was attempted, but no search results were cited"
    else:
        basis = "the model's general knowledge (no web search was performed)"

    print(f"\nBasis: {basis}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print(DIVIDER)


# --- Entry point ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="AI Research Assistant")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--debug", action="store_true", help="show every step of the agent loop")
    args = parser.parse_args()
    # Don't crash on characters the Windows console can't encode (e.g. when output is redirected).
    sys.stdout.reconfigure(errors="replace")

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 1

    debug = args.debug or settings.debug
    agent = ResearchAgent(
        llm=LLMClient(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        ),
        tools=build_default_registry(settings),
        max_iterations=settings.max_iterations,
        on_event=debug_printer if debug else progress_printer,
    )

    print("=" * 60)
    print(" AI Research Assistant")
    print(f" model: {settings.llm_model} | debug: {'on' if debug else 'off'}")
    print("=" * 60)

    if args.question:
        return ask(agent, args.question)

    print("Type a research question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nAsk a research question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if question:
            ask(agent, question)


def ask(agent: ResearchAgent, question: str) -> int:
    try:
        result = agent.run(question)
    except LLMError as exc:
        print("\n" + textwrap.fill(f"LLM error: {exc}", width=100), file=sys.stderr)
        return 1
    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
