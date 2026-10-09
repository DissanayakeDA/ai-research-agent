"""Ask a question about your documents: retrieve passages, then generate a cited answer.

    python -m scripts.ask "Which solver was faster than the genetic algorithm?"
    python -m scripts.ask "..." --top-k 6 --min-score 0.6
    python -m scripts.ask "..." --show-context     # also print the passages sent to the LLM
    python -m scripts.ask "..." --show-reasoning   # also print the model's hidden reasoning

Each question makes ONE call to the LLM in LLM_BASE_URL (Groq by default), sending the
question and the retrieved passages. It counts against your rate limits and may be billed.
If no passage passes --min-score, the LLM is not called at all.
"""

import argparse
import logging
import shutil
import sys
import textwrap

from app.config import ConfigError, configure_logging, load_settings
from app.embeddings import EmbeddingError
from app.llm_client import LLMError
from app.pipeline import build_pipeline
from app.schemas import Answer
from app.vector_store import VectorStoreError

WIDTH = min(100, shutil.get_terminal_size((100, 20)).columns - 1)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # LLMs write characters like "‑" and "×"

    parser = argparse.ArgumentParser(description="Answer a question from your documents, with citations.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, help="passages to retrieve (default: TOP_K)")
    parser.add_argument("--min-score", type=float, help="ignore passages with a lower similarity score")
    parser.add_argument("--show-context", action="store_true", help="print the passages sent to the LLM")
    parser.add_argument("--show-reasoning", action="store_true", help="print the model's reasoning, if it returns one")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 1
    configure_logging(settings.log_level)
    for name in ("app.retriever", "app.llm_client"):
        logging.getLogger(name).setLevel(logging.WARNING)  # timings are printed below instead
    top_k = args.top_k if args.top_k is not None else settings.top_k

    try:
        pipeline = build_pipeline(settings)
        answer = pipeline.ask(args.question, top_k, args.min_score)
    except (EmbeddingError, VectorStoreError, LLMError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1

    print_answer(answer, args.show_context, args.show_reasoning)
    return 0


def print_answer(answer: Answer, show_context: bool, show_reasoning: bool) -> None:
    print(f"Question: {answer.question}")
    print(f"Retrieved {len(answer.retrieved)} passage(s) in {answer.retrieval_ms:.0f} ms")
    if answer.model:
        print(
            f"{answer.model} answered in {answer.generation_ms / 1000:.1f} s "
            f"({answer.prompt_tokens:,} prompt + {answer.completion_tokens:,} completion tokens)"
        )
    else:
        print("The LLM was not called: no passage was relevant enough.")

    print("\nAnswer:")
    for paragraph in answer.answer.splitlines():
        print(textwrap.fill(paragraph, WIDTH, initial_indent="    ", subsequent_indent="    ") if paragraph.strip() else "")

    if answer.sources:
        print("\nSources cited:")
        for chunk in answer.sources:
            print(f"    [{chunk.rank}] {chunk.filename}, {_pages(chunk.page_start, chunk.page_end)}  (score {chunk.score:.3f})")
    uncited = [chunk.rank for chunk in answer.retrieved if chunk.rank not in answer.cited]
    if uncited:
        print("Retrieved but not cited: " + " ".join(f"[{n}]" for n in uncited))
    if answer.invalid_citations:
        print(f"WARNING: the answer cites {answer.invalid_citations}, but only passages 1-{len(answer.retrieved)} were sent.")
    if answer.found and not answer.cited:
        print("WARNING: the answer cites no passage, so nothing in it is traceable to your documents.")

    if show_context:
        print("\n--- Passages sent to the LLM ---")
        for chunk in answer.retrieved:
            print(f"\n[{chunk.rank}] {chunk.filename}, {_pages(chunk.page_start, chunk.page_end)} (score {chunk.score:.3f})")
            print(textwrap.fill(" ".join(chunk.text.split()), WIDTH, initial_indent="    ", subsequent_indent="    "))
    if show_reasoning and answer.reasoning:
        print("\n--- Model reasoning (not shown to users; useful for debugging) ---")
        print(textwrap.fill(" ".join(answer.reasoning.split()), WIDTH, initial_indent="    ", subsequent_indent="    "))


def _pages(start: int, end: int) -> str:
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


if __name__ == "__main__":
    sys.exit(main())
