"""Ask the index a question and see which chunks come back - retrieval only, no LLM.

    python -m scripts.search "Which solver was faster than the genetic algorithm?"
    python -m scripts.search "How was PPO configured?" --top-k 10
    python -m scripts.search "What is the capital of France?" --min-score 0.6
    python -m scripts.search "..." --full     # print whole chunks instead of previews

Runs entirely on your machine: the question is embedded locally and searched in ChromaDB.
"""

import argparse
import logging
import shutil
import sys
import textwrap
import time

from app.config import ConfigError, configure_logging, load_settings
from app.embeddings import Embedder, EmbeddingError
from app.retriever import Retriever
from app.vector_store import VectorStoreError, open_store

PREVIEW_CHARS = 300
# Wrap text ourselves, narrower than the terminal: when the terminal wraps a long line,
# copying it can drop the space at the wrap point ("producing a" -> "producinga").
WIDTH = min(100, shutil.get_terminal_size((100, 20)).columns - 1)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Show the chunks retrieved for a question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, help="how many chunks to return (default: TOP_K)")
    parser.add_argument("--min-score", type=float, help="drop chunks with a lower similarity score")
    parser.add_argument("--full", action="store_true", help="print each chunk's full text")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 1
    configure_logging(settings.log_level)
    logging.getLogger("app.retriever").setLevel(logging.WARNING)  # timing is printed below instead
    top_k = args.top_k if args.top_k is not None else settings.top_k

    try:
        store = open_store(settings)
        if store.count() == 0:
            print("The index is empty. Add PDFs first: python -m scripts.ingest data\\pdfs")
            return 1
        retriever = Retriever(Embedder(settings.embedding_model), store)
        started = time.perf_counter()
        results = retriever.retrieve(args.question, top_k, args.min_score)
        milliseconds = (time.perf_counter() - started) * 1000
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1

    min_note = f", min_score={args.min_score}" if args.min_score is not None else ""
    print(f"Question: {args.question}")
    print(f"Searched {store.count()} chunks in '{store.collection_name}' (top_k={top_k}{min_note}) in {milliseconds:.0f} ms")
    if not results:
        print("\nNo chunk scored above the minimum - the documents probably don't cover this question.")
        return 0

    for chunk in results:
        pages = f"p.{chunk.page_start}" if chunk.page_start == chunk.page_end else f"pp.{chunk.page_start}-{chunk.page_end}"
        text = " ".join(chunk.text.split())  # one paragraph per chunk is easier to scan
        if not args.full and len(text) > PREVIEW_CHARS:
            text = text[:PREVIEW_CHARS] + "..."
        print(f"\n#{chunk.rank}  score {chunk.score:.3f}  {chunk.filename}  {pages}  (chunk {chunk.chunk_index})")
        print(textwrap.fill(text, WIDTH, initial_indent="    ", subsequent_indent="    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
