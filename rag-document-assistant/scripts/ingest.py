r"""Index PDFs into ChromaDB: extract -> chunk -> embed -> store.

    python -m scripts.ingest data\pdfs               # every PDF in the folder
    python -m scripts.ingest data\pdfs\paper.pdf     # one file
    python -m scripts.ingest --list                  # what is in the index
    python -m scripts.ingest data\pdfs --rebuild     # empty this configuration's index first

Safe to run repeatedly: files already in the index (under any name) are skipped.
The index used depends on EMBEDDING_MODEL, CHUNK_SIZE and CHUNK_OVERLAP in .env.
"""

import argparse
import logging
import sys
from pathlib import Path

from app.config import ConfigError, configure_logging, load_settings
from app.embeddings import Embedder, EmbeddingError
from app.ingestion import ingest_pdf
from app.vector_store import VectorStore, VectorStoreError, open_store


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Index PDFs into the vector database.")
    parser.add_argument("path", type=Path, nargs="?", help="a PDF file, or a folder of PDFs")
    parser.add_argument("--list", action="store_true", help="list the documents in the index")
    parser.add_argument("--rebuild", action="store_true", help="delete this configuration's index before indexing")
    args = parser.parse_args()
    if not args.list and args.path is None:
        parser.error("give a PDF file or folder, or use --list")

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 1
    configure_logging(settings.log_level)
    logging.getLogger("app.ingestion").setLevel(logging.ERROR)  # this script prints its own per-file status

    try:
        store = open_store(settings)
    except VectorStoreError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"Index: {settings.chroma_dir}  collection '{store.collection_name}'  ({store.count()} chunks)")

    if args.list:
        return list_documents(store)
    if args.rebuild:
        store.reset()
        print("Rebuild: emptied this collection.")

    pdfs = sorted(args.path.glob("*.pdf")) if args.path.is_dir() else [args.path]
    if not pdfs or not pdfs[0].is_file():
        print(f"No PDF files found at {args.path}")
        return 1

    print(f"Loading embedding model {settings.embedding_model}...")
    try:
        embedder = Embedder(settings.embedding_model)
    except EmbeddingError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"Model ready: {embedder.dimension} dimensions, max {embedder.max_tokens} tokens per chunk\n")

    results = []
    for number, path in enumerate(pdfs, start=1):
        result = ingest_pdf(path.read_bytes(), path.name, settings, embedder, store)
        results.append(result)
        name_column = path.name if len(path.name) <= 50 else path.name[:47] + "..."
        if result.status == "indexed":
            detail = f"{result.chunks} chunks from {result.pages} pages ({result.seconds:.1f}s)"
            if result.truncated_chunks:
                detail += f", {result.truncated_chunks} over the token limit"
        else:
            detail = result.message
        print(f"[{number:>2}/{len(pdfs)}] {name_column:50} {result.status.upper():8} {detail}")

    counts = {status: sum(r.status == status for r in results) for status in ("indexed", "skipped", "failed")}
    truncated = sum(r.truncated_chunks for r in results)
    print(
        f"\n{counts['indexed']} indexed, {counts['skipped']} skipped, {counts['failed']} failed. "
        f"The index now holds {store.count()} chunks."
    )
    if truncated:
        print(
            f"Note: {truncated} chunk(s) were longer than {embedder.max_tokens} tokens; "
            "the model only embedded their beginning."
        )
    return 1 if counts["failed"] else 0


def list_documents(store: VectorStore) -> int:
    documents = store.list_documents()
    if not documents:
        print("The index is empty. Add PDFs with: python -m scripts.ingest data\\pdfs")
        return 0
    print(f"\n{'chunks':>6}  {'document id':14}  file")
    for document in documents:
        print(f"{document.chunk_count:>6}  {document.doc_id[:12]}..  {document.filename}")
    print(f"\n{len(documents)} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
