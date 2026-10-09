r"""Look inside ingestion: what text comes out of a PDF, and how it gets chunked.

    python -m scripts.inspect_chunks data\pdfs\paper.pdf             # summary + first 3 chunks
    python -m scripts.inspect_chunks data\pdfs\paper.pdf --show all  # every chunk
    python -m scripts.inspect_chunks data\pdfs\paper.pdf --page 2    # cleaned text of page 2
    python -m scripts.inspect_chunks data\pdfs                        # one line per PDF in a folder
    python -m scripts.inspect_chunks data\pdfs --chunk-size 500 --chunk-overlap 50

Chunk size and overlap default to your .env values; the flags override them for experiments.
Nothing is stored and nothing leaves your machine - this only reads the files.
"""

import argparse
import statistics
import sys
from pathlib import Path

from app.chunker import chunk_document
from app.config import ConfigError, configure_logging, load_settings
from app.document_loader import DocumentLoadError, load_pdf_file
from app.schemas import Chunk


def main() -> int:
    # PDF text contains characters (Greek letters, dashes...) the Windows console code page lacks.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Inspect PDF text extraction and chunking.")
    parser.add_argument("path", type=Path, help="a PDF file, or a folder of PDFs")
    parser.add_argument("--chunk-size", type=int, help="characters per chunk (default: CHUNK_SIZE)")
    parser.add_argument("--chunk-overlap", type=int, help="characters of overlap (default: CHUNK_OVERLAP)")
    parser.add_argument("--show", default="3", help="how many chunks to print: a number or 'all' (default: 3)")
    parser.add_argument("--page", type=int, help="print the cleaned text of this page instead of chunks")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 1
    configure_logging(settings.log_level)

    size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap
    if size <= 0 or not 0 <= overlap < size:
        parser.error("need chunk-size > 0 and 0 <= chunk-overlap < chunk-size")
    if args.show != "all" and not args.show.isdigit():
        parser.error("--show must be a number or 'all'")

    if args.path.is_dir():
        return summarize_folder(args.path, size, overlap, settings.max_upload_bytes)
    if args.path.is_file():
        return inspect_file(args.path, size, overlap, settings.max_upload_bytes, args.show, args.page)
    print(f"Not found: {args.path}")
    return 1


def inspect_file(path: Path, size: int, overlap: int, max_bytes: int, show: str, page: int | None) -> int:
    try:
        document = load_pdf_file(path, max_bytes)
    except DocumentLoadError as exc:
        print(f"FAILED: {exc}")
        return 1

    empty = f"  (no text on pages {document.empty_pages})" if document.empty_pages else "  (all have text)"
    print(f"File:        {document.filename}")
    print(f"Document ID: {document.doc_id[:16]}...  (SHA-256 of the file bytes)")
    print(f"Pages:       {document.page_count}{empty}")
    print(f"Characters:  {document.char_count:,}")

    if page is not None:
        if not 1 <= page <= document.page_count:
            print(f"\nPage {page} does not exist (1-{document.page_count}).")
            return 1
        print(f"\n--- Page {page}: cleaned text, paragraphs separated by blank lines ---\n")
        print(document.pages[page - 1].text or "(no text on this page)")
        return 0

    chunks = chunk_document(document, size, overlap)
    lengths = [len(chunk.text) for chunk in chunks]
    print(f"Chunking:    size={size}, overlap={overlap} -> {len(chunks)} chunks")
    print(f"Chunk chars: min {min(lengths)}, average {statistics.mean(lengths):.0f}, max {max(lengths)}")

    limit = len(chunks) if show == "all" else int(show)
    previous: Chunk | None = None
    for chunk in chunks[:limit]:
        pages = f"page {chunk.page_start}" if chunk.page_start == chunk.page_end else f"pages {chunk.page_start}-{chunk.page_end}"
        shared = max(0, previous.char_end - chunk.char_start) if previous else 0
        overlap_note = f" | first {shared} chars repeat the end of chunk {previous.chunk_index}" if shared else ""
        print(f"\n--- Chunk {chunk.chunk_index} | {pages} | {len(chunk.text)} chars{overlap_note} ---")
        print(chunk.text)
        previous = chunk
    if limit < len(chunks):
        print(f"\n... {len(chunks) - limit} more chunks (use --show all)")
    return 0


def summarize_folder(folder: Path, size: int, overlap: int, max_bytes: int) -> int:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No .pdf files in {folder}")
        return 1

    print(f"Chunking: size={size}, overlap={overlap}\n")
    print(f"{'file':50} {'pages':>5} {'chars':>8} {'chunks':>6}  status")
    first_seen: dict[str, str] = {}  # doc_id -> filename
    unique_chunks = 0
    for path in pdfs:
        name = path.name if len(path.name) <= 50 else path.name[:47] + "..."
        try:
            document = load_pdf_file(path, max_bytes)
        except DocumentLoadError as exc:
            print(f"{name:50} {'-':>5} {'-':>8} {'-':>6}  FAILED: {exc}")
            continue

        chunks = chunk_document(document, size, overlap)
        if document.doc_id in first_seen:
            status = f"DUPLICATE of {first_seen[document.doc_id]} (same bytes)"
        else:
            first_seen[document.doc_id] = path.name
            unique_chunks += len(chunks)
            status = f"ok, no text on pages {document.empty_pages}" if document.empty_pages else "ok"
        print(f"{name:50} {document.page_count:>5} {document.char_count:>8,} {len(chunks):>6}  {status}")

    print(f"\n{len(pdfs)} files, {len(first_seen)} unique documents, {unique_chunks} chunks from unique documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
