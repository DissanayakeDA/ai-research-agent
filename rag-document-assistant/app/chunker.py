"""Ingestion step 2: split each document into overlapping chunks.

Why not store whole documents?
  * Retrieval returns whole chunks. A single vector for a 20-page paper is a blurry average
    of every topic in it; a ~1000-character chunk is usually about one thing, so its vector
    can match a specific question.
  * The embedding model reads at most 512 tokens (~2000 characters) and ignores the rest.
  * The LLM then receives only a few relevant chunks instead of whole documents:
    cheaper, faster, and less to get distracted by.

Why overlap? A sentence cut at a chunk boundary still appears complete in the next chunk.

Sizes are in characters, because they are easy to reason about and to inspect. English
averages roughly 4 characters per token, so 1000 characters is about 250 tokens.
"""

import bisect

from app.schemas import Chunk, LoadedDocument

# Where a chunk may end, best first: a paragraph break, then a sentence end, then any space.
BREAK_LEVELS = (("\n\n",), (". ", "? ", "! "), (" ",))

PAGE_SEPARATOR = "\n\n"  # a page break also counts as a paragraph break


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return the (start, end) character positions of each chunk.

    A chunk is at most `chunk_size` characters. It ends at the best break found in its
    second half (so chunks are never tiny), and the next chunk starts about
    `chunk_overlap` characters before that end.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")

    spans = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_break(text, earliest=start + chunk_size // 2, latest=end)
        spans.append((start, end))
        if end == len(text):
            break
        start = _next_start(text, previous_start=start, end=end, overlap=chunk_overlap)
    return spans


def _find_break(text: str, earliest: int, latest: int) -> int:
    """Position just after the best separator inside text[earliest:latest]."""
    for separators in BREAK_LEVELS:
        positions = [(text.rfind(sep, earliest, latest), sep) for sep in separators]
        ends = [position + len(sep) for position, sep in positions if position != -1]
        if ends:
            return max(ends)  # the latest break of this kind keeps the chunk as long as allowed
    return latest  # one unbroken run of characters (e.g. a long URL): cut it mid-word


def _next_start(text: str, previous_start: int, end: int, overlap: int) -> int:
    start = max(end - overlap, previous_start + 1)  # always move forward, or we'd loop forever
    while start < end and not text[start - 1].isspace():  # don't begin in the middle of a word
        start += 1
    return start


def chunk_document(document: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split a document into chunks, recording the page(s) each chunk came from."""
    text, page_starts, page_numbers = _join_pages(document)

    def page_at(position: int) -> int:
        return page_numbers[bisect.bisect_right(page_starts, position) - 1]

    chunks: list[Chunk] = []
    for start, end in split_text(text, chunk_size, chunk_overlap):
        # Trim surrounding whitespace, moving the span with it so page numbers stay exact.
        raw = text[start:end]
        start += len(raw) - len(raw.lstrip())
        end -= len(raw) - len(raw.rstrip())
        if start >= end:
            continue
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{index}",
                doc_id=document.doc_id,
                filename=document.filename,
                chunk_index=index,
                page_start=page_at(start),
                page_end=page_at(end - 1),
                char_start=start,
                char_end=end,
                text=text[start:end],
            )
        )
    return chunks


def _join_pages(document: LoadedDocument) -> tuple[str, list[int], list[int]]:
    """Concatenate the page texts, remembering where each page starts in the result.

    Chunking the joined text (instead of each page separately) keeps a paragraph that
    continues onto the next page together; the page map still tells us which page(s)
    every chunk came from.
    """
    texts, starts, numbers = [], [], []
    offset = 0
    for page in document.pages:
        if not page.text:
            continue
        texts.append(page.text)
        starts.append(offset)
        numbers.append(page.page_number)
        offset += len(page.text) + len(PAGE_SEPARATOR)
    return PAGE_SEPARATOR.join(texts), starts, numbers
