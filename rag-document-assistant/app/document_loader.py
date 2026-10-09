"""Ingestion step 1: validate a PDF and extract clean text, page by page.

PyMuPDF reads the *text layer* that tools like Word or LaTeX embed in a PDF. A scanned PDF
is only pictures of pages - it has no text layer, so nothing comes out, and it would need
OCR (optical character recognition) before it could be searched.

PDF text is laid out for printing, not for reading by a program, so we clean it:
  * ligatures   - some fonts store "fi" as one character, "ﬁ"; "classiﬁcation" would then
                  never match a search for "classification"
  * hyphenation - words split at the end of a line ("sched-" / "uling") are rejoined
  * line breaks - inside a paragraph they are just layout, so they become spaces
"""

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

import pymupdf

from app.schemas import LoadedDocument, Page

logger = logging.getLogger(__name__)

# Fewer characters than this per page (on average) means there is no real text layer.
# Scanned PDFs often still contain a few stray characters, e.g. a scanner app's watermark.
MIN_CHARS_PER_PAGE = 20

HYPHENATED_LINE_BREAK = re.compile(r"(\w)-\n([a-z])")  # "sched-\nuling" -> "scheduling"
WHITESPACE = re.compile(r"\s+")


class DocumentLoadError(Exception):
    """The file can't be used. The message explains why and is safe to show to users."""


def load_pdf_file(path: Path, max_bytes: int | None = None) -> LoadedDocument:
    return load_pdf_bytes(path.read_bytes(), path.name, max_bytes)


def load_pdf_bytes(data: bytes, filename: str, max_bytes: int | None = None) -> LoadedDocument:
    """Validate PDF bytes and extract the cleaned text of every page."""
    if not data:
        raise DocumentLoadError(f"'{filename}' is empty (0 bytes).")
    if max_bytes is not None and len(data) > max_bytes:
        raise DocumentLoadError(
            f"'{filename}' is {len(data) / 1_048_576:.1f} MB; the limit is {max_bytes / 1_048_576:.0f} MB."
        )
    # Real PDFs begin with "%PDF-". This catches renamed files, e.g. a .docx saved as .pdf.
    if b"%PDF-" not in data[:1024]:
        raise DocumentLoadError(f"'{filename}' is not a PDF file.")

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises different exception types for different damage
        raise DocumentLoadError(f"'{filename}' could not be opened; the PDF may be damaged ({exc}).") from exc

    with pdf:
        if pdf.needs_pass:
            raise DocumentLoadError(f"'{filename}' is password-protected. Remove the password and try again.")
        if pdf.page_count == 0:
            raise DocumentLoadError(f"'{filename}' has no pages.")
        try:
            pages = [Page(page_number=number, text=_page_text(page)) for number, page in enumerate(pdf, start=1)]
        except Exception as exc:
            raise DocumentLoadError(f"'{filename}': text extraction failed; the PDF may be damaged ({exc}).") from exc

    document = LoadedDocument(doc_id=hashlib.sha256(data).hexdigest(), filename=filename, pages=pages)

    if document.char_count < MIN_CHARS_PER_PAGE * document.page_count:
        raise DocumentLoadError(
            f"'{filename}' has no extractable text ({document.char_count} characters in "
            f"{document.page_count} pages). It is probably a scanned document - images of pages - "
            "and would need OCR before it can be searched."
        )
    if document.empty_pages:
        logger.warning("'%s': no text on page(s) %s - images or scanned pages?", filename, document.empty_pages)
    return document


def _page_text(page: pymupdf.Page) -> str:
    """One page as paragraphs separated by blank lines.

    PyMuPDF groups the text into blocks (roughly paragraphs). We keep the PDF's own block
    order: for most papers it already reads the left column, then the right one. Sorting
    blocks by position on the page would interleave the two columns line by line.
    """
    paragraphs = []
    for _x0, _y0, _x1, _y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 0 = text, 1 = image
            continue
        paragraph = clean_text(text)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    """Undo print layout: expand ligatures, rejoin hyphenated words, flatten line breaks."""
    text = unicodedata.normalize("NFKC", text)  # "ﬁ" -> "fi", non-breaking space -> space
    text = HYPHENATED_LINE_BREAK.sub(r"\1\2", text)
    return WHITESPACE.sub(" ", text).strip()
