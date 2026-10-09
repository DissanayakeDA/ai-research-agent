"""Tests for app/document_loader.py: validation, text extraction, and cleaning.

The PDFs are generated in memory with PyMuPDF, so the tests don't depend on your papers.
"""

import pymupdf
import pytest

from app.document_loader import DocumentLoadError, clean_text, load_pdf_bytes
from helpers import make_pdf


def test_extracts_text_page_by_page():
    document = load_pdf_bytes(
        make_pdf("Exam timetabling assigns exams to rooms.", "Genetic algorithms search for timetables."),
        "paper.pdf",
    )

    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].text == "Exam timetabling assigns exams to rooms."
    assert document.pages[1].text == "Genetic algorithms search for timetables."
    assert document.filename == "paper.pdf"


def test_rejoins_words_hyphenated_at_a_line_end():
    document = load_pdf_bytes(make_pdf("Examination sched-\nuling is a hard problem."), "paper.pdf")

    assert document.pages[0].text == "Examination scheduling is a hard problem."


def test_clean_text_expands_ligatures_and_flattens_line_breaks():
    assert clean_text("classiﬁcation  of\nexam   data ") == "classification of exam data"


def test_document_id_depends_on_content_not_filename():
    data = make_pdf("Exam timetabling assigns exams to rooms.")

    same_a = load_pdf_bytes(data, "a.pdf")
    same_b = load_pdf_bytes(data, "renamed copy.pdf")
    different = load_pdf_bytes(make_pdf("A completely different document text."), "a.pdf")

    assert same_a.doc_id == same_b.doc_id
    assert same_a.doc_id != different.doc_id


def test_rejects_a_file_that_is_not_a_pdf():
    with pytest.raises(DocumentLoadError, match="not a PDF"):
        load_pdf_bytes(b"PK\x03\x04 this is really a .docx (zip) file", "report.pdf")


def test_rejects_an_empty_file():
    with pytest.raises(DocumentLoadError, match="empty"):
        load_pdf_bytes(b"", "empty.pdf")


def test_rejects_a_file_over_the_size_limit():
    data = make_pdf("Exam timetabling assigns exams to rooms.")

    with pytest.raises(DocumentLoadError, match="limit"):
        load_pdf_bytes(data, "big.pdf", max_bytes=len(data) - 1)


def test_rejects_a_damaged_pdf():
    with pytest.raises(DocumentLoadError):
        load_pdf_bytes(b"%PDF-1.7\n" + b"\x00garbage" * 200, "broken.pdf")


def test_rejects_a_password_protected_pdf():
    data = make_pdf(
        "Exam timetabling assigns exams to rooms.",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )

    with pytest.raises(DocumentLoadError, match="password"):
        load_pdf_bytes(data, "locked.pdf")


def test_pdf_without_text_is_reported_as_probably_scanned():
    with pytest.raises(DocumentLoadError, match="OCR"):
        load_pdf_bytes(make_pdf("", ""), "scan.pdf")


def test_pages_without_text_are_kept_and_reported():
    document = load_pdf_bytes(
        make_pdf("Exam timetabling assigns exams to rooms.", "", "Genetic algorithms search for timetables."),
        "paper.pdf",
    )

    assert document.page_count == 3
    assert document.empty_pages == [2]
