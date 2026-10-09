"""Tests for app/chunker.py: chunk sizes, overlap, natural breaks, and page tracking."""

import pytest

from app.chunker import chunk_document, split_text
from app.schemas import LoadedDocument, Page

SENTENCE = "Exam timetabling assigns every exam to a room and a time slot. "


def paragraphs(count: int, sentences_each: int = 5) -> str:
    return "\n\n".join(f"Paragraph {i}. " + (SENTENCE * sentences_each).strip() for i in range(count))


def make_document(*page_texts: str) -> LoadedDocument:
    pages = [Page(page_number=number, text=text) for number, text in enumerate(page_texts, start=1)]
    return LoadedDocument(doc_id="abc123", filename="paper.pdf", pages=pages)


SETTINGS = [(200, 0), (500, 50), (1000, 150), (300, 299)]


@pytest.mark.parametrize("size, overlap", SETTINGS)
def test_chunks_never_exceed_chunk_size(size, overlap):
    for start, end in split_text(paragraphs(20), size, overlap):
        assert 0 < end - start <= size


@pytest.mark.parametrize("size, overlap", SETTINGS)
def test_every_character_ends_up_in_some_chunk(size, overlap):
    text = paragraphs(20)
    covered = set()
    for start, end in split_text(text, size, overlap):
        covered.update(range(start, end))

    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


@pytest.mark.parametrize("size, overlap", [(500, 50), (1000, 150), (300, 299)])
def test_consecutive_chunks_overlap_by_at_most_chunk_overlap(size, overlap):
    spans = split_text(paragraphs(20), size, overlap)

    for (previous_start, previous_end), (start, _end) in zip(spans, spans[1:]):
        assert previous_start < start < previous_end  # moves forward, but repeats some text
        assert previous_end - start <= overlap


def test_without_overlap_chunks_do_not_overlap():
    spans = split_text(paragraphs(20), 500, 0)

    for (_start, previous_end), (start, _end) in zip(spans, spans[1:]):
        assert start >= previous_end


def test_chunks_never_start_in_the_middle_of_a_word():
    text = paragraphs(20)

    for start, _end in split_text(text, 500, 50)[1:]:
        assert text[start - 1].isspace()


def test_prefers_to_end_a_chunk_at_a_paragraph_break():
    first = (SENTENCE * 10).strip()  # ~630 characters
    text = first + "\n\n" + (SENTENCE * 10).strip()

    start, end = split_text(text, 1000, 0)[0]

    assert text[start:end].strip() == first


def test_falls_back_to_a_sentence_end():
    text = (SENTENCE * 40).strip()  # one paragraph, no blank lines

    for start, end in split_text(text, 500, 0)[:-1]:
        assert text[start:end].rstrip().endswith(".")


def test_cuts_mid_word_only_when_there_is_no_space_at_all():
    text = "x" * 2500  # e.g. a very long URL

    spans = split_text(text, 1000, 100)

    # Hard cuts at the size limit. There's no word start inside the overlap region,
    # so each chunk begins where the previous one ended.
    assert spans == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_empty_text_gives_no_chunks():
    assert split_text("", 1000, 150) == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_rejects_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", size, overlap)


def test_chunk_records_the_pages_it_came_from():
    document = make_document(paragraphs(2), paragraphs(2))  # each page ~640 characters

    chunks = chunk_document(document, 1000, 100)

    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(chunk.page_start == 1 and chunk.page_end == 2 for chunk in chunks)


def test_empty_pages_are_skipped_but_page_numbers_stay_true():
    document = make_document(paragraphs(3), "", paragraphs(3))

    chunks = chunk_document(document, 500, 50)

    pages = {page for chunk in chunks for page in (chunk.page_start, chunk.page_end)}
    assert pages == {1, 3}


def test_chunk_ids_are_stable_and_unique():
    document = make_document(paragraphs(10))

    first_run = [chunk.chunk_id for chunk in chunk_document(document, 500, 50)]
    second_run = [chunk.chunk_id for chunk in chunk_document(document, 500, 50)]

    assert first_run == second_run
    assert len(set(first_run)) == len(first_run)
    assert first_run[0] == "abc123:0"


def test_chunk_text_is_trimmed():
    for chunk in chunk_document(make_document(paragraphs(10)), 500, 50):
        assert chunk.text == chunk.text.strip()
        assert chunk.char_end - chunk.char_start == len(chunk.text)
