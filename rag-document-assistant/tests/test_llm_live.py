"""Live tests against the real LLM in your .env (Groq by default).

Skipped by default because they use tokens from your rate limit and may be billed.
Run them with:  pytest -m llm
"""

import pytest

from app.config import load_settings
from app.generator import Generator
from app.pipeline import build_pipeline
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.llm

PASSAGE = RetrievedChunk(
    rank=1, score=0.8, chunk_id="doc:1", doc_id="doc", filename="compare.pdf", chunk_index=0,
    page_start=1, page_end=1, text="The CP-SAT solver solved the problem 175 times faster than the genetic algorithm.",
)


@pytest.fixture(scope="module")
def generator() -> Generator:
    return build_pipeline(load_settings()).generator


def test_grounded_answer_cites_the_passage(generator):
    answer = generator.answer("Which solver was faster?", [PASSAGE])

    assert answer.found
    assert answer.cited == [1]
    assert "CP-SAT" in answer.answer.replace("‑", "-")  # models sometimes use a non-breaking hyphen


def test_question_outside_the_passages_is_declined(generator):
    answer = generator.answer("What is the capital of France?", [PASSAGE])

    assert not answer.found
