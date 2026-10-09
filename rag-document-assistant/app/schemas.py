"""Data models that flow between pipeline stages.

    PDF bytes -> LoadedDocument (pages of text) -> Chunk -> vectors in ChromaDB

Pydantic validates them, and in Phase 6 FastAPI can return them as JSON directly.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Page(BaseModel):
    page_number: int = Field(ge=1)  # 1-based, matching what PDF viewers show
    text: str  # cleaned text, paragraphs separated by blank lines; "" if the page has no text layer


class LoadedDocument(BaseModel):
    doc_id: str  # SHA-256 of the file bytes: identical files get identical IDs, whatever their name
    filename: str
    pages: list[Page]  # every page, including empty ones, so page numbers stay true to the PDF

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    @property
    def empty_pages(self) -> list[int]:
        return [page.page_number for page in self.pages if not page.text]


class Chunk(BaseModel):
    chunk_id: str  # "<doc_id>:<chunk_index>" - stable, so re-ingesting a file overwrites, not duplicates
    doc_id: str
    filename: str
    chunk_index: int = Field(ge=0)  # position within the document
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)  # a chunk can continue onto the next page
    char_start: int = Field(ge=0)  # position in the document's joined text; shows where chunks overlap
    char_end: int = Field(ge=0)
    text: str


class StoredDocument(BaseModel):
    """A document as it exists in the vector store."""

    doc_id: str
    filename: str
    chunk_count: int


class IngestResult(BaseModel):
    """What happened to one file during ingestion - shown as its processing status."""

    filename: str
    status: Literal["indexed", "skipped", "failed"]
    message: str  # human-readable reason, safe to show to users
    doc_id: str | None = None
    pages: int = 0
    chunks: int = 0
    truncated_chunks: int = 0  # chunks longer than the embedding model's token limit
    seconds: float = 0.0


class RetrievedChunk(BaseModel):
    """A stored chunk returned for a question, with how close it is to the question."""

    rank: int = Field(ge=1)  # 1 = most similar
    score: float  # cosine similarity of question and chunk vectors; higher = closer
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str


class Answer(BaseModel):
    """The result of asking a question: the answer, and exactly which evidence it used."""

    question: str
    answer: str
    found: bool  # False when the model said the documents don't contain the answer
    cited: list[int]  # passage numbers the answer cites that really exist, e.g. [1, 3]
    invalid_citations: list[int]  # cited numbers with no matching passage - should stay empty
    retrieved: list[RetrievedChunk]  # the passages sent to the LLM; passage [n] is rank n
    model: str | None = None  # None when the LLM was not called (nothing relevant retrieved)
    reasoning: str | None = None  # a reasoning model's hidden thinking, for inspection
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def sources(self) -> list[RetrievedChunk]:
        """The retrieved passages the answer actually cites."""
        return [chunk for chunk in self.retrieved if chunk.rank in self.cited]
