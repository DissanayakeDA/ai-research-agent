"""Source tracking and the final response generator.

Rule: the LLM never writes the source list. It only cites numbers like [1] or [2][3].
Python keeps the real URLs (SourceTracker) and builds the source list from them
(build_final_response). If the model cites a number that was never retrieved, or pastes a
URL that did not come from a tool, we flag it instead of showing it as a source.
"""

import re

from app.models import AgentResult, SearchResult, Source

# Matches [1], [12] and grouped citations like [1, 3]. Up to 3 digits, so years like [2026] don't match.
CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")


class SourceTracker:
    """Numbers every URL the tools return during one run, de-duplicated by URL.

    Numbering is global across searches: if the agent searches twice, the second search's
    results continue at [6], [7], ... so citations stay unambiguous.
    """

    def __init__(self) -> None:
        self._by_url: dict[str, Source] = {}

    def add(self, results: list[SearchResult]) -> list[Source]:
        numbered = []
        for result in results:
            source = self._by_url.get(result.url)
            if source is None:
                source = Source(id=len(self._by_url) + 1, **result.model_dump())
                self._by_url[result.url] = source
            numbered.append(source)
        return numbered

    def get(self, source_id: int) -> Source | None:
        return next((s for s in self._by_url.values() if s.id == source_id), None)

    def all(self) -> list[Source]:
        return list(self._by_url.values())


def build_final_response(
    question: str,
    answer_text: str | None,
    sources: SourceTracker,
    *,
    iterations: int,
    tool_calls_made: int,
    stopped_reason: str = "completed",
) -> AgentResult:
    answer = (answer_text or "").strip() or "(The model returned an empty answer.)"
    warnings: list[str] = []

    # 1. Resolve citations to real, retrieved sources - in the order they first appear.
    cited: list[Source] = []
    for group in CITATION_RE.findall(answer):
        for number in (int(n) for n in group.split(",")):
            source = sources.get(number)
            if source is None:
                warning = f"The answer cites [{number}], but no source with that number was retrieved."
                if warning not in warnings:
                    warnings.append(warning)
            elif source not in cited:
                cited.append(source)

    # 2. Any URL written into the answer text must be one the tools actually returned.
    known_urls = {s.url for s in sources.all()}
    for url in URL_RE.findall(answer):
        if url.rstrip(".,;:") not in known_urls:
            warnings.append(f"The answer mentions a URL that did not come from search results: {url}")

    # 3. Searched but cited nothing? The answer may not be grounded in the results.
    if sources.all() and not cited:
        warnings.append("Web search returned results, but the answer does not cite any of them.")

    if stopped_reason == "max_iterations":
        warnings.append("The agent hit its iteration limit and was forced to answer with what it had.")

    return AgentResult(
        question=question,
        answer=answer,
        cited_sources=cited,
        retrieved_sources=sources.all(),
        tool_calls_made=tool_calls_made,
        iterations=iterations,
        stopped_reason=stopped_reason,
        warnings=warnings,
    )
