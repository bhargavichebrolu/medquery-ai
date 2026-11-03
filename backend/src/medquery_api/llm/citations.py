"""Checking the citations a provider returns (PRD 3.4).

A citation is kept only when all of these hold; otherwise it is dropped and counted with the
first reason that applies:

1. `unknown_document`: its document_index is a position in the request's documents.
2. `invalid_range`: 0 <= start_char < end_char <= len(document text).
3. `text_mismatch`: document.text[start_char:end_char] equals cited_text exactly.

Grounding coverage is the share of answer sentences that overlap, in at least one letter, an
answer part with a valid citation. Sentences are split deterministically: every line is split
after ".", "!" or "?" (with any closing quotes or brackets) followed by whitespace; markdown
headings and pieces without a letter (bullet markers, "1.") are not sentences. Abbreviations
such as "e.g." therefore end a sentence.

Each answer also gets a grounding status (PRD 3.6), the first of these that applies:

- `refused`: the model declined to answer.
- `not_enough_evidence`: the whole answer, ignoring surrounding whitespace, is exactly the
  sentence the instructions require when the documents are not enough (NOT_ENOUGH_EVIDENCE).
- `not_grounded`: no sentence has a valid citation (including an answer with no sentences), so
  the answer is never presented as grounded.
- `grounded`: every sentence has at least one valid citation.
- `partly_grounded`: some sentences do and some do not.

These are checks, not proof. A valid citation shows the quoted passage really is in the
retrieved chunk; it does not show the answer states that passage correctly, and coverage does
not measure whether the answer is right (answer quality is scored in Part 4.2).
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from medquery_api.llm.base import AnswerPart, Citation, GenerationResult, SourceDocument
from medquery_api.llm.mock import NOT_ENOUGH_EVIDENCE

Grounding = Literal["grounded", "partly_grounded", "not_grounded", "not_enough_evidence", "refused"]

UNKNOWN_DOCUMENT = "unknown_document"
INVALID_RANGE = "invalid_range"
TEXT_MISMATCH = "text_mismatch"
REASONS = (UNKNOWN_DOCUMENT, INVALID_RANGE, TEXT_MISMATCH)

_LINE = re.compile(r"[^\n]+")
_HEADING = re.compile(r"\s*#{1,6}\s")
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")
_LETTER = re.compile(r"[^\W\d_]")


@dataclass(frozen=True)
class DroppedCitation:
    part_index: int  # which answer part carried it
    citation: Citation  # exactly as the provider returned it
    reason: str  # one of REASONS


@dataclass(frozen=True)
class Sentence:
    # Character positions in the whole answer (all parts' text joined), without surrounding
    # whitespace: the sentence is answer_text[start:end].
    start: int
    end: int
    supported: bool  # at least one valid citation supports it


@dataclass(frozen=True)
class CitationCheck:
    parts: tuple[AnswerPart, ...]  # the answer parts with only their valid citations
    dropped: tuple[DroppedCitation, ...]
    total: int  # citations the provider returned
    sentence_list: tuple[Sentence, ...]

    @property
    def sentences(self) -> int:
        return len(self.sentence_list)

    @property
    def supported_sentences(self) -> int:
        return sum(sentence.supported for sentence in self.sentence_list)

    @property
    def valid(self) -> int:
        return self.total - len(self.dropped)

    @property
    def dropped_by_reason(self) -> dict[str, int]:
        counts = Counter(d.reason for d in self.dropped)
        return {reason: counts[reason] for reason in REASONS}

    @property
    def validity_rate(self) -> float | None:
        """Valid citations as a share of all citations; None when there were none."""
        return self.valid / self.total if self.total else None

    @property
    def grounding_coverage(self) -> float | None:
        """Supported sentences as a share of all sentences; None for an empty answer."""
        return self.supported_sentences / self.sentences if self.sentences else None


def citation_problem(citation: Citation, documents: tuple[SourceDocument, ...]) -> str | None:
    """Why a citation is invalid, or None when it is valid."""
    index = citation.document_index
    if not _is_int(index) or not 0 <= index < len(documents):
        return UNKNOWN_DOCUMENT
    text = documents[index].text
    start, end = citation.start_char, citation.end_char
    if not (_is_int(start) and _is_int(end)) or not 0 <= start < end <= len(text):
        return INVALID_RANGE
    if text[start:end] != citation.cited_text:
        return TEXT_MISMATCH
    return None


def check_citations(
    parts: tuple[AnswerPart, ...], documents: tuple[SourceDocument, ...]
) -> CitationCheck:
    """Keep each part's valid citations, count the dropped ones and measure coverage."""
    checked: list[AnswerPart] = []
    dropped: list[DroppedCitation] = []
    for part_index, part in enumerate(parts):
        valid = []
        for citation in part.citations:
            problem = citation_problem(citation, documents)
            if problem is None:
                valid.append(citation)
            else:
                dropped.append(DroppedCitation(part_index, citation, problem))
        checked.append(AnswerPart(part.text, tuple(valid)))

    text = "".join(part.text for part in checked)
    supported_spans = []
    position = 0
    for part in checked:
        if part.citations:
            supported_spans.append((position, position + len(part.text)))
        position += len(part.text)
    sentences = tuple(
        Sentence(
            start,
            end,
            any(_shares_a_letter(text, (start, end), span) for span in supported_spans),
        )
        for start, end in sentence_spans(text)
    )
    return CitationCheck(
        parts=tuple(checked),
        dropped=tuple(dropped),
        total=sum(len(part.citations) for part in parts),
        sentence_list=sentences,
    )


def grounding_status(result: GenerationResult, check: CitationCheck) -> Grounding:
    """How far the answer is supported by valid citations (rules in the module docstring)."""
    if result.refused:
        return "refused"
    if result.text.strip() == NOT_ENOUGH_EVIDENCE:
        return "not_enough_evidence"
    if check.supported_sentences == 0:
        return "not_grounded"
    if check.supported_sentences == check.sentences:
        return "grounded"
    return "partly_grounded"


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """The (start, end) character positions of each sentence in an answer, without the
    whitespace around it."""
    spans = []
    for line in _LINE.finditer(text):
        if _HEADING.match(line.group()):
            continue
        start = line.start()
        for end in _SENTENCE_END.finditer(text, line.start(), line.end()):
            spans.append((start, end.end()))
            start = end.end()
        spans.append((start, line.end()))
    return [_strip(text, start, end) for start, end in spans if _LETTER.search(text, start, end)]


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while text[start].isspace():
        start += 1
    while text[end - 1].isspace():
        end -= 1
    return start, end


def _shares_a_letter(text: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    start, end = max(a[0], b[0]), min(a[1], b[1])
    return start < end and _LETTER.search(text, start, end) is not None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
