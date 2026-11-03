"""What every LLM provider takes and returns (PRD 3.1).

A provider receives instructions, the source documents (retrieved chunks, verbatim) and a
question, and returns the answer as parts, each with the citations that support it. Citations
point back to a document by its position in the request, so the caller can map them to the
retrieved chunks. This module only carries citations; `llm.citations` checks them (PRD 3.4).
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SourceDocument:
    title: str  # e.g. "Asthma (chunk 1)"
    text: str  # the chunk exactly as stored; never edited or summarised


@dataclass(frozen=True)
class GenerationRequest:
    system: str  # instructions for the model
    documents: tuple[SourceDocument, ...]
    question: str
    max_tokens: int = 4096

    def __post_init__(self):
        if not self.question.strip():
            raise ValueError("the question must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass(frozen=True)
class Citation:
    document_index: int  # position of the cited document in GenerationRequest.documents
    start_char: int  # the cited passage is document.text[start_char:end_char]
    end_char: int
    cited_text: str


@dataclass(frozen=True)
class AnswerPart:
    text: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str  # the model that produced the answer (a fallback model, if one ran)
    parts: tuple[AnswerPart, ...]
    stop_reason: str  # e.g. "end_turn", "max_tokens", "refusal"
    usage: TokenUsage
    latency_ms: float
    refusal_category: str | None = None
    fell_back: bool = False  # a fallback model ran after the requested model declined

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class LLMUnavailableError(RuntimeError):
    """The provider could not produce an answer. The message is safe to show: it never
    contains credentials, request content or file paths."""
