"""A stand-in LLM for tests and local development: no network, no credentials, no cost.

By default it answers by quoting the first line of the first document and citing it, or, with
no documents, says the evidence is not enough. Tests can pass a `responder` to return any
result they need.
"""

from collections.abc import Callable

from medquery_api.llm.base import (
    AnswerPart,
    Citation,
    GenerationRequest,
    GenerationResult,
    TokenUsage,
)

NOT_ENOUGH_EVIDENCE = (
    "The provided documents do not contain enough information to answer this question."
)


class MockProvider:
    name = "mock"
    model = "mock"

    def __init__(self, responder: Callable[[GenerationRequest], GenerationResult] | None = None):
        self._responder = responder
        self.requests: list[GenerationRequest] = []  # every request received, for tests

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self._responder is not None:
            return self._responder(request)
        return GenerationResult(
            provider=self.name,
            model=self.model,
            parts=_default_answer(request),
            stop_reason="end_turn",
            # A rough, deterministic stand-in for token counts (about 4 characters per token).
            usage=TokenUsage(
                input_tokens=_estimate_tokens(request),
                output_tokens=20,
            ),
            latency_ms=0.0,
        )


def _default_answer(request: GenerationRequest) -> tuple[AnswerPart, ...]:
    if not request.documents:
        return (AnswerPart(NOT_ENOUGH_EVIDENCE),)
    text = request.documents[0].text
    first_line = next((line for line in text.splitlines() if line.strip()), text)
    start = text.index(first_line)
    citation = Citation(
        document_index=0,
        start_char=start,
        end_char=start + len(first_line),
        cited_text=first_line,
    )
    return (
        AnswerPart("According to the provided documents: "),
        AnswerPart(first_line, (citation,)),
    )


def _estimate_tokens(request: GenerationRequest) -> int:
    characters = len(request.system) + len(request.question)
    characters += sum(len(d.title) + len(d.text) for d in request.documents)
    return max(1, characters // 4)
