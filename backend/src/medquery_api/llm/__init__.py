"""LLM providers (PRD 3.1): one interface, a mock for tests, and Claude.

Which provider the application uses is configured, never assumed:

    MEDQUERY_LLM_PROVIDER  "mock" (default: no network, no cost) or "claude"
    MEDQUERY_LLM_MODEL     Claude model id (default "claude-sonnet-5-5")
    MEDQUERY_LLM_EFFORT    low | medium | high | xhigh | max (default "low")

Importing this package does not import the `anthropic` SDK.
"""

import os

from medquery_api.llm.base import (
    AnswerPart,
    Citation,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    LLMUnavailableError,
    SourceDocument,
    TokenUsage,
)
from medquery_api.llm.claude import DEFAULT_EFFORT, DEFAULT_MODEL, ClaudeProvider
from medquery_api.llm.mock import MockProvider

PROVIDER_ENV = "MEDQUERY_LLM_PROVIDER"
MODEL_ENV = "MEDQUERY_LLM_MODEL"
EFFORT_ENV = "MEDQUERY_LLM_EFFORT"
PROVIDERS = ("mock", "claude")


def create_provider(name: str | None = None) -> LLMProvider:
    """The configured provider. Creating it makes no network call and needs no credentials."""
    chosen = (name or os.environ.get(PROVIDER_ENV) or "mock").strip().lower()
    if chosen == "mock":
        return MockProvider()
    if chosen == "claude":
        return ClaudeProvider(
            model=os.environ.get(MODEL_ENV) or DEFAULT_MODEL,
            effort=os.environ.get(EFFORT_ENV) or DEFAULT_EFFORT,
        )
    raise ValueError(f"unknown LLM provider {chosen!r}; choose from {', '.join(PROVIDERS)}")


__all__ = [
    "AnswerPart",
    "Citation",
    "ClaudeProvider",
    "GenerationRequest",
    "GenerationResult",
    "LLMProvider",
    "LLMUnavailableError",
    "MockProvider",
    "SourceDocument",
    "TokenUsage",
    "create_provider",
]
