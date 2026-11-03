"""Claude, through Anthropic's Messages API (PRD 3.1).

Each source document is sent as its own document block, verbatim, with citations enabled, and
the question follows the documents. Refused requests are re-run server-side on Anthropic's
recommended fallback model (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`);
the result records whether that happened and which model answered.

Credentials are never handled here: the SDK reads them from the environment
(ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN) or an `ant auth login` profile. The `anthropic`
package is imported only when a request is made, so importing this module stays fast.
"""

import time
from collections.abc import Callable
from typing import Any

from medquery_api.llm.base import (
    AnswerPart,
    Citation,
    GenerationRequest,
    GenerationResult,
    LLMUnavailableError,
    TokenUsage,
)

DEFAULT_MODEL = "claude-sonnet-5-5"
# "low" suits short question answering (Anthropic's guidance for chat); to be measured in 3.3.
DEFAULT_EFFORT = "low"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _default_client_factory(timeout: float, max_retries: int) -> Any:
    import anthropic  # heavy: imported only when a real client is needed

    return anthropic.Anthropic(timeout=timeout, max_retries=max_retries)


class ClaudeProvider:
    name = "claude"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        client_factory: Callable[[float, int], Any] = _default_client_factory,
    ):
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}; got {effort!r}")
        self.model = model
        self.effort = effort
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._client_factory = client_factory

    def __repr__(self) -> str:
        return f"ClaudeProvider(model={self.model!r}, effort={self.effort!r})"

    def build_params(self, request: GenerationRequest) -> dict[str, Any]:
        """The keyword arguments for `client.beta.messages.create`."""
        content: list[dict[str, Any]] = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": document.text},
                "title": document.title,
                "citations": {"enabled": True},
            }
            for document in request.documents
        ]
        content.append({"type": "text", "text": request.question})
        return {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import anthropic

        params = self.build_params(request)
        started = time.perf_counter()
        try:
            if self._client is None:
                self._client = self._client_factory(self._timeout, self._max_retries)
            response = self._client.beta.messages.create(**params)
        except anthropic.AuthenticationError as error:
            raise LLMUnavailableError(
                "Claude rejected the credentials. Check ANTHROPIC_API_KEY or `ant auth login`."
            ) from error
        except anthropic.PermissionDeniedError as error:
            raise LLMUnavailableError(
                "The Anthropic credentials are not allowed to use this model."
            ) from error
        except anthropic.RateLimitError as error:
            raise LLMUnavailableError(
                "Claude's rate limit was reached. Try again shortly."
            ) from error
        except anthropic.APITimeoutError as error:
            raise LLMUnavailableError("Claude did not answer in time.") from error
        except anthropic.APIConnectionError as error:
            raise LLMUnavailableError("Could not reach Anthropic's API.") from error
        except anthropic.APIStatusError as error:
            raise LLMUnavailableError(
                f"Anthropic's API returned HTTP {error.status_code} ({error.type or 'error'})."
            ) from error
        except anthropic.AnthropicError as error:  # e.g. no credentials could be found
            raise LLMUnavailableError(
                "No usable Anthropic credentials were found. Set ANTHROPIC_API_KEY or run "
                "`ant auth login`."
            ) from error
        latency_ms = (time.perf_counter() - started) * 1000
        return parse_response(response, latency_ms)


def parse_response(response: Any, latency_ms: float) -> GenerationResult:
    """Turn a Messages API response into a GenerationResult.

    Text blocks become answer parts with their character-location citations; thinking and
    fallback blocks are not part of the answer.
    """
    parts: list[AnswerPart] = []
    fell_back = False
    for block in response.content:
        if block.type == "text":
            citations = tuple(
                Citation(
                    document_index=c.document_index,
                    start_char=c.start_char_index,
                    end_char=c.end_char_index,
                    cited_text=c.cited_text,
                )
                for c in (block.citations or [])
                if c.type == "char_location"
            )
            parts.append(AnswerPart(block.text, citations))
        elif block.type == "fallback":
            fell_back = True
    usage = response.usage
    iterations = getattr(usage, "iterations", None) or []
    fell_back = fell_back or any(getattr(i, "type", None) == "fallback_message" for i in iterations)
    stop_details = getattr(response, "stop_details", None)
    return GenerationResult(
        provider="claude",
        model=response.model,
        parts=tuple(parts),
        stop_reason=response.stop_reason or "unknown",
        usage=TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        ),
        latency_ms=latency_ms,
        refusal_category=getattr(stop_details, "category", None) if stop_details else None,
        fell_back=fell_back,
    )
