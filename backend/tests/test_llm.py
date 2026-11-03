"""Tests for medquery_api.llm (PRD 3.1).

Ordinary tests never call Anthropic: the mock provider needs nothing, and every Claude test
injects a stand-in client. The single test that calls the real API is marked `llm` and is
opt-in (`uv run pytest -m llm`); run it only after approving its cost.
"""

import subprocess
import sys

import httpx2
import pytest

from medquery_api.llm import (
    ClaudeProvider,
    GenerationRequest,
    LLMUnavailableError,
    MockProvider,
    SourceDocument,
    create_provider,
)
from medquery_api.llm.claude import FALLBACK_BETA, parse_response
from medquery_api.llm.mock import NOT_ENOUGH_EVIDENCE

ASTHMA = "# Asthma\n\nAsthma is a common condition that affects your breathing.\n\nSee https://www.example.org/help."
HIVES = "## Symptoms of hives\n\nThe main symptom of hives is an itchy rash."
SENTENCE = "Asthma is a common condition that affects your breathing."


def request(documents=(ASTHMA, HIVES), question="What are the symptoms of asthma?"):
    return GenerationRequest(
        system="Answer only from the documents.",
        documents=tuple(
            SourceDocument(title=f"Source {i + 1}", text=text) for i, text in enumerate(documents)
        ),
        question=question,
    )


# --- Requests ---


@pytest.mark.parametrize(
    ("question", "max_tokens", "message"),
    [("   ", 100, "question must not be empty"), ("Why?", 0, "max_tokens must be at least 1")],
)
def test_invalid_requests_are_rejected(question, max_tokens, message):
    with pytest.raises(ValueError, match=message):
        GenerationRequest(system="s", documents=(), question=question, max_tokens=max_tokens)


# --- Mock provider ---


def test_the_mock_answers_by_citing_the_first_line_of_the_first_document():
    provider = MockProvider()
    result = provider.generate(request())

    cited = result.parts[1].citations[0]
    assert result.provider == "mock" and result.stop_reason == "end_turn" and not result.refused
    assert cited.document_index == 0
    assert ASTHMA[cited.start_char : cited.end_char] == cited.cited_text == "# Asthma"
    assert result.text == "According to the provided documents: # Asthma"
    assert provider.requests == [request()]


def test_the_mock_says_the_evidence_is_not_enough_without_documents():
    result = MockProvider().generate(request(documents=()))

    assert result.text == NOT_ENOUGH_EVIDENCE
    assert all(not part.citations for part in result.parts)


def test_the_mock_can_return_any_result_a_test_needs():
    canned = MockProvider().generate(request(documents=()))
    provider = MockProvider(responder=lambda _request: canned)

    assert provider.generate(request()) is canned


# --- Choosing a provider ---


def test_the_default_provider_is_the_mock(monkeypatch):
    monkeypatch.delenv("MEDQUERY_LLM_PROVIDER", raising=False)

    assert isinstance(create_provider(), MockProvider)


def test_claude_is_chosen_by_configuration_with_its_defaults(monkeypatch):
    monkeypatch.setenv("MEDQUERY_LLM_PROVIDER", "claude")
    monkeypatch.delenv("MEDQUERY_LLM_MODEL", raising=False)
    monkeypatch.delenv("MEDQUERY_LLM_EFFORT", raising=False)

    provider = create_provider()

    assert isinstance(provider, ClaudeProvider)
    assert (provider.model, provider.effort) == ("claude-sonnet-5-5", "low")


def test_model_and_effort_can_be_configured(monkeypatch):
    monkeypatch.setenv("MEDQUERY_LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("MEDQUERY_LLM_EFFORT", "medium")

    provider = create_provider("claude")

    assert (provider.model, provider.effort) == ("claude-haiku-5-5", "medium")


@pytest.mark.parametrize(("name", "message"), [("gpt", "unknown LLM provider"), (None, "effort")])
def test_bad_configuration_is_rejected(monkeypatch, name, message):
    monkeypatch.setenv("MEDQUERY_LLM_PROVIDER", "claude")
    monkeypatch.setenv("MEDQUERY_LLM_EFFORT", "extreme")

    with pytest.raises(ValueError, match=message):
        create_provider(name)


def test_importing_the_package_or_using_the_mock_does_not_load_the_sdk():
    code = (
        "import sys\n"
        "from medquery_api.llm import GenerationRequest, MockProvider, create_provider\n"
        "create_provider('claude')\n"
        "MockProvider().generate(GenerationRequest(system='s', documents=(), question='q'))\n"
        "print('anthropic' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "False"


# --- Claude provider: the request ---


def test_each_chunk_is_sent_verbatim_as_its_own_cited_document_before_the_question():
    params = ClaudeProvider().build_params(request())
    content = params["messages"][0]["content"]

    assert params["model"] == "claude-sonnet-5-5"
    assert params["system"] == "Answer only from the documents."
    assert params["max_tokens"] == 4096
    assert params["output_config"] == {"effort": "low"}
    assert params["betas"] == [FALLBACK_BETA] and params["fallbacks"] == "default"
    assert [block["type"] for block in content] == ["document", "document", "text"]
    assert content[0] == {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": ASTHMA},
        "title": "Source 1",
        "citations": {"enabled": True},
    }
    assert content[1]["source"]["data"] == HIVES
    assert content[2] == {"type": "text", "text": "What are the symptoms of asthma?"}


def test_the_provider_does_not_show_its_client_or_credentials():
    assert repr(ClaudeProvider()) == "ClaudeProvider(model='claude-sonnet-5-5', effort='low')"


# --- Claude provider: the response ---


def message(**overrides):
    """A Messages API response, as the SDK's own BetaMessage model."""
    from anthropic.types.beta import BetaMessage

    data = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "The main symptoms include ", "citations": None},
            {
                "type": "text",
                "text": "breathing problems",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": SENTENCE,
                        "document_index": 0,
                        "document_title": "Source 1",
                        "start_char_index": ASTHMA.index(SENTENCE),
                        "end_char_index": ASTHMA.index(SENTENCE) + len(SENTENCE),
                        "file_id": None,
                    }
                ],
            },
            {"type": "text", "text": ".", "citations": None},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 812,
            "output_tokens": 64,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    data.update(overrides)
    return BetaMessage.model_validate(data)


class FakeClient:
    """Stands in for anthropic.Anthropic: records the call and returns or raises."""

    def __init__(self, outcome):
        self.calls = []
        self.outcome = outcome
        self.beta = self
        self.messages = self

    def create(self, **params):
        self.calls.append(params)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_text_and_citations_are_read_from_the_response():
    client = FakeClient(message())
    result = ClaudeProvider(client=client).generate(request())

    assert client.calls == [ClaudeProvider().build_params(request())]
    assert result.text == "The main symptoms include breathing problems."
    assert [len(p.citations) for p in result.parts] == [0, 1, 0]
    cited = result.parts[1].citations[0]
    start = ASTHMA.index(SENTENCE)
    assert (cited.document_index, cited.start_char, cited.end_char) == (
        0,
        start,
        start + len(SENTENCE),
    )
    assert ASTHMA[cited.start_char : cited.end_char] == cited.cited_text
    assert (result.provider, result.model, result.stop_reason) == (
        "claude",
        "claude-sonnet-5-5",
        "end_turn",
    )
    assert (result.usage.input_tokens, result.usage.output_tokens) == (812, 64)
    assert not result.refused and not result.fell_back and result.latency_ms >= 0


def test_a_refusal_is_reported_with_its_category():
    response = message(
        content=[],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "bio", "explanation": "Declined."},
    )

    result = parse_response(response, latency_ms=5.0)

    assert result.refused and result.refusal_category == "bio" and result.text == ""


def test_an_answer_from_a_fallback_model_is_recorded():
    response = message(
        model="claude-opus-4-8",
        content=[
            {
                "type": "fallback",
                "from": {"model": "claude-sonnet-5-5"},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "Answer.", "citations": None},
        ],
    )

    result = parse_response(response, latency_ms=5.0)

    assert result.fell_back and result.model == "claude-opus-4-8" and result.text == "Answer."


def test_a_fallback_is_also_recognised_from_usage_alone():
    # Turns served by a fallback model can carry no fallback block; usage still says so.
    usage = {
        "input_tokens": 812,
        "output_tokens": 64,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iterations": [
            {
                "type": "fallback_message",
                "model": "claude-opus-4-8",
                "input_tokens": 812,
                "output_tokens": 64,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
        ],
    }

    result = parse_response(message(model="claude-opus-4-8", usage=usage), latency_ms=5.0)

    assert result.fell_back and result.model == "claude-opus-4-8"


# --- Claude provider: errors ---


def sdk_error(kind, status=None):
    import anthropic

    http_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    if kind in ("APIConnectionError", "APITimeoutError"):
        return getattr(anthropic, kind)(request=http_request)
    response = httpx2.Response(status, request=http_request)
    return getattr(anthropic, kind)("error from the API", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (lambda: sdk_error("AuthenticationError", 401), "rejected the credentials"),
        (lambda: sdk_error("PermissionDeniedError", 403), "not allowed to use this model"),
        (lambda: sdk_error("RateLimitError", 429), "rate limit"),
        (lambda: sdk_error("InternalServerError", 529), "HTTP 529"),
        (lambda: sdk_error("APITimeoutError"), "did not answer in time"),
        (lambda: sdk_error("APIConnectionError"), "Could not reach"),
    ],
    ids=["401", "403", "429", "529", "timeout", "connection"],
)
def test_api_errors_become_safe_messages(monkeypatch, error, message):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret-value")
    provider = ClaudeProvider(client=FakeClient(error()))

    with pytest.raises(LLMUnavailableError, match=message) as raised:
        provider.generate(request())

    assert "sk-ant-test-secret-value" not in str(raised.value)
    assert ASTHMA not in str(raised.value)


def test_missing_credentials_become_a_safe_message():
    import anthropic

    def no_credentials(timeout, max_retries):
        raise anthropic.AnthropicError("Could not resolve authentication method")

    provider = ClaudeProvider(client_factory=no_credentials)

    with pytest.raises(LLMUnavailableError, match="No usable Anthropic credentials"):
        provider.generate(request())


# --- The real API (opt-in: uv run pytest -m llm) ---


@pytest.mark.llm
def test_a_real_claude_answer_cites_the_supplied_document():
    provider = ClaudeProvider()
    result = provider.generate(
        GenerationRequest(
            system="Answer only from the documents, citing them. Be brief.",
            documents=(SourceDocument(title="Asthma (chunk 1)", text=ASTHMA),),
            question="What does asthma affect?",
            max_tokens=1024,
        )
    )
    print(
        f"measured: model {result.model}, stop {result.stop_reason}, {result.latency_ms:.0f} ms, "
        f"tokens in {result.usage.input_tokens} / out {result.usage.output_tokens}"
    )

    assert not result.refused
    cited = [c for part in result.parts for c in part.citations]
    assert cited and all(ASTHMA[c.start_char : c.end_char] == c.cited_text for c in cited)
