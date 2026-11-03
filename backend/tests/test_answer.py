"""Tests for grounded answers: POST /api/answer, cost estimates and the review command (PRD 3.3),
with checked citations (PRD 3.4; the checks themselves are tested in test_citations.py).

Everything here uses a tiny generated knowledge base, a stand-in embedding model and the mock
LLM provider: no research data, models, network or cost. (The opt-in `llm` test in
test_llm.py is the only one that calls a real model.)
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from test_retrieval import Holder, make_data_dir

from medquery_api import main as main_module
from medquery_api.answering import load_evaluation_questions, main, review
from medquery_api.knowledge_base import load_knowledge_base
from medquery_api.knowledge_base.store import _load
from medquery_api.llm import (
    AnswerPart,
    Citation,
    ClaudeProvider,
    GenerationResult,
    LLMUnavailableError,
    MockProvider,
    TokenUsage,
)
from medquery_api.llm.context import SYSTEM_INSTRUCTIONS
from medquery_api.llm.mock import NOT_ENOUGH_EVIDENCE
from medquery_api.main import (
    app,
    get_data_dir,
    get_knowledge_base_loader,
    get_llm_provider,
    get_query_embedder,
)
from medquery_api.pricing import PRICES, estimate_cost

client = TestClient(app)
QUESTION = "What are the symptoms of asthma?"


@pytest.fixture
def setup(tmp_path):
    """The API with a tiny data folder, a stand-in embedder and a mock LLM provider."""
    make_data_dir(tmp_path)
    holder = Holder()
    state = {"provider": MockProvider(), "holder": holder, "kb_loads": 0, "folder": tmp_path}

    def loader():
        def load():
            state["kb_loads"] += 1
            return load_knowledge_base(tmp_path)

        return load

    app.dependency_overrides[get_data_dir] = lambda: tmp_path
    app.dependency_overrides[get_knowledge_base_loader] = loader
    app.dependency_overrides[get_query_embedder] = lambda: holder.embedder
    app.dependency_overrides[get_llm_provider] = lambda: state["provider"]
    _load.cache_clear()
    yield state
    app.dependency_overrides.clear()
    _load.cache_clear()


def ask(question=QUESTION, model="BGE", **extra):
    return client.post("/api/answer", json={"question": question, "model": model, **extra})


def scripted(result):
    return MockProvider(responder=lambda _request: result)


def result(**overrides):
    values = {
        "provider": "claude",
        "model": "claude-sonnet-5-5",
        "parts": (AnswerPart("Answer."),),
        "stop_reason": "end_turn",
        "usage": TokenUsage(input_tokens=1200, output_tokens=300),
        "latency_ms": 1500.0,
    }
    values.update(overrides)
    return GenerationResult(**values)


# --- The answer ---


def test_an_answer_comes_with_its_sources_citations_usage_cost_and_timings(setup):
    response = ask(topK=3)
    body = response.json()

    assert response.status_code == 200
    assert list(body) == [
        "question",
        "embeddingModel",
        "topK",
        "llm",
        "parts",
        "grounding",
        "sentences",
        "citationStatus",
        "citationCheck",
        "droppedCitations",
        "sources",
        "stopReason",
        "refused",
        "refusalCategory",
        "usage",
        "cost",
        "timings",
    ]
    assert body["llm"] == {"provider": "mock", "model": "mock", "fellBack": False}
    assert body["citationStatus"] == "checked"
    assert body["citationCheck"] == {
        "total": 1,
        "valid": 1,
        "dropped": {"unknownDocument": 0, "invalidRange": 0, "textMismatch": 0},
        "validityRate": 1.0,
        "sentences": 1,
        "supportedSentences": 1,
        "groundingCoverage": 1.0,
    }
    assert body["droppedCitations"] == []
    answer_text = "".join(part["text"] for part in body["parts"])
    assert body["grounding"] == "grounded"
    assert body["sentences"] == [{"start": 0, "end": len(answer_text), "supported": True}]
    assert [s["label"] for s in body["sources"]] == ["S1", "S2", "S3"]
    assert [s["rank"] for s in body["sources"]] == [1, 2, 3]
    first = body["sources"][0]
    assert first["text"] == f"{first['sourceFile']} part {first['chunkId']}"
    assert first["title"].endswith(f"(chunk {first['chunkId']} of {first['chunkCount']})")
    citation = body["parts"][1]["citations"][0]
    assert citation["sourceLabel"] == "S1" and citation["documentIndex"] == 0
    assert first["text"][citation["startChar"] : citation["endChar"]] == citation["citedText"]
    assert (body["stopReason"], body["refused"], body["refusalCategory"]) == (
        "end_turn",
        False,
        None,
    )
    assert body["cost"]["estimatedUsd"] == 0.0
    assert list(body["timings"]) == [
        "modelLoadMs",
        "embeddingMs",
        "searchMs",
        "generationMs",
        "totalMs",
    ]
    assert body["timings"]["totalMs"] >= body["timings"]["generationMs"]


def test_the_provider_receives_the_context_built_from_the_retrieval(setup):
    body = ask(topK=4).json()
    request = setup["provider"].requests[0]

    assert request.system == SYSTEM_INSTRUCTIONS
    assert request.question == QUESTION
    assert [d.text for d in request.documents] == [s["text"] for s in body["sources"]]
    assert [d.title for d in request.documents] == [s["title"] for s in body["sources"]]


def test_invalid_citations_are_dropped_and_reported_and_valid_ones_kept(setup):
    def respond(request):
        second = request.documents[1].text
        valid = Citation(document_index=1, start_char=0, end_char=4, cited_text=second[:4])
        unknown = Citation(document_index=7, start_char=0, end_char=4, cited_text="text")
        mismatch = Citation(document_index=0, start_char=0, end_char=4, cited_text="nope")
        parts = (
            AnswerPart("Supported claim.", (valid, unknown)),
            AnswerPart(" Unsupported claim.", (mismatch,)),
        )
        return result(parts=parts)

    setup["provider"] = MockProvider(responder=respond)
    body = ask(topK=3).json()

    assert [c["sourceLabel"] for c in body["parts"][0]["citations"]] == ["S2"]
    assert body["parts"][1]["citations"] == []
    assert body["parts"][1]["text"] == " Unsupported claim."
    assert body["droppedCitations"] == [
        {
            "partIndex": 0,
            "documentIndex": 7,
            "citedText": "text",
            "startChar": 0,
            "endChar": 4,
            "reason": "unknown_document",
        },
        {
            "partIndex": 1,
            "documentIndex": 0,
            "citedText": "nope",
            "startChar": 0,
            "endChar": 4,
            "reason": "text_mismatch",
        },
    ]
    check = body["citationCheck"]
    assert (check["total"], check["valid"], check["validityRate"]) == (3, 1, pytest.approx(1 / 3))
    assert check["dropped"] == {"unknownDocument": 1, "invalidRange": 0, "textMismatch": 1}
    assert (check["sentences"], check["supportedSentences"]) == (2, 1)
    assert check["groundingCoverage"] == 0.5


def test_a_malformed_citation_is_dropped_without_a_server_error(setup):
    malformed = Citation(document_index=None, start_char="0", end_char=None, cited_text=None)
    setup["provider"] = scripted(result(parts=(AnswerPart("Claim.", (malformed,)),)))

    response = ask()

    assert response.status_code == 200
    assert response.json()["droppedCitations"] == [
        {
            "partIndex": 0,
            "documentIndex": None,
            "citedText": None,
            "startChar": None,
            "endChar": None,
            "reason": "unknown_document",
        }
    ]


@pytest.mark.parametrize(
    ("parts", "stop_reason", "grounding", "supported"),
    [
        ((AnswerPart(NOT_ENOUGH_EVIDENCE),), "end_turn", "not_enough_evidence", [False]),
        (
            (AnswerPart("Asthma is caused by cats. It is rare."),),
            "end_turn",
            "not_grounded",
            [False, False],
        ),
        ((AnswerPart("Partial answer."),), "refusal", "refused", [False]),
    ],
    ids=["not enough evidence", "not grounded", "refused"],
)
def test_the_grounding_status_and_sentences_are_returned(
    setup, parts, stop_reason, grounding, supported
):
    setup["provider"] = scripted(result(parts=parts, stop_reason=stop_reason))

    body = ask().json()

    assert body["grounding"] == grounding
    assert [s["supported"] for s in body["sentences"]] == supported


def test_a_partly_grounded_answer_marks_which_sentences_are_supported(setup):
    def respond(request):
        text = request.documents[0].text
        valid = Citation(document_index=0, start_char=0, end_char=4, cited_text=text[:4])
        return result(
            parts=(AnswerPart("Supported claim.", (valid,)), AnswerPart(" Unsupported claim."))
        )

    setup["provider"] = MockProvider(responder=respond)
    body = ask().json()
    text = "".join(part["text"] for part in body["parts"])

    assert body["grounding"] == "partly_grounded"
    assert [(text[s["start"] : s["end"]], s["supported"]) for s in body["sentences"]] == [
        ("Supported claim.", True),
        ("Unsupported claim.", False),
    ]


def test_a_refusal_and_a_fallback_are_passed_through(setup):
    setup["provider"] = scripted(
        result(parts=(), stop_reason="refusal", refusal_category="bio", fell_back=True)
    )

    body = ask().json()

    assert (body["stopReason"], body["refused"], body["refusalCategory"]) == (
        "refusal",
        True,
        "bio",
    )
    assert body["llm"]["fellBack"] is True and body["parts"] == []
    assert body["citationCheck"]["validityRate"] is None
    assert body["citationCheck"]["groundingCoverage"] is None


def test_the_cost_of_a_claude_answer_is_estimated_from_the_dated_price(setup):
    setup["provider"] = scripted(result())

    cost = ask().json()["cost"]

    assert cost["estimatedUsd"] == pytest.approx((1200 * 2.00 + 300 * 10.00) / 1_000_000)
    assert cost["pricesCheckedOn"] == "2026-11-02"


def test_the_question_text_is_not_logged(setup, caplog):
    with caplog.at_level(logging.DEBUG):
        ask("A private question about my rash")

    assert "private question" not in caplog.text


# --- Input checks and errors ---


@pytest.mark.parametrize(
    "body",
    [
        {"model": "BGE"},
        {"question": "   ", "model": "BGE"},
        {"question": "x" * 1001, "model": "BGE"},
        {"question": "q", "model": "GPT"},
        {"question": "q", "model": "BGE", "topK": 0},
        {"question": "q", "model": "BGE", "topK": 21},
        {"question": "q", "model": "BGE", "topK": "3"},
        {"question": "q", "model": "BGE", "unexpected": 1},
    ],
    ids=[
        "no question",
        "blank",
        "too long",
        "unknown model",
        "topK 0",
        "topK 21",
        "topK text",
        "extra",
    ],
)
def test_invalid_requests_are_rejected_before_anything_runs(setup, body):
    response = client.post("/api/answer", json=body)

    assert response.status_code == 422
    assert setup["kb_loads"] == 0 and setup["holder"].loaded == []
    assert setup["provider"].requests == []


def test_a_provider_error_gives_a_safe_503(setup):
    def fail(_request):
        raise LLMUnavailableError("Claude's rate limit was reached. Try again shortly.")

    setup["provider"] = MockProvider(responder=fail)
    response = ask()

    assert response.status_code == 503
    assert response.json() == {"detail": "Claude's rate limit was reached. Try again shortly."}


def test_a_missing_embedding_model_gives_a_503_and_no_generation(setup):
    setup["holder"].missing = ("BGE",)

    response = ask()

    assert response.status_code == 503 and "download BGE" in response.json()["detail"]
    assert setup["provider"].requests == []


def test_missing_data_gives_a_503_without_paths(setup):
    folder = setup["folder"] / "nothing-here"
    app.dependency_overrides[get_knowledge_base_loader] = lambda: get_knowledge_base_loader(folder)

    response = ask()

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Research data folder not found.")
    assert str(setup["folder"]) not in response.text


# --- Cost estimates ---


def test_cost_estimates_cover_mock_known_and_unknown_models():
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )

    assert estimate_cost("mock", "mock", usage).usd == 0.0
    assert estimate_cost("claude", "claude-sonnet-5-5", usage).usd == pytest.approx(
        2.00 + 10.00 + 0.10 + 2.50
    )
    unknown = estimate_cost("claude", "claude-opus-4-8", usage)
    assert unknown.usd is None and "No checked price for claude-opus-4-8" in unknown.note
    assert all(price.checked_on == "2026-11-02" for price in PRICES.values())


# --- The review command ---


def test_the_review_answers_each_question_with_the_mock(setup):
    questions = [
        {"question_id": 1, "question": QUESTION, "relevant_sources": ["asthma.txt"]},
        {"question_id": 2, "question": "What is hives?", "relevant_sources": ["hives.txt"]},
    ]

    rows = review(
        questions,
        "BGE",
        5,
        knowledge_base=load_knowledge_base(setup["folder"]),
        embedder=setup["holder"].embedder,
        provider=MockProvider(),
    )

    assert [r["question_id"] for r in rows] == [1, 2]
    assert all(r["provider"] == "mock" and r["estimated_cost_usd"] == 0.0 for r in rows)
    assert all(len(r["sources"]) == 5 and r["citations"] for r in rows)
    assert all(r["citation_check"]["validity_rate"] == 1.0 for r in rows)
    assert all(r["dropped_citations"] == [] for r in rows)
    assert all(r["grounding"] == "grounded" and r["unsupported_sentences"] == [] for r in rows)


def test_the_review_refuses_a_paid_provider_without_permission(capsys):
    assert main(["review", "--provider", "claude"]) == 2
    assert "--allow-paid-calls" in capsys.readouterr().err


def test_the_review_will_not_write_into_the_archive():
    with pytest.raises(SystemExit, match="Refusing to write inside the research archive"):
        main(["review", "--out", "../Healthcare_RAG/results/answers.json"])


def test_the_evaluation_questions_are_the_researchs_ten():
    questions = load_evaluation_questions()

    assert [q["question_id"] for q in questions] == list(range(1, 11))
    assert all(q["historical_generator"] == "meta-llama/Llama-3.1-8B-Instruct" for q in questions)
    assert json.dumps(questions)  # plain data, ready to save for review


# --- The configured LLM (GET /api/llm) ---


def test_the_configured_provider_and_model_are_reported(setup):
    assert client.get("/api/llm").json() == {"provider": "mock", "model": "mock"}


def test_a_claude_configuration_reveals_only_the_provider_and_model(setup, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    setup["provider"] = ClaudeProvider(model="claude-sonnet-5-5", effort="low")

    response = client.get("/api/llm")

    assert response.json() == {"provider": "claude", "model": "claude-sonnet-5-5"}
    assert "sk-ant" not in response.text and "effort" not in response.text


def test_an_unknown_provider_setting_gives_a_safe_503(monkeypatch):
    monkeypatch.setenv("MEDQUERY_LLM_PROVIDER", "gpt")
    monkeypatch.setattr(main_module, "_llm_provider", None)

    response = client.get("/api/llm")

    assert response.status_code == 503
    assert "MEDQUERY_LLM_PROVIDER" in response.json()["detail"]
