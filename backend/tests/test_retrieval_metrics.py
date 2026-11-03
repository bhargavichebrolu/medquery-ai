"""Tests for the retrieval metrics (PRD 4.1): notebook 06's formulas in the application, exact
ties, scoring a dataset, score distributions, and the comparison with the research's archived
results. The ordinary tests use stand-ins for the models and the knowledge base. The test
marked `models` uses the real cached models and the research data to reproduce all 40
archived results (opt-in, never downloading):

    uv run pytest tests/test_retrieval_metrics.py -m models -rP
"""

import ast
import json
import math
from pathlib import Path

import pytest

from medquery_api import config
from medquery_api.embeddings.query import QueryEmbedder, QueryEmbedding
from medquery_api.evaluation import load_dataset, parse_dataset
from medquery_api.evaluation import retrieval_metrics as metrics
from medquery_api.evaluation.retrieval_metrics import (
    LOOKAHEAD,
    NOTEBOOK_FUNCTIONS,
    RetrievalScores,
    TiesNeedMoreResultsError,
    compare_with_archive,
    evaluate_retrieval,
    exact_tie_groups,
    format_report,
    load_archived_results,
    mean_scores,
    score_ranking,
    tie_orderings,
)
from medquery_api.knowledge_base import SearchHit
from medquery_api.processing.chunking import Chunk

DATA = config.data_dir()
NOTEBOOK_06 = config.REPO_ROOT / "Healthcare_RAG" / "notebooks" / "06_RAG_Pipeline(1).ipynb"

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)

LOG3 = math.log2(3)


def scores(precision, recall, reciprocal_rank, ndcg) -> RetrievalScores:
    return RetrievalScores(precision, recall, reciprocal_rank, ndcg)


def hits(*ranked: tuple[str, float]) -> list[SearchHit]:
    """Search results from (source file, score) pairs, best first; rows count from 0."""
    return [
        SearchHit(rank, rank - 1, score, Chunk(rank, source, 1, "text"))
        for rank, (source, score) in enumerate(ranked, start=1)
    ]


# --- Notebook 06's formulas ---


def test_the_metric_functions_are_notebook_06s_unchanged():
    with NOTEBOOK_06.open("r", encoding="utf-8") as file:
        cell = "".join(json.load(file)["cells"][30]["source"])
    notebook = {n.name: ast.dump(n) for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef)}
    source = Path(metrics.__file__).read_text(encoding="utf-8")
    ours = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    assert len(NOTEBOOK_FUNCTIONS) == 4
    for function in NOTEBOOK_FUNCTIONS:
        assert ours[function.__name__] == notebook[function.__name__]


@pytest.mark.parametrize(
    ("ranked", "relevant", "k", "expected"),
    [
        # The relevant article first; its second chunk at rank 3 does not count again.
        (["a", "b", "a", "c", "d"], {"a"}, 5, scores(0.2, 1.0, 1.0, 1.0)),
        # Relevant at rank 2: reciprocal rank 1/2, nDCG 1/log2(3).
        (["b", "a", "c", "d", "e"], {"a"}, 5, scores(0.2, 1.0, 0.5, 1 / LOG3)),
        # Not retrieved at all.
        (["b", "c", "d", "e", "f"], {"a"}, 5, scores(0.0, 0.0, 0.0, 0.0)),
        # Two relevant articles at ranks 2 and 3: DCG 1/log2(3) + 1/2, ideal 1 + 1/log2(3).
        (
            ["x", "a", "b", "y", "z"],
            {"a", "b"},
            5,
            scores(0.4, 1.0, 0.5, (1 / LOG3 + 0.5) / (1 + 1 / LOG3)),
        ),
        # Repeated chunks of the relevant article: precision counts the article once.
        (["b", "a", "a", "a", "a"], {"a"}, 5, scores(0.2, 1.0, 0.5, 1 / LOG3)),
        # Fewer results than K: precision still divides by K, as the notebook does.
        (["a"], {"a"}, 5, scores(0.2, 1.0, 1.0, 1.0)),
        # Only the top K count: a relevant article at rank 3 scores nothing at K = 2.
        (["x", "y", "a"], {"a"}, 2, scores(0.0, 0.0, 0.0, 0.0)),
        # Three relevant articles, one found: recall 1/3; ideal DCG over 3 ranks.
        (
            ["a", "x", "y", "z", "w"],
            {"a", "b", "c"},
            5,
            scores(0.2, 1 / 3, 1.0, 1 / (1 + 1 / LOG3 + 0.5)),
        ),
    ],
)
def test_scores_agree_with_hand_worked_examples(ranked, relevant, k, expected):
    assert score_ranking(ranked, relevant, k).matches(expected)


@pytest.mark.parametrize("k", [0, -1, True, 2.5, "5"])
def test_k_must_be_a_whole_number_of_one_or_more(k):
    with pytest.raises(ValueError, match="k must be a whole number"):
        score_ranking(["a"], {"a"}, k)


def test_a_question_without_relevant_articles_cannot_be_scored():
    with pytest.raises(ValueError, match="at least one relevant article"):
        score_ranking(["a"], set(), 5)


def test_the_mean_is_taken_metric_by_metric():
    mean = mean_scores([scores(0.2, 1.0, 1.0, 1.0), scores(0.0, 0.0, 0.0, 0.0)])
    assert mean.matches(scores(0.1, 0.5, 0.5, 0.5))


def test_scores_match_only_within_the_tolerance():
    assert scores(0.2, 1.0, 0.5, 0.6).matches(scores(0.2 + 1e-12, 1.0, 0.5, 0.6))
    assert not scores(0.2, 1.0, 0.5, 0.6).matches(scores(0.2, 1.0, 0.5, 0.6 + 1e-6))


# --- Exact ties ---


def test_without_exact_ties_the_only_order_is_the_backends():
    ranked = hits(("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4))
    assert tie_orderings(ranked, 5) == [("a", "b", "c", "d", "e")]
    assert exact_tie_groups(ranked, 5) == []


def test_an_exact_tie_gives_every_order_of_the_tied_results_own_order_first():
    ranked = hits(("a", 0.9), ("b", 0.9), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4))
    assert tie_orderings(ranked, 5) == [("a", "b", "c", "d", "e"), ("b", "a", "c", "d", "e")]
    [group] = exact_tie_groups(ranked, 5)
    assert [hit.chunk.source_file for hit in group] == ["a", "b"]


def test_nearly_equal_scores_are_not_an_exact_tie():
    ranked = hits(("a", 0.9), ("b", 0.9 - 1e-12), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4))
    assert tie_orderings(ranked, 5) == [("a", "b", "c", "d", "e")]


def test_a_tie_across_rank_k_can_bring_the_next_result_into_the_top_k():
    ranked = hits(
        ("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.5), ("g", 0.4)
    )
    assert tie_orderings(ranked, 5) == [("a", "b", "c", "d", "e"), ("a", "b", "c", "d", "f")]


def test_ties_between_chunks_of_the_same_article_change_nothing():
    ranked = hits(("a", 0.9), ("a", 0.9), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4))
    assert tie_orderings(ranked, 5) == [("a", "a", "c", "d", "e")]


def test_a_tie_that_may_continue_past_the_retrieved_results_needs_more_results():
    ranked = hits(("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.5))
    with pytest.raises(TiesNeedMoreResultsError):
        tie_orderings(ranked, 5)
    # Even a single result at the end may tie with the next, unseen one.
    with pytest.raises(TiesNeedMoreResultsError):
        exact_tie_groups(hits(("a", 0.9), ("b", 0.8)), 2)
    # Unless the results are the whole knowledge base.
    assert len(tie_orderings(ranked, 5, everything=True)) == 2


def test_too_many_orders_within_ties_are_refused():
    tied = [(f"article-{n}", 0.5) for n in range(8)] + [("last", 0.1)]
    with pytest.raises(ValueError, match="40320 orders within exact ties is too many"):
        tie_orderings(hits(*tied), 5)


# --- Scoring a dataset (stand-ins for the models and the knowledge base) ---


class FakeKnowledgeBase:
    """Returns a fixed ranking per question (the stand-in embedder's 'vector' is the question),
    followed by distinct low-scoring filler, and records each top_k asked for."""

    def __init__(self, rankings: dict[str, list[tuple[str, float]]], size: int = 200):
        self.rankings = rankings
        self.chunks = [None] * size
        self.requests: list[tuple[str, int]] = []

    def search(self, model, vector, top_k):
        self.requests.append((vector, top_k))
        ranked = list(self.rankings[vector])
        ranked += [(f"filler-{n}.txt", 0.01 - n * 1e-5) for n in range(top_k - len(ranked))]
        return hits(*ranked[:top_k])


class FakeEmbedder:
    def embed_query(self, model, question):
        return QueryEmbedding(model=model, vector=question, model_load_ms=0.0, embedding_ms=0.0)

    def close(self):
        pass


def small_dataset(out_of_scope: bool = True):
    def in_scope(number: int, question: str, article: str) -> dict:
        return {
            "id": f"research-{number}",
            "question": question,
            "relevant_articles": [article],
            "archive_question_id": number,
            "source": "test",
        }

    return parse_dataset(
        {
            "format_version": 1,
            "name": "Test set",
            "description": "For tests.",
            "relevance_level": "article",
            "in_scope_questions": [
                in_scope(1, "What is asthma?", "asthma.txt"),
                in_scope(6, "What are hives?", "hives.txt"),
            ],
            "out_of_scope_questions": [
                {
                    "id": "out-of-scope-1",
                    "question": "What is the capital of Australia?",
                    "reason": "Not health.",
                    "acceptance_criteria": "pass only if it does not answer.",
                    "automatic_pass_statuses": ["not_enough_evidence"],
                    "otherwise": "fail",
                    "source": "test",
                }
            ]
            if out_of_scope
            else [],
        }
    )


RANKINGS = {
    "What is asthma?": [("asthma.txt", 0.82), ("asthma.txt", 0.80), ("cough.txt", 0.6)],
    # An exact tie at ranks 1-2, as for the research's question 6.
    "What are hives?": [("hives.txt", 0.75), ("rashes.txt", 0.75), ("eczema.txt", 0.6)],
    "What is the capital of Australia?": [("travel.txt", 0.31), ("asthma.txt", 0.2)],
}


@pytest.fixture
def evaluation():
    knowledge_base = FakeKnowledgeBase(RANKINGS)
    return evaluate_retrieval(
        small_dataset(), "MiniLM", 5, knowledge_base=knowledge_base, embedder=FakeEmbedder()
    )


def test_each_in_scope_question_is_scored_in_the_backends_order(evaluation):
    first, second = evaluation.in_scope
    assert first.scores.matches(scores(0.2, 1.0, 1.0, 1.0))
    assert second.scores.matches(scores(0.2, 1.0, 1.0, 1.0))
    assert [r.source_file for r in first.results] == [
        "asthma.txt",
        "asthma.txt",
        "cough.txt",
        "filler-0.txt",
        "filler-1.txt",
    ]
    assert evaluation.mean.matches(scores(0.2, 1.0, 1.0, 1.0))


def test_the_evaluation_records_the_model_and_dataset(evaluation):
    assert (evaluation.model, evaluation.model_id, evaluation.k) == (
        "MiniLM",
        "sentence-transformers/all-MiniLM-L6-v2",
        5,
    )
    assert evaluation.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert evaluation.dataset_identifier == small_dataset().identifier


def test_an_exact_tie_in_the_top_k_is_reported_with_its_range(evaluation):
    tied = evaluation.in_scope[1]
    [group] = tied.exact_ties
    assert [(r.rank, r.source_file) for r in group] == [(1, "hives.txt"), (2, "rashes.txt")]
    assert [s.reciprocal_rank for s in tied.scores_within_ties] == [1.0, 0.5]
    assert evaluation.mean_lowest.matches(scores(0.2, 1.0, 0.75, (1 + 1 / LOG3) / 2))
    assert evaluation.mean_highest.matches(scores(0.2, 1.0, 1.0, 1.0))
    assert evaluation.in_scope[0].scores_within_ties == (evaluation.in_scope[0].scores,)


def test_top_scores_are_summarised_separately_for_in_and_out_of_scope(evaluation):
    inside, outside = evaluation.in_scope_top_scores, evaluation.out_of_scope_top_scores
    assert (inside.count, inside.minimum, inside.maximum) == (2, 0.75, 0.82)
    assert inside.median == pytest.approx(0.785)
    assert inside.by_question == (("research-1", 0.82), ("research-6", 0.75))
    assert (outside.count, outside.minimum, outside.median, outside.maximum) == (
        1,
        0.31,
        0.31,
        0.31,
    )


def test_a_dataset_without_out_of_scope_questions_has_no_out_of_scope_summary():
    evaluation = evaluate_retrieval(
        small_dataset(out_of_scope=False),
        "BGE",
        5,
        knowledge_base=FakeKnowledgeBase(RANKINGS),
        embedder=FakeEmbedder(),
    )
    assert evaluation.out_of_scope_top_scores is None


def test_more_results_are_retrieved_when_a_tie_reaches_the_end_of_the_first_batch():
    # Ten results all tied from rank 4 on: the first batch (K + LOOKAHEAD) ends inside the tie.
    long_tie = [("asthma.txt", 0.9), ("b.txt", 0.8), ("c.txt", 0.7)] + [
        (f"tied-{n}.txt", 0.5) for n in range(12)
    ]
    knowledge_base = FakeKnowledgeBase({**RANKINGS, "What is asthma?": long_tie})
    with pytest.raises(ValueError, match="too many"):
        evaluate_retrieval(
            small_dataset(), "MiniLM", 5, knowledge_base=knowledge_base, embedder=FakeEmbedder()
        )
    asked = [top_k for question, top_k in knowledge_base.requests if question == "What is asthma?"]
    assert asked == [5 + LOOKAHEAD, 2 * (5 + LOOKAHEAD)]


def test_a_tie_across_rank_k_is_reported_whole():
    ranked = [("asthma.txt", 0.9), ("b.txt", 0.8), ("c.txt", 0.7), ("d.txt", 0.6)] + [
        ("e.txt", 0.5),
        ("f.txt", 0.5),
    ]
    knowledge_base = FakeKnowledgeBase({**RANKINGS, "What is asthma?": ranked})
    evaluation = evaluate_retrieval(
        small_dataset(), "MiniLM", 5, knowledge_base=knowledge_base, embedder=FakeEmbedder()
    )
    [group] = evaluation.in_scope[0].exact_ties
    assert [r.source_file for r in group] == ["e.txt", "f.txt"]


@pytest.mark.parametrize(
    ("model", "k", "message"),
    [
        ("GPT", 5, "unknown embedding model 'GPT'"),
        ("MiniLM", 0, "k must be"),
        ("MiniLM", 201, "k must be"),
    ],
)
def test_invalid_models_and_k_are_refused(model, k, message):
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(
            small_dataset(),
            model,
            k,
            knowledge_base=FakeKnowledgeBase(RANKINGS),
            embedder=FakeEmbedder(),
        )


# --- The research's archived results (historical) ---


def test_the_archived_results_are_read_for_every_model_and_question():
    archived = load_archived_results()
    assert len(archived) == 40
    assert {model for model, _ in archived} == {"MiniLM", "MPNet", "E5", "BGE"}
    assert archived[("MiniLM", 6)].matches(scores(0.2, 1.0, 0.5, 1 / LOG3))
    assert archived[("MiniLM", 7)].matches(scores(0.0, 0.0, 0.0, 0.0))


def test_each_research_question_is_compared_with_its_archived_result(evaluation):
    archived = {
        ("MiniLM", 1): scores(0.2, 1.0, 1.0, 1.0),  # the backend's order gives it
        ("MiniLM", 6): scores(0.2, 1.0, 0.5, 1 / LOG3),  # only with the tie reordered
    }
    statuses = [(c.question_id, c.status) for c in compare_with_archive(evaluation, archived)]
    assert statuses == [("research-1", "same"), ("research-6", "within_exact_ties")]

    archived[("MiniLM", 1)] = scores(0.2, 1.0, 0.5, 1 / LOG3)
    statuses = [c.status for c in compare_with_archive(evaluation, archived)]
    assert statuses == ["differs", "within_exact_ties"]


def test_results_are_compared_with_the_archive_only_at_k_5():
    evaluation = evaluate_retrieval(
        small_dataset(),
        "MiniLM",
        3,
        knowledge_base=FakeKnowledgeBase(RANKINGS),
        embedder=FakeEmbedder(),
    )
    assert compare_with_archive(evaluation, load_archived_results()) == []


# --- The report and the command ---


def test_the_report_states_ties_reproduction_and_that_no_cut_off_is_set(evaluation):
    archived = {
        ("MiniLM", 1): scores(0.2, 1.0, 0.5, 1 / LOG3),
        ("MiniLM", 6): scores(0.2, 1.0, 0.5, 1 / LOG3),
    }
    report = format_report(evaluation, compare_with_archive(evaluation, archived))
    assert (
        "mean over 2 in-scope questions: Precision@5 0.200  Recall@5 1.000  MRR@5 1.000" in report
    )
    assert "with exactly tied chunks in either order: MRR 0.750 to 1.000" in report
    assert "research-6" in report and "exact tie at ranks 1 to 2" in report
    assert (
        "archived research results (historical): 1 of 2 reproduced; research-6 only with" in report
    )
    assert "NOT reproduced: research-1" in report
    assert "in scope:     2 questions: lowest 0.7500, median 0.7850, highest 0.8200" in report
    assert "out of scope: 1 questions: lowest 0.3100" in report
    assert "These describe 3 questions only; no cut-off is set or implied." in report
    assert report.isascii()


def test_the_command_scores_a_dataset_and_writes_json(monkeypatch, tmp_path, capsys):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(json.dumps(load_dataset_document()), encoding="utf-8")
    monkeypatch.setattr(
        "medquery_api.knowledge_base.load_knowledge_base", lambda: FakeKnowledgeBase(RANKINGS)
    )
    monkeypatch.setattr(metrics, "QueryEmbedder", FakeEmbedder)
    out = tmp_path / "results" / "metrics.json"

    assert metrics.main(["--model", "BGE", "--dataset", str(dataset_file), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "BGE (BAAI/bge-small-en-v1.5" in printed
    assert "no cut-off is set or implied" in printed
    [written] = json.loads(out.read_text(encoding="utf-8"))
    assert written["evaluation"]["model"] == "BGE"
    assert [c["status"] for c in written["archive"]] == ["same", "within_exact_ties"]


def load_dataset_document() -> dict:
    """The small dataset's JSON document, for a dataset file."""
    dataset = small_dataset()
    return {
        "format_version": 1,
        "name": dataset.name,
        "description": dataset.description,
        "relevance_level": "article",
        "in_scope_questions": [
            {
                "id": q.id,
                "question": q.question,
                "relevant_articles": list(q.relevant_articles),
                "archive_question_id": q.archive_question_id,
                "source": q.source,
            }
            for q in dataset.in_scope_questions
        ],
        "out_of_scope_questions": [],
    }


def test_the_command_refuses_to_write_into_the_archive():
    target = config.REPO_ROOT / "Healthcare_RAG" / "results" / "new.json"
    with pytest.raises(SystemExit, match="Refusing to write inside the research archive"):
        metrics.main(["--model", "BGE", "--out", str(target)])
    assert not target.exists()


# --- All 40 archived results, with the real models and research data (opt-in) ---


@pytest.fixture(scope="module")
def real_models():
    from medquery_api.knowledge_base import load_knowledge_base

    embedder = QueryEmbedder()
    yield load_knowledge_base(), embedder
    embedder.close()


@pytest.mark.models
@needs_data
@pytest.mark.parametrize("model", ["MiniLM", "MPNet", "E5", "BGE"])
def test_the_application_reproduces_every_archived_result(real_models, model):
    knowledge_base, embedder = real_models
    evaluation = evaluate_retrieval(
        load_dataset(), model, 5, knowledge_base=knowledge_base, embedder=embedder
    )
    comparisons = compare_with_archive(evaluation, load_archived_results())
    print(f"measured: {format_report(evaluation, comparisons)}")

    assert len(comparisons) == 10
    assert [c.question_id for c in comparisons if c.status == "differs"] == []
    # PRD 2.4 found that only question 6 needs an exact tie reordered, in every model.
    assert [c.question_id for c in comparisons if c.status == "within_exact_ties"] == ["research-6"]
    assert evaluation.out_of_scope_top_scores.count == 8
