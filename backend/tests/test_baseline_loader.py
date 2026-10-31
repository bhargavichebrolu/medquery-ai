from pathlib import Path

import pytest
from pydantic import ValidationError

from medquery.baseline.loader import (
    RETRIEVAL_SUMMARY_CSV,
    RetrievalMetrics,
    load_baseline,
    load_retrieval_summary,
)
from medquery.config import get_settings

MODELS = {"MiniLM", "MPNet", "E5", "BGE"}


def test_loader_parses_csv_and_tags_provenance(tmp_path: Path) -> None:
    (tmp_path / RETRIEVAL_SUMMARY_CSV).write_text(
        "Embedding Model,Precision@5,Recall@5,MRR,nDCG@5\nM,0.2,1.0,0.5,0.6\n", encoding="utf-8"
    )
    [row] = load_retrieval_summary(tmp_path)
    assert row == RetrievalMetrics(
        model="M", precision_at_5=0.2, recall_at_5=1.0, mrr=0.5, ndcg_at_5=0.6
    )
    assert row.provenance == "original_dissertation"


def test_metrics_outside_unit_interval_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalMetrics(model="M", precision_at_5=0.2, recall_at_5=1.0, mrr=1.5, ndcg_at_5=0.6)


def test_baseline_models_are_immutable() -> None:
    row = RetrievalMetrics(model="M", precision_at_5=0.2, recall_at_5=1.0, mrr=0.5, ndcg_at_5=0.6)
    with pytest.raises(ValidationError):
        row.mrr = 1.0  # type: ignore[misc]


# --- Tests against the real archive (skipped if it is not present) -----------------


@pytest.fixture(scope="module")
def baseline():
    results_dir = get_settings().archive_results_dir
    if not results_dir.is_dir():
        pytest.skip("Research archive not available on this machine")
    return load_baseline(results_dir)


@pytest.mark.archive
def test_original_retrieval_summary_values(baseline) -> None:
    assert {r.model for r in baseline.retrieval_summary} == MODELS
    mpnet = baseline.retrieval_for("MPNet")
    assert (mpnet.precision_at_5, mpnet.recall_at_5, mpnet.mrr) == (0.2, 1.0, 0.95)
    assert mpnet.ndcg_at_5 == pytest.approx(0.9630929753571458)
    assert baseline.retrieval_for("MiniLM").mrr == 0.8


@pytest.mark.archive
def test_summary_equals_mean_of_per_question_results(baseline) -> None:
    """The saved summary must be exactly derivable from the saved per-question rows."""
    for summary in baseline.retrieval_summary:
        rows = [r for r in baseline.retrieval_per_question if r.model == summary.model]
        assert len(rows) == 10
        for metric in ("precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"):
            mean = sum(getattr(r, metric) for r in rows) / len(rows)
            assert mean == pytest.approx(getattr(summary, metric)), (summary.model, metric)


@pytest.mark.archive
def test_precision_at_5_ceiling_is_documented_behaviour(baseline) -> None:
    """One relevant source per question, deduplicated, divided by k=5 -> max 0.2."""
    assert max(r.precision_at_5 for r in baseline.retrieval_per_question) == pytest.approx(0.2)


@pytest.mark.archive
def test_faiss_build_records(baseline) -> None:
    dims = {r.model: r.dimension for r in baseline.faiss_builds}
    assert dims == {"MiniLM": 384, "MPNet": 768, "E5": 768, "BGE": 384}
    assert all(r.vectors == 3796 and r.index_type == "IndexFlatIP" for r in baseline.faiss_builds)


@pytest.mark.archive
def test_latency_records(baseline) -> None:
    by_metric = {r.metric: r for r in baseline.latency}
    assert set(by_metric) == {"Retrieval Time", "Generation Time", "Total Response Time"}
    assert by_metric["Generation Time"].mean_seconds == pytest.approx(10.69818)
