"""Typed, read-only access to the original dissertation results.

Values are parsed straight from the CSVs in `Healthcare_RAG/results/`; nothing is
copied or recomputed. Every model carries `provenance="original_dissertation"` so
downstream code can always distinguish these numbers from newly executed runs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Provenance = Literal["original_dissertation"]
ORIGINAL: Provenance = "original_dissertation"

RETRIEVAL_SUMMARY_CSV = "retrieval_model_comparison.csv"
RETRIEVAL_PER_QUESTION_CSV = "corrected_retrieval_results.csv"
FAISS_BUILD_CSV = "faiss_experimental_summary.csv"
LATENCY_CSV = "rag_performance_summary.csv"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievalMetrics(_Frozen):
    """Mean source-level retrieval metrics for one embedding model (k = 5, 10 questions)."""

    provenance: Provenance = ORIGINAL
    model: str
    precision_at_5: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)


class QuestionRetrievalResult(_Frozen):
    provenance: Provenance = ORIGINAL
    model: str
    question_id: int
    question: str
    precision_at_5: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)


class FaissBuildRecord(_Frozen):
    provenance: Provenance = ORIGINAL
    model: str
    vectors: int
    dimension: int
    index_type: str
    build_time_seconds: float
    original_index_path: str


class LatencyStat(_Frozen):
    """Llama 3.1 8B Instruct (4-bit) + MPNet, top-k 5, Colab T4 GPU, 10 questions."""

    provenance: Provenance = ORIGINAL
    metric: str
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float


class BaselineResults(_Frozen):
    retrieval_summary: list[RetrievalMetrics]
    retrieval_per_question: list[QuestionRetrievalResult]
    faiss_builds: list[FaissBuildRecord]
    latency: list[LatencyStat]

    def retrieval_for(self, model: str) -> RetrievalMetrics:
        for row in self.retrieval_summary:
            if row.model == model:
                return row
        raise KeyError(model)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_retrieval_summary(results_dir: Path) -> list[RetrievalMetrics]:
    return [
        RetrievalMetrics(
            model=row["Embedding Model"],
            precision_at_5=float(row["Precision@5"]),
            recall_at_5=float(row["Recall@5"]),
            mrr=float(row["MRR"]),
            ndcg_at_5=float(row["nDCG@5"]),
        )
        for row in _read_csv(results_dir / RETRIEVAL_SUMMARY_CSV)
    ]


def load_retrieval_per_question(results_dir: Path) -> list[QuestionRetrievalResult]:
    return [
        QuestionRetrievalResult(
            model=row["model"],
            question_id=int(row["question_id"]),
            question=row["question"],
            precision_at_5=float(row["precision_at_5"]),
            recall_at_5=float(row["recall_at_5"]),
            mrr=float(row["mrr"]),
            ndcg_at_5=float(row["ndcg_at_5"]),
        )
        for row in _read_csv(results_dir / RETRIEVAL_PER_QUESTION_CSV)
    ]


def load_faiss_builds(results_dir: Path) -> list[FaissBuildRecord]:
    return [
        FaissBuildRecord(
            model=row["Model"],
            vectors=int(row["Vectors"]),
            dimension=int(row["Dimension"]),
            index_type=row["Index_Type"],
            build_time_seconds=float(row["Build_Time_Seconds"]),
            original_index_path=row["Index_File"],
        )
        for row in _read_csv(results_dir / FAISS_BUILD_CSV)
    ]


def load_latency(results_dir: Path) -> list[LatencyStat]:
    return [
        LatencyStat(
            metric=row["Metric"],
            mean_seconds=float(row["Mean (seconds)"]),
            std_seconds=float(row["Std Dev (seconds)"]),
            min_seconds=float(row["Minimum (seconds)"]),
            max_seconds=float(row["Maximum (seconds)"]),
        )
        for row in _read_csv(results_dir / LATENCY_CSV)
    ]


def load_baseline(results_dir: Path) -> BaselineResults:
    return BaselineResults(
        retrieval_summary=load_retrieval_summary(results_dir),
        retrieval_per_question=load_retrieval_per_question(results_dir),
        faiss_builds=load_faiss_builds(results_dir),
        latency=load_latency(results_dir),
    )
