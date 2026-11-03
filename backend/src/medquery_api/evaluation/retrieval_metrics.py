"""Retrieval metrics (PRD 4.1): how well a ranking finds each question's relevant articles.

The four formulas are notebook 06's (cell 30), copied unchanged so the application scores a
ranking exactly as the research did; a test compares them with the notebook. Relevance is by
article: a retrieved chunk is relevant when its article (source file) is one of the question's
relevant articles, and an article counts once however many of its chunks are retrieved.

- **Precision@K:** relevant articles in the top K, divided by K. It divides by K even when
  fewer results were retrieved, and with one relevant article per question it can never
  exceed 1/K (0.20 at K = 5).
- **Recall@K:** relevant articles found in the top K, divided by the number of relevant
  articles.
- **MRR@K:** 1 / the rank of the first relevant result in the top K, or 0 if there is none;
  its mean over questions is the mean reciprocal rank. (The notebook's function reads the
  whole list it is given; the research gave it the top 5, so its "MRR" is MRR@5.)
- **nDCG@K:** each relevant article's first appearance in the top K counts 1 / log2(rank + 1),
  divided by the best possible total.

**Exact ties.** Chunks with identical text have identical vectors, so their scores are exactly
equal. The backend lists such chunks lower row first (PRD 2.3); the research's FAISS search
did not follow that order (where its records show an exact tie, the higher row came first),
and FAISS defines no order for ties. So when exactly tied results reach the top K, the
research may have scored the same retrieval differently. The reported scores are the
backend's own order; each question also lists every distinct set of scores obtainable by
reordering only exactly tied results, and an archived result counts as reproduced when the
backend's order, or such a reordering, gives it (PRD 2.4's rule).

**Score distributions** report each question's top similarity score, for the in-scope and
out-of-scope questions separately. Scores are comparable only within one model. They describe
these questions only: no cut-off is set or implied (decided for Part 4).

    uv run python -m medquery_api.evaluation.retrieval_metrics [--model MPNet] [--k 5]
        [--out results.json]
"""

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from medquery_api import config
from medquery_api.embeddings.models import MODELS
from medquery_api.embeddings.query import QueryEmbedder
from medquery_api.evaluation.dataset import EvaluationDataset, load_dataset
from medquery_api.knowledge_base import KnowledgeBase, SearchHit
from medquery_api.retrieval import retrieve

ARCHIVED_RESULTS = (
    config.REPO_ROOT / "Healthcare_RAG" / "results" / "corrected_retrieval_results.csv"
)
ARCHIVED_K = 5  # the research evaluated the top 5 only
TOLERANCE = 1e-9  # archived values were written as full-precision floats
LOOKAHEAD = 5  # extra results retrieved to see whether an exact tie continues past the top K
MAX_ORDERINGS = 10_000


# --- Notebook 06, cell 30: copied unchanged (tests/test_retrieval_metrics.py checks this) ---


def precision_at_k_source_level(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]
    unique_retrieved = list(dict.fromkeys(retrieved))
    relevant_retrieved = sum(1 for source in unique_retrieved if source in relevant_sources)
    return relevant_retrieved / k


def recall_at_k_source_level(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]
    unique_retrieved = set(retrieved)
    relevant_retrieved = len(unique_retrieved.intersection(relevant_sources))
    return relevant_retrieved / len(relevant_sources)


def reciprocal_rank_source_level(retrieved_sources, relevant_sources):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def ndcg_at_k_source_level(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]
    seen_sources = set()
    relevance_scores = []
    for source in retrieved:
        if source in relevant_sources and source not in seen_sources:
            relevance_scores.append(1)
            seen_sources.add(source)
        else:
            relevance_scores.append(0)
    dcg = 0.0
    for rank, relevance in enumerate(relevance_scores, start=1):
        dcg += relevance / math.log2(rank + 1)
    ideal_relevant = min(len(relevant_sources), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


NOTEBOOK_FUNCTIONS = (
    precision_at_k_source_level,
    recall_at_k_source_level,
    reciprocal_rank_source_level,
    ndcg_at_k_source_level,
)


# --- Scoring one ranking ---


@dataclass(frozen=True)
class RetrievalScores:
    precision: float  # Precision@K
    recall: float  # Recall@K
    reciprocal_rank: float  # its mean over questions is MRR@K
    ndcg: float  # nDCG@K

    def matches(self, other: "RetrievalScores") -> bool:
        return all(abs(getattr(self, name) - getattr(other, name)) <= TOLERANCE for name in METRICS)


METRICS = ("precision", "recall", "reciprocal_rank", "ndcg")


def score_ranking(sources: Sequence[str], relevant: Collection[str], k: int) -> RetrievalScores:
    """The four metrics at K for a ranked list of source files (best first)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a whole number of 1 or more, got {k!r}")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("a question needs at least one relevant article to be scored")
    ranked = list(sources)
    return RetrievalScores(
        precision=precision_at_k_source_level(ranked, relevant, k),
        recall=recall_at_k_source_level(ranked, relevant, k),
        reciprocal_rank=reciprocal_rank_source_level(ranked[:k], relevant),
        ndcg=ndcg_at_k_source_level(ranked, relevant, k),
    )


def mean_scores(scores: Sequence[RetrievalScores]) -> RetrievalScores:
    return RetrievalScores(
        **{name: statistics.fmean(getattr(s, name) for s in scores) for name in METRICS}
    )


# --- Exact ties ---


class TiesNeedMoreResultsError(ValueError):
    """The results reaching the top K use up every retrieved result, so the last of them may
    be exactly tied with results not yet retrieved; retrieve more and try again."""


def _runs_reaching_top_k(
    hits: Sequence[SearchHit], k: int, *, everything: bool = False
) -> list[list[SearchHit]]:
    """The runs of exactly equal scores (a run may be a single result) that start within the
    top K. `everything` says `hits` is the whole knowledge base, so nothing lies beyond it."""
    runs: list[list[SearchHit]] = []
    for hit in hits:
        if runs and runs[-1][-1].score == hit.score:
            runs[-1].append(hit)
        else:
            runs.append([hit])
    reaching, position = [], 0
    for run in runs:
        if position >= k:
            break
        reaching.append(run)
        position += len(run)
    if position >= len(hits) and not everything:
        raise TiesNeedMoreResultsError(
            f"the top {k} reach result {len(hits)}, the last retrieved, which may be exactly "
            "tied with the next"
        )
    return reaching


def exact_tie_groups(
    hits: Sequence[SearchHit], k: int, *, everything: bool = False
) -> list[list[SearchHit]]:
    """The runs of two or more results with exactly equal scores that reach into the top K."""
    return [run for run in _runs_reaching_top_k(hits, k, everything=everything) if len(run) > 1]


def tie_orderings(
    hits: Sequence[SearchHit], k: int, *, everything: bool = False
) -> list[tuple[str, ...]]:
    """Every top-K list of source files obtainable by reordering only exactly tied results.
    Without exact ties there is one: the backend's own order (always listed first)."""
    reaching = _runs_reaching_top_k(hits, k, everything=everything)
    count = math.prod(math.factorial(len(run)) for run in reaching)
    if count > MAX_ORDERINGS:
        raise ValueError(f"{count} orders within exact ties is too many to score")
    own = tuple(hit.chunk.source_file for hit in hits[:k])
    others = {
        tuple(hit.chunk.source_file for run in choice for hit in run)[:k]
        for choice in itertools.product(*(itertools.permutations(run) for run in reaching))
    }
    return [own, *sorted(others - {own})]


# --- Evaluating a dataset ---


@dataclass(frozen=True)
class RankedChunk:
    rank: int
    source_file: str
    chunk_id: int
    row: int  # position in the index; exactly tied chunks are listed lower row first
    score: float  # cosine similarity with the question


@dataclass(frozen=True)
class InScopeResult:
    question_id: str
    archive_question_id: int | None
    question: str
    relevant_articles: tuple[str, ...]
    results: tuple[RankedChunk, ...]  # the top K, in the backend's order
    scores: RetrievalScores  # for the backend's order
    exact_ties: tuple[tuple[RankedChunk, ...], ...]  # tied runs reaching the top K, if any
    # Every distinct set of scores obtainable by reordering only exactly tied results; just
    # `scores` when nothing is tied.
    scores_within_ties: tuple[RetrievalScores, ...]

    @property
    def top_score(self) -> float:
        return self.results[0].score


@dataclass(frozen=True)
class OutOfScopeResult:
    question_id: str
    question: str
    results: tuple[RankedChunk, ...]  # the top K; out-of-scope questions have no relevance

    @property
    def top_score(self) -> float:
        return self.results[0].score


@dataclass(frozen=True)
class ScoreSummary:
    """The top similarity score of each question in one group, for one model."""

    count: int
    minimum: float
    median: float
    maximum: float
    by_question: tuple[tuple[str, float], ...]  # (question id, top score), in dataset order


def summarise_top_scores(results: Sequence[InScopeResult | OutOfScopeResult]) -> ScoreSummary:
    if not results:
        raise ValueError("no questions to summarise")
    scores = [result.top_score for result in results]
    return ScoreSummary(
        count=len(scores),
        minimum=min(scores),
        median=statistics.median(scores),
        maximum=max(scores),
        by_question=tuple((result.question_id, result.top_score) for result in results),
    )


@dataclass(frozen=True)
class RetrievalEvaluation:
    dataset_name: str
    dataset_identifier: str
    model: str
    model_id: str
    revision: str
    k: int
    in_scope: tuple[InScopeResult, ...]
    out_of_scope: tuple[OutOfScopeResult, ...]
    mean: RetrievalScores  # over the in-scope questions, backend's order
    mean_lowest: RetrievalScores  # each metric's lowest mean over orders within exact ties
    mean_highest: RetrievalScores  # and its highest
    in_scope_top_scores: ScoreSummary
    out_of_scope_top_scores: ScoreSummary | None  # None when the dataset has none


def _ranked(hits: Sequence[SearchHit]) -> tuple[RankedChunk, ...]:
    return tuple(
        RankedChunk(h.rank, h.chunk.source_file, h.chunk.chunk_id, h.row, h.score) for h in hits
    )


def _retrieve_enough(
    question: str, model: str, k: int, knowledge_base: KnowledgeBase, embedder: QueryEmbedder
) -> tuple[list[SearchHit], bool]:
    """The top K plus enough beyond it to see where any exact tie ends, and whether that is
    the whole knowledge base."""
    available = len(knowledge_base.chunks)
    count = min(k + LOOKAHEAD, available)
    while True:
        retrieval = retrieve(
            question, model, count, knowledge_base=knowledge_base, embedder=embedder
        )
        everything = count == available
        try:
            _runs_reaching_top_k(retrieval.hits, k, everything=everything)
            return retrieval.hits, everything
        except TiesNeedMoreResultsError:
            count = min(count * 2, available)


def evaluate_retrieval(
    dataset: EvaluationDataset,
    model: str,
    k: int,
    *,
    knowledge_base: KnowledgeBase,
    embedder: QueryEmbedder,
) -> RetrievalEvaluation:
    """Retrieve every dataset question with one model and score the in-scope ones at K."""
    if not dataset.in_scope_questions:
        raise ValueError("the dataset has no in-scope questions to score")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(knowledge_base.chunks):
        raise ValueError(f"k must be a whole number from 1 to {len(knowledge_base.chunks)}")

    spec = MODELS.get(model)
    if spec is None:
        raise ValueError(f"unknown embedding model {model!r} (choose from {', '.join(MODELS)})")
    in_scope = []
    for question in dataset.in_scope_questions:
        hits, everything = _retrieve_enough(question.question, model, k, knowledge_base, embedder)
        relevant = question.relevant_articles
        own = score_ranking([h.chunk.source_file for h in hits], relevant, k)
        alternatives: list[RetrievalScores] = []
        for ordering in tie_orderings(hits, k, everything=everything):
            scores = score_ranking(ordering, relevant, k)
            if not any(scores.matches(seen) for seen in alternatives):
                alternatives.append(scores)
        in_scope.append(
            InScopeResult(
                question_id=question.id,
                archive_question_id=question.archive_question_id,
                question=question.question,
                relevant_articles=question.relevant_articles,
                results=_ranked(hits[:k]),
                scores=own,
                exact_ties=tuple(
                    _ranked(run) for run in exact_tie_groups(hits, k, everything=everything)
                ),
                scores_within_ties=tuple(alternatives),
            )
        )

    out_of_scope = []
    for question in dataset.out_of_scope_questions:
        retrieval = retrieve(
            question.question, model, k, knowledge_base=knowledge_base, embedder=embedder
        )
        out_of_scope.append(
            OutOfScopeResult(question.id, question.question, _ranked(retrieval.hits))
        )

    def extreme(choose) -> RetrievalScores:
        return RetrievalScores(
            **{
                name: statistics.fmean(
                    choose(getattr(s, name) for s in r.scores_within_ties) for r in in_scope
                )
                for name in METRICS
            }
        )

    return RetrievalEvaluation(
        dataset_name=dataset.name,
        dataset_identifier=dataset.identifier,
        model=spec.name,
        model_id=spec.model_id,
        revision=spec.revision,
        k=k,
        in_scope=tuple(in_scope),
        out_of_scope=tuple(out_of_scope),
        mean=mean_scores([r.scores for r in in_scope]),
        mean_lowest=extreme(min),
        mean_highest=extreme(max),
        in_scope_top_scores=summarise_top_scores(in_scope),
        out_of_scope_top_scores=summarise_top_scores(out_of_scope) if out_of_scope else None,
    )


# --- Comparing with the research's archived results (historical) ---

Reproduction = Literal["same", "within_exact_ties", "differs"]


@dataclass(frozen=True)
class ArchiveComparison:
    question_id: str
    archive_question_id: int
    archived: RetrievalScores  # historical: the research's recorded values
    ours: RetrievalScores  # the backend's order
    status: Reproduction


def load_archived_results(path: Path = ARCHIVED_RESULTS) -> dict[tuple[str, int], RetrievalScores]:
    """The research's per-question results at K = 5, by (model, question number). Read-only."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            (row["model"], int(row["question_id"])): RetrievalScores(
                precision=float(row["precision_at_5"]),
                recall=float(row["recall_at_5"]),
                reciprocal_rank=float(row["mrr"]),
                ndcg=float(row["ndcg_at_5"]),
            )
            for row in csv.DictReader(file)
        }


def compare_with_archive(
    evaluation: RetrievalEvaluation, archived: dict[tuple[str, int], RetrievalScores]
) -> list[ArchiveComparison]:
    """Each research question's scores against the archived ones (only at K = 5, the research's
    K). `same`: the backend's order gives them; `within_exact_ties`: only a reordering of
    exactly tied results does; `differs`: neither."""
    if evaluation.k != ARCHIVED_K:
        return []
    comparisons = []
    for result in evaluation.in_scope:
        key = (evaluation.model, result.archive_question_id)
        if result.archive_question_id is None or key not in archived:
            continue
        recorded = archived[key]
        if result.scores.matches(recorded):
            status: Reproduction = "same"
        elif any(s.matches(recorded) for s in result.scores_within_ties):
            status = "within_exact_ties"
        else:
            status = "differs"
        comparisons.append(
            ArchiveComparison(
                result.question_id, result.archive_question_id, recorded, result.scores, status
            )
        )
    return comparisons


# --- Report ---


def _metrics_line(scores: RetrievalScores, k: int) -> str:
    return (
        f"Precision@{k} {scores.precision:.3f}  Recall@{k} {scores.recall:.3f}  "
        f"MRR@{k} {scores.reciprocal_rank:.3f}  nDCG@{k} {scores.ndcg:.3f}"
    )


def _summary_line(summary: ScoreSummary) -> str:
    return (
        f"{summary.count} questions: lowest {summary.minimum:.4f}, median {summary.median:.4f}, "
        f"highest {summary.maximum:.4f}"
    )


def format_report(evaluation: RetrievalEvaluation, comparisons: list[ArchiveComparison]) -> str:
    k = evaluation.k
    lines = [
        f"{evaluation.model} ({evaluation.model_id} at {evaluation.revision[:7]}), K = {k}",
        f"  dataset: {evaluation.dataset_name} ({evaluation.dataset_identifier[:19]}...)",
        f"  mean over {len(evaluation.in_scope)} in-scope questions: "
        + _metrics_line(evaluation.mean, k),
    ]
    if evaluation.mean_lowest != evaluation.mean_highest:
        ranges = "  ".join(
            f"{label} {getattr(evaluation.mean_lowest, name):.3f} to "
            f"{getattr(evaluation.mean_highest, name):.3f}"
            for name, label in zip(METRICS, ("Precision", "Recall", "MRR", "nDCG"), strict=True)
            if getattr(evaluation.mean_lowest, name) != getattr(evaluation.mean_highest, name)
        )
        lines.append(f"    with exactly tied chunks in either order: {ranges}")
    for result in evaluation.in_scope:
        first = next(
            (r.rank for r in result.results if r.source_file in result.relevant_articles), None
        )
        note = f"first relevant at rank {first}" if first else f"no relevant article in top {k}"
        for run in result.exact_ties:
            note += f"; exact tie at ranks {run[0].rank} to {run[-1].rank}"
        lines.append(f"    {result.question_id:<12} {_metrics_line(result.scores, k)}  ({note})")
    if comparisons:
        same = [c for c in comparisons if c.status == "same"]
        tied = [c.question_id for c in comparisons if c.status == "within_exact_ties"]
        differs = [c.question_id for c in comparisons if c.status == "differs"]
        line = (
            f"  archived research results (historical): {len(same) + len(tied)} of "
            f"{len(comparisons)} reproduced"
        )
        if tied:
            line += f"; {', '.join(tied)} only with exactly tied chunks reordered"
        if differs:
            line += f"; NOT reproduced: {', '.join(differs)}"
        lines.append(line)
    lines.append("  top similarity score per question (comparable within this model only):")
    lines.append(f"    in scope:     {_summary_line(evaluation.in_scope_top_scores)}")
    if evaluation.out_of_scope_top_scores:
        lines.append(f"    out of scope: {_summary_line(evaluation.out_of_scope_top_scores)}")
        for question_id, score in evaluation.out_of_scope_top_scores.by_question:
            lines.append(f"      {question_id:<16} {score:.4f}")
    lines.append(
        f"  These describe {len(evaluation.in_scope) + len(evaluation.out_of_scope)} questions "
        "only; no cut-off is set or implied."
    )
    return "\n".join(lines)


def _refuse_archive_path(path: Path) -> Path:
    resolved = path.resolve()
    archive = (config.REPO_ROOT / "Healthcare_RAG").resolve()
    if resolved == archive or archive in resolved.parents:
        raise SystemExit(f"Refusing to write inside the research archive: {path}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m medquery_api.evaluation.retrieval_metrics",
        description="Score retrieval on the evaluation dataset (local models only; no API).",
    )
    parser.add_argument(
        "--model", action="append", choices=list(MODELS), help="repeat for several (default: all)"
    )
    parser.add_argument("--k", type=int, default=ARCHIVED_K)
    parser.add_argument("--dataset", type=Path, help="a dataset file (default: the bundled one)")
    parser.add_argument("--out", type=Path, help="write the results as JSON (not in the archive)")
    args = parser.parse_args(argv)
    out = _refuse_archive_path(args.out) if args.out else None

    from medquery_api.knowledge_base import load_knowledge_base

    dataset = load_dataset(args.dataset) if args.dataset else load_dataset()
    knowledge_base = load_knowledge_base()
    embedder = QueryEmbedder()
    archived = load_archived_results()
    results = []
    try:
        for model in args.model or list(MODELS):
            evaluation = evaluate_retrieval(
                dataset, model, args.k, knowledge_base=knowledge_base, embedder=embedder
            )
            comparisons = compare_with_archive(evaluation, archived)
            print(format_report(evaluation, comparisons), end="\n\n")
            results.append(
                {"evaluation": asdict(evaluation), "archive": [asdict(c) for c in comparisons]}
            )
    finally:
        embedder.close()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
