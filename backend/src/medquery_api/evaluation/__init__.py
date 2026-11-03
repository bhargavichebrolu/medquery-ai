"""Evaluation (Part 4). PRD 4.0: the evaluation dataset format, its loader and its checks.

Check a dataset file with `uv run python -m medquery_api.evaluation [PATH]`.
"""

from medquery_api.evaluation.dataset import (
    DEFAULT_DATASET,
    FORMAT_VERSION,
    EvaluationDataset,
    EvaluationDatasetError,
    InScopeQuestion,
    OutOfScopeQuestion,
    ReferenceAnswer,
    article_catalogue,
    dataset_identifier,
    load_dataset,
    missing_articles,
    parse_dataset,
)

__all__ = [
    "DEFAULT_DATASET",
    "FORMAT_VERSION",
    "EvaluationDataset",
    "EvaluationDatasetError",
    "InScopeQuestion",
    "OutOfScopeQuestion",
    "ReferenceAnswer",
    "article_catalogue",
    "dataset_identifier",
    "load_dataset",
    "missing_articles",
    "parse_dataset",
]
