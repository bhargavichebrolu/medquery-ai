"""Token-length statistics and truncation analysis for embedding inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict


class TruncatedInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    tokens: int
    tokens_lost: int


class TokenLengthStats(BaseModel):
    """Summary of untruncated token counts against a model's maximum input length."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int
    max_seq_length: int | None
    min: int
    mean: float
    median: float
    p95: float
    max: int
    truncated_count: int
    truncated_fraction: float
    tokens_lost_total: int
    worst: tuple[TruncatedInput, ...]


def summarise_token_lengths(
    item_ids: Sequence[str],
    token_counts: Sequence[int],
    max_seq_length: int | None,
    top_n: int = 10,
) -> TokenLengthStats:
    if len(item_ids) != len(token_counts):
        raise ValueError("item_ids and token_counts must have the same length")
    if not token_counts:
        raise ValueError("token_counts is empty")
    counts = np.asarray(token_counts, dtype=np.int64)
    limit = max_seq_length if max_seq_length is not None else np.iinfo(np.int64).max
    overflow = np.clip(counts - limit, 0, None)
    order = np.argsort(-counts, kind="stable")
    worst = tuple(
        TruncatedInput(item_id=item_ids[i], tokens=int(counts[i]), tokens_lost=int(overflow[i]))
        for i in order[:top_n]
        if overflow[i] > 0
    )
    truncated = int((overflow > 0).sum())
    return TokenLengthStats(
        count=len(counts),
        max_seq_length=max_seq_length,
        min=int(counts.min()),
        mean=round(float(counts.mean()), 2),
        median=float(np.median(counts)),
        p95=round(float(np.percentile(counts, 95)), 2),
        max=int(counts.max()),
        truncated_count=truncated,
        truncated_fraction=round(truncated / len(counts), 4),
        tokens_lost_total=int(overflow.sum()),
        worst=worst,
    )
