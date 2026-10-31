"""Backend-independent embedder interface and embedding-run records."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from medquery.embeddings.specs import EmbeddingFingerprint, EmbeddingModelSpec
from medquery.ingestion.models import Chunk

TextKind = Literal["document", "query"]

# Tolerance for the unit-norm check on normalised float32 embeddings.
_NORM_TOLERANCE = 1e-3


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbedderInfo(_Frozen):
    """What actually ran: recorded with every embedding run for reproducibility."""

    name: str
    model_id: str
    requested_revision: str | None
    resolved_revision: str | None
    dimension: int
    max_seq_length: int | None
    normalize: bool
    query_prefix: str
    passage_prefix: str
    backend: str
    device: str
    device_name: str | None = None
    library_versions: dict[str, str] = {}
    model_prompts: dict[str, str] = {}


class Embedder(ABC):
    """Turns text into vectors according to an `EmbeddingModelSpec`.

    Subclasses implement `_encode`, `_token_counts` and `info`. This base class applies
    the spec's query/passage prefixes, so the same rules are used for indexing and for
    querying, and it validates every output (shape, dtype, finiteness, unit norm).
    """

    def __init__(self, spec: EmbeddingModelSpec) -> None:
        self.spec = spec

    # --- To implement -------------------------------------------------------------

    @abstractmethod
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...

    @abstractmethod
    def _token_counts(self, texts: list[str]) -> list[int]: ...

    @property
    @abstractmethod
    def max_seq_length(self) -> int | None:
        """Maximum input length in tokens; longer inputs are truncated by the model."""

    @abstractmethod
    def info(self) -> EmbedderInfo: ...

    # --- Public API -----------------------------------------------------------------

    def fingerprint(self) -> EmbeddingFingerprint:
        return self.spec.fingerprint()

    def prepare(self, texts: Sequence[str], kind: TextKind) -> list[str]:
        """Return the exact strings the model sees (prefix applied)."""
        if isinstance(texts, str):
            raise TypeError("Expected a sequence of strings, not a single string")
        prefix = self.spec.query_prefix if kind == "query" else self.spec.passage_prefix
        prepared = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"Expected str, got {type(text).__name__}")
            prepared.append(f"{prefix}{text}")
        return prepared

    def embed_documents(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        return self._embed(self.prepare(texts, "document"), batch_size)

    def embed_queries(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        return self._embed(self.prepare(texts, "query"), batch_size)

    def count_tokens(self, texts: Sequence[str], kind: TextKind = "document") -> list[int]:
        """Untruncated token counts, including special tokens and the prefix."""
        prepared = self.prepare(texts, kind)
        return self._token_counts(prepared) if prepared else []

    # --- Internals --------------------------------------------------------------------

    def _embed(self, prepared: list[str], batch_size: int | None) -> np.ndarray:
        size = self.spec.batch_size if batch_size is None else batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")
        if not prepared:
            return np.zeros((0, self.spec.dimension), dtype=np.float32)
        vectors = np.asarray(self._encode(prepared, size), dtype=np.float32)
        self._validate(vectors, len(prepared))
        return vectors

    def _validate(self, vectors: np.ndarray, expected_rows: int) -> None:
        if vectors.shape != (expected_rows, self.spec.dimension):
            raise ValueError(
                f"Embedding shape {vectors.shape} != expected "
                f"{(expected_rows, self.spec.dimension)}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values")
        if self.spec.normalize:
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
                raise ValueError("Embeddings are not unit-normalised")


class EmbeddingRunRecord(_Frozen):
    """Metadata for a newly executed embedding run (never a historical result)."""

    provenance: Literal["new_run"] = "new_run"
    created_at: datetime
    embedder: EmbedderInfo
    fingerprint: EmbeddingFingerprint
    batch_size: int
    num_texts: int
    seconds: float


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    chunk_uids: tuple[str, ...]
    record: EmbeddingRunRecord


def embed_chunks(
    embedder: Embedder, chunks: Sequence[Chunk], batch_size: int | None = None
) -> EmbeddingResult:
    """Embed chunks as documents, timing the run and recording its configuration."""
    size = embedder.spec.batch_size if batch_size is None else batch_size
    start = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks], batch_size=size)
    seconds = time.perf_counter() - start
    record = EmbeddingRunRecord(
        created_at=datetime.now(UTC),
        embedder=embedder.info(),
        fingerprint=embedder.fingerprint(),
        batch_size=size,
        num_texts=len(chunks),
        seconds=seconds,
    )
    return EmbeddingResult(
        vectors=vectors, chunk_uids=tuple(c.chunk_uid for c in chunks), record=record
    )
