"""Backend-independent vector store interface.

A store holds the chunks of one knowledge base together with their document
embeddings, all produced by a single embedding configuration (its fingerprint). Queries
must come from a compatible embedder; this is checked on every search, so a store can
never be searched with vectors from a different vector space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from medquery.embeddings.specs import EmbeddingFingerprint
from medquery.ingestion.models import Chunk

Provenance = Literal["new_run", "original_dissertation"]

# Tolerance for the unit-norm check on normalised float32 vectors (same as embedders).
NORM_TOLERANCE = 1e-3
KNOWLEDGE_BASE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"


class VectorStoreError(RuntimeError):
    """Raised for invalid store operations or corrupted store files."""


class ReadOnlyStoreError(VectorStoreError):
    """Raised when modifying a read-only store (for example the research archive)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchResult(_Frozen):
    rank: int = Field(ge=1)
    score: float
    chunk: Chunk


class StoreManifest(_Frozen):
    """Describes a saved store: what it contains and how it was built."""

    format_version: Literal[1] = 1
    store_type: str
    knowledge_base_id: str = Field(pattern=KNOWLEDGE_BASE_ID_PATTERN)
    fingerprint: EmbeddingFingerprint
    vector_count: int = Field(ge=0)
    provenance: Provenance
    created_at: datetime
    build_config: dict[str, Any] = Field(default_factory=dict)
    files: dict[str, str] = Field(
        default_factory=dict, description="File name -> SHA-256 of the stored files."
    )


class VectorStore(ABC):
    """Stores chunk vectors and searches them by inner product (cosine when normalised)."""

    @property
    @abstractmethod
    def knowledge_base_id(self) -> str: ...

    @property
    @abstractmethod
    def fingerprint(self) -> EmbeddingFingerprint: ...

    @property
    @abstractmethod
    def read_only(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Add chunks with their document embeddings (row i belongs to chunks[i])."""

    @abstractmethod
    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        *,
        query_fingerprint: EmbeddingFingerprint,
        source_ids: Collection[str] | None = None,
    ) -> list[list[SearchResult]]:
        """Return the top-k chunks for each query vector, best first.

        `source_ids` restricts results to chunks of the given documents. Fewer than k
        results are returned when fewer chunks are available.
        """

    @abstractmethod
    def save(self, directory: Path, *, overwrite: bool = False) -> StoreManifest: ...
