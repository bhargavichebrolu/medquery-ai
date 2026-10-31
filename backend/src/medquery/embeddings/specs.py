"""Embedding model specifications and query/document compatibility rules.

A spec fully describes how text becomes a vector: model, weights revision, dimension,
normalisation and the prefixes applied to queries and passages. Two embedding sets are
compatible (searchable against each other) only if those properties match. Batch size
does not change the vectors, so it is configuration rather than identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfigError(ValueError):
    """Raised when a model or spec is misconfigured."""


class EmbeddingCompatibilityError(ValueError):
    """Raised when query embeddings cannot be compared with indexed document embeddings."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EmbeddingFingerprint(_Frozen):
    """The properties that determine whether two embedding sets share a vector space."""

    model_id: str
    revision: str | None
    dimension: int
    normalize: bool
    query_prefix: str
    passage_prefix: str


class EmbeddingModelSpec(_Frozen):
    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1, description="Hugging Face model identifier.")
    revision: str | None = Field(
        default=None,
        description="Pinned weights revision (commit hash). None = unpinned/unknown.",
    )
    dimension: int = Field(gt=0)
    normalize: bool = True
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = Field(default=32, gt=0)

    def fingerprint(self) -> EmbeddingFingerprint:
        return EmbeddingFingerprint(
            model_id=self.model_id,
            revision=self.revision,
            dimension=self.dimension,
            normalize=self.normalize,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )


# Settings used by notebooks 04-06 (dissertation baseline). Never modify; derive
# experimental specs with `model_copy(update=...)` instead. The historical weights
# revisions were not recorded, so `revision` is None (unknown), not reconstructed.
BASELINE_EMBEDDING_SPECS: Mapping[str, EmbeddingModelSpec] = MappingProxyType(
    {
        "MiniLM": EmbeddingModelSpec(
            name="MiniLM",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision=None,
            dimension=384,
            normalize=True,
            query_prefix="",
            passage_prefix="",
            batch_size=32,
        ),
        "MPNet": EmbeddingModelSpec(
            name="MPNet",
            model_id="sentence-transformers/all-mpnet-base-v2",
            revision=None,
            dimension=768,
            normalize=True,
            query_prefix="",
            passage_prefix="",
            batch_size=16,
        ),
        "E5": EmbeddingModelSpec(
            name="E5",
            model_id="intfloat/multilingual-e5-base",
            revision=None,
            dimension=768,
            normalize=True,
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=16,
        ),
        "BGE": EmbeddingModelSpec(
            name="BGE",
            model_id="BAAI/bge-small-en-v1.5",
            revision=None,
            dimension=384,
            normalize=True,
            query_prefix="",
            passage_prefix="",
            batch_size=32,
        ),
    }
)


class CompatibilityReport(_Frozen):
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def check_compatibility(
    indexed: EmbeddingFingerprint, query: EmbeddingFingerprint
) -> CompatibilityReport:
    """Compare the fingerprint of indexed documents with that of the query embedder."""
    errors = [
        f"{field} differs: indexed={getattr(indexed, field)!r}, query={getattr(query, field)!r}"
        for field in ("model_id", "dimension", "normalize", "query_prefix", "passage_prefix")
        if getattr(indexed, field) != getattr(query, field)
    ]
    warnings: list[str] = []
    if indexed.revision and query.revision and indexed.revision != query.revision:
        errors.append(f"revision differs: indexed={indexed.revision!r}, query={query.revision!r}")
    elif indexed.revision is None or query.revision is None:
        warnings.append("revision unknown on at least one side; identical weights not verifiable")
    return CompatibilityReport(errors=tuple(errors), warnings=tuple(warnings))


def ensure_compatible(
    indexed: EmbeddingFingerprint, query: EmbeddingFingerprint
) -> CompatibilityReport:
    report = check_compatibility(indexed, query)
    if not report.ok:
        raise EmbeddingCompatibilityError("; ".join(report.errors))
    return report
