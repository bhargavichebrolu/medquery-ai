"""Embedding models: specs, compatibility rules, embedders and token analysis.

`SentenceTransformerEmbedder` is imported from its own module so that importing this
package does not pull in torch.
"""

from medquery.embeddings.base import (
    Embedder,
    EmbedderInfo,
    EmbeddingResult,
    EmbeddingRunRecord,
    embed_chunks,
)
from medquery.embeddings.fake import FakeEmbedder
from medquery.embeddings.specs import (
    BASELINE_EMBEDDING_SPECS,
    CompatibilityReport,
    EmbeddingCompatibilityError,
    EmbeddingConfigError,
    EmbeddingFingerprint,
    EmbeddingModelSpec,
    check_compatibility,
    ensure_compatible,
)
from medquery.embeddings.tokens import TokenLengthStats, summarise_token_lengths

__all__ = [
    "BASELINE_EMBEDDING_SPECS",
    "CompatibilityReport",
    "Embedder",
    "EmbedderInfo",
    "EmbeddingCompatibilityError",
    "EmbeddingConfigError",
    "EmbeddingFingerprint",
    "EmbeddingModelSpec",
    "EmbeddingResult",
    "EmbeddingRunRecord",
    "FakeEmbedder",
    "TokenLengthStats",
    "check_compatibility",
    "embed_chunks",
    "ensure_compatible",
    "summarise_token_lengths",
]
