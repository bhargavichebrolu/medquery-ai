"""Deterministic, dependency-free embedder for tests and offline development.

Vectors are derived from a SHA-256 hash of the (prefixed) text, so identical inputs
give identical vectors and the prefix rules are observable. Tokens are whitespace
words plus two special tokens. The vectors carry no semantic meaning.
"""

from __future__ import annotations

import hashlib

import numpy as np

from medquery.embeddings.base import Embedder, EmbedderInfo
from medquery.embeddings.specs import EmbeddingModelSpec


class FakeEmbedder(Embedder):
    def __init__(self, spec: EmbeddingModelSpec, max_seq_length: int | None = 128) -> None:
        super().__init__(spec)
        self._max_seq_length = max_seq_length
        self.encode_calls: list[tuple[int, int]] = []  # (num_texts, batch_size)

    @property
    def max_seq_length(self) -> int | None:
        return self._max_seq_length

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.encode_calls.append((len(texts), batch_size))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
            vector = np.random.default_rng(seed).standard_normal(self.spec.dimension)
            if self.spec.normalize:
                vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows).astype(np.float32)

    def _token_counts(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 2 for text in texts]

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name=self.spec.name,
            model_id=self.spec.model_id,
            requested_revision=self.spec.revision,
            resolved_revision=None,
            dimension=self.spec.dimension,
            max_seq_length=self._max_seq_length,
            normalize=self.spec.normalize,
            query_prefix=self.spec.query_prefix,
            passage_prefix=self.spec.passage_prefix,
            backend="fake",
            device="cpu",
        )
