"""Turn text into vectors with one of the research's embedding models.

Models are only ever loaded from the local Hugging Face cache; they are downloaded only by the
explicit `download` command (see __main__.py). They run on the CPU unless
MEDQUERY_EMBEDDING_DEVICE says otherwise. The heavy libraries (sentence-transformers, PyTorch)
are imported only when a model is opened.
"""

import gc
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from medquery_api.embeddings.models import EmbeddingModel, get_model

DEVICE_ENV = "MEDQUERY_EMBEDDING_DEVICE"

# Vectors must have length 1 within this tolerance (float32 rounding; see
# docs/research-pipeline.md, stage 4).
NORM_TOLERANCE = 1e-4

# At most one model is held in memory at a time; opening a second one first is an error.
_one_model_at_a_time = threading.Lock()


class ModelNotCachedError(RuntimeError):
    """The model (at its pinned revision) is not in the local Hugging Face cache."""


class InvalidEmbeddingsError(ValueError):
    """A model returned vectors of the wrong shape, non-finite values, or not of length 1."""


def default_device() -> str:
    return os.environ.get(DEVICE_ENV, "cpu")


def load_sentence_transformer(
    spec: EmbeddingModel,
    device: str,
    *,
    local_files_only: bool = True,
    factory: Callable[..., Any] | None = None,
) -> Any:
    """Load a model at its pinned revision. By default only from the local cache."""
    if factory is None:
        from sentence_transformers import SentenceTransformer  # heavy: imported only here

        factory = SentenceTransformer
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise ValueError(
                f"{DEVICE_ENV}={device} but this PyTorch build cannot use a GPU; use 'cpu'"
            )
    try:
        return factory(
            spec.model_id,
            revision=spec.revision,
            device=device,
            local_files_only=local_files_only,
        )
    except OSError as error:
        if not local_files_only:
            raise
        raise ModelNotCachedError(
            f"{spec.name} ({spec.model_id} at revision {spec.revision[:12]}) is not in the local "
            f"model cache. Download it with: uv run python -m medquery_api.embeddings download "
            f"{spec.name}"
        ) from error


class Embedder:
    """An open model: embeds documents and queries with that model's prefixes."""

    def __init__(self, spec: EmbeddingModel, model: Any):
        self.spec = spec
        self.model = model

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        """One vector per text, in input order: an array of shape (len(texts), dimension)."""
        return self._encode([self.spec.document_prefix + text for text in texts], show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        """The vector for one question: an array of shape (dimension,)."""
        return self._encode([self.spec.query_prefix + text], False)[0]

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.spec.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        check_vectors(vectors, len(texts), self.spec.dimension)
        return vectors


def check_vectors(vectors: np.ndarray, count: int, dimension: int) -> None:
    """Raise InvalidEmbeddingsError unless these are `count` finite unit vectors of `dimension`."""
    if vectors.shape != (count, dimension):
        raise InvalidEmbeddingsError(
            f"expected {count} vectors of size {dimension}, got shape {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise InvalidEmbeddingsError("vectors contain NaN or infinite values")
    deviation = np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1.0).max()
    if deviation > NORM_TOLERANCE:
        raise InvalidEmbeddingsError(
            f"vector length differs from 1 by {deviation:.3e} (tolerance {NORM_TOLERANCE:g})"
        )


@contextmanager
def open_model(
    name: str,
    device: str | None = None,
    *,
    loader: Callable[..., Any] = load_sentence_transformer,
) -> Iterator[Embedder]:
    """Load a model for the duration of a `with` block, then release it.

    Only one model can be open at a time, so memory holds at most one model.
    """
    spec = get_model(name)
    if not _one_model_at_a_time.acquire(blocking=False):
        raise RuntimeError("another embedding model is already open; close it first")
    try:
        chosen_device = device or default_device()
        embedder = Embedder(spec, loader(spec, chosen_device))
        try:
            yield embedder
        finally:
            embedder.model = None
            del embedder
            gc.collect()
            if chosen_device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
    finally:
        _one_model_at_a_time.release()
