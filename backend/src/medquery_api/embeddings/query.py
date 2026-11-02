"""Embed questions for the API, keeping at most one model loaded between requests.

Loading a model takes seconds, so the API keeps the last one used and swaps it only when a
request asks for a different model. One lock covers choosing, loading and using the model, so
a request can never have its model swapped out while it is being used. Requests therefore
embed one at a time.
"""

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from medquery_api.embeddings.encoder import (
    Embedder,
    ModelNotCachedError,
    default_device,
    load_sentence_transformer,
)
from medquery_api.embeddings.models import get_model


class QueryModelUnavailableError(RuntimeError):
    """The requested model could not be loaded (not in the local cache, or no such device)."""


@dataclass(frozen=True)
class QueryEmbedding:
    model: str
    vector: np.ndarray
    model_load_ms: float  # time spent loading or switching the model; 0 if already loaded
    embedding_ms: float  # time spent turning the question into a vector


class QueryEmbedder:
    def __init__(
        self,
        device: str | None = None,
        loader: Callable[..., Any] = load_sentence_transformer,
    ):
        self._device = device
        self._loader = loader
        self._lock = threading.Lock()
        self._embedder: Embedder | None = None

    @property
    def loaded_model(self) -> str | None:
        embedder = self._embedder
        return embedder.spec.name if embedder else None

    def embed_query(self, model: str, question: str) -> QueryEmbedding:
        """The question's vector with this model (adding the model's query prefix)."""
        spec = get_model(model)
        with self._lock:
            model_load_ms = 0.0
            if self._embedder is None or self._embedder.spec.name != spec.name:
                started = time.perf_counter()
                self._release()  # the previous model goes before the next one is loaded
                try:
                    loaded = self._loader(spec, self._device or default_device())
                except ModelNotCachedError as error:
                    raise QueryModelUnavailableError(str(error)) from error
                except ValueError as error:  # e.g. a GPU device that this PyTorch cannot use
                    raise QueryModelUnavailableError(str(error)) from error
                self._embedder = Embedder(spec, loaded)
                model_load_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            vector = self._embedder.embed_query(question)
            embedding_ms = (time.perf_counter() - started) * 1000
        return QueryEmbedding(spec.name, vector, model_load_ms, embedding_ms)

    def close(self) -> None:
        """Release the loaded model, if any."""
        with self._lock:
            self._release()

    def _release(self) -> None:
        if self._embedder is not None:
            self._embedder.model = None
            self._embedder = None
            gc.collect()
