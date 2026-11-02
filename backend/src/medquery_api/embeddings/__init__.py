"""Embedding models from the research pipeline (notebook 04).

Importing this package does not import PyTorch or sentence-transformers; they are loaded only
when a model is opened.
"""

from medquery_api.embeddings.encoder import (
    Embedder,
    InvalidEmbeddingsError,
    ModelNotCachedError,
    open_model,
)
from medquery_api.embeddings.models import MODELS, EmbeddingModel, get_model

__all__ = [
    "MODELS",
    "EmbeddingModel",
    "Embedder",
    "InvalidEmbeddingsError",
    "ModelNotCachedError",
    "get_model",
    "open_model",
]
