"""Vector stores: backend-independent interface and the exact FAISS implementation."""

from medquery.vectorstore.archive import BASELINE_KNOWLEDGE_BASE_ID, load_archived_store
from medquery.vectorstore.base import (
    ReadOnlyStoreError,
    SearchResult,
    StoreManifest,
    VectorStore,
    VectorStoreError,
)
from medquery.vectorstore.faiss_store import FaissVectorStore

__all__ = [
    "BASELINE_KNOWLEDGE_BASE_ID",
    "FaissVectorStore",
    "ReadOnlyStoreError",
    "SearchResult",
    "StoreManifest",
    "VectorStore",
    "VectorStoreError",
    "load_archived_store",
]
