"""The research knowledge base: chunks plus the four FAISS indexes, with exact vector search."""

from medquery_api.knowledge_base.flat_index import IndexFormatError, read_flat_index
from medquery_api.knowledge_base.store import (
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    SearchHit,
    load_knowledge_base,
)

__all__ = [
    "IndexFormatError",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "KnowledgeBaseNotFoundError",
    "SearchHit",
    "load_knowledge_base",
    "read_flat_index",
]
