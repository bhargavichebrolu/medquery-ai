"""Retrieval: a question in, the most similar research chunks out.

The question is embedded with the chosen model (PRD 2.2, including E5's "query: " prefix) and
searched exactly against that model's research index (PRD 2.3), as notebook 06's
retrieve_documents did.
"""

import time
from dataclasses import dataclass

from medquery_api.embeddings.models import EmbeddingModel, get_model
from medquery_api.embeddings.query import QueryEmbedder
from medquery_api.knowledge_base import KnowledgeBase, SearchHit


@dataclass(frozen=True)
class Retrieval:
    question: str
    model: EmbeddingModel
    top_k: int
    hits: list[SearchHit]
    model_load_ms: float  # 0 when the model was already loaded
    embedding_ms: float
    search_ms: float


def retrieve(
    question: str,
    model: str,
    top_k: int,
    *,
    knowledge_base: KnowledgeBase,
    embedder: QueryEmbedder,
) -> Retrieval:
    spec = get_model(model)
    query = embedder.embed_query(spec.name, question)
    started = time.perf_counter()
    hits = knowledge_base.search(spec.name, query.vector, top_k)
    search_ms = (time.perf_counter() - started) * 1000
    return Retrieval(
        question=question,
        model=spec,
        top_k=top_k,
        hits=hits,
        model_load_ms=query.model_load_ms,
        embedding_ms=query.embedding_ms,
        search_ms=search_ms,
    )
