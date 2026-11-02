"""The MedQuery AI web API."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic.alias_generators import to_camel

from medquery_api import config
from medquery_api.dataset import DataFolderNotFoundError, summarise_dataset
from medquery_api.embeddings.query import QueryEmbedder, QueryModelUnavailableError
from medquery_api.knowledge_base import (
    IndexFormatError,
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    load_knowledge_base,
)
from medquery_api.retrieval import retrieve

app = FastAPI(title="MedQuery AI API", version="0.1.0")
logger = logging.getLogger("medquery_api")

DATA_NOT_FOUND = (
    "Research data folder not found. Restore Healthcare_RAG/data or set MEDQUERY_DATA_DIR."
)


class CamelModel(BaseModel):
    # Python uses snake_case; the JSON sent to and from the frontend uses camelCase.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class DatasetSummaryResponse(CamelModel):
    articles: int
    chunks: int
    embedding_indexes: list[str]
    source: str


class RetrieveRequest(CamelModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    # Surrounding whitespace is removed first; what remains must be 1 to 1,000 characters.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    model: Literal["MiniLM", "MPNet", "E5", "BGE"]
    top_k: Annotated[int, Field(strict=True, ge=1, le=20)] = 5


class RetrievedChunk(CamelModel):
    rank: int
    score: float  # cosine similarity between the question and the chunk
    source_file: str
    chunk_id: int
    chunk_length: int
    text: str


class RetrievalTimings(CamelModel):
    model_load_ms: float  # loading or switching the model; 0 when it was already loaded
    embedding_ms: float
    search_ms: float


class RetrieveResponse(CamelModel):
    question: str
    model: str
    model_id: str
    revision: str
    top_k: int
    results: list[RetrievedChunk]
    timings: RetrievalTimings


def get_data_dir() -> Path:
    """Supplied to endpoints by FastAPI; tests replace it with a temporary folder."""
    return config.data_dir()


def get_knowledge_base_loader(
    data_dir: Annotated[Path, Depends(get_data_dir)],
) -> Callable[[], KnowledgeBase]:
    """A function that loads the knowledge base, called only once a request is valid."""
    return lambda: load_knowledge_base(data_dir)


_query_embedder = QueryEmbedder()


def get_query_embedder() -> QueryEmbedder:
    """The process's query embedder (it loads models only when used)."""
    return _query_embedder


@app.get("/api/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/dataset/summary", response_model=DatasetSummaryResponse)
def dataset_summary(data_dir: Annotated[Path, Depends(get_data_dir)]) -> DatasetSummaryResponse:
    try:
        summary = summarise_dataset(data_dir)
    except DataFolderNotFoundError as error:
        raise HTTPException(status_code=503, detail=DATA_NOT_FOUND) from error

    source = "Healthcare_RAG/data" if data_dir == config.DEFAULT_DATA_DIR else "MEDQUERY_DATA_DIR"
    return DatasetSummaryResponse(
        articles=summary.articles,
        chunks=summary.chunks,
        embedding_indexes=summary.embedding_indexes,
        source=source,
    )


@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    request: RetrieveRequest,
    load: Annotated[Callable[[], KnowledgeBase], Depends(get_knowledge_base_loader)],
    embedder: Annotated[QueryEmbedder, Depends(get_query_embedder)],
) -> RetrieveResponse:
    """The research chunks most similar to a question, using one embedding model."""
    try:
        knowledge_base = load()
    except KnowledgeBaseNotFoundError as error:
        raise HTTPException(status_code=503, detail=DATA_NOT_FOUND) from error
    except (KnowledgeBaseError, IndexFormatError, OSError) as error:
        logger.error("Research data failed its checks: %s", error)
        raise HTTPException(
            status_code=503,
            detail="The research data did not pass its integrity checks, so retrieval is "
            "unavailable. The server log says which check failed.",
        ) from error

    try:
        result = retrieve(
            request.question,
            request.model,
            request.top_k,
            knowledge_base=knowledge_base,
            embedder=embedder,
        )
    except QueryModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return RetrieveResponse(
        question=result.question,
        model=result.model.name,
        model_id=result.model.model_id,
        revision=result.model.revision,
        top_k=result.top_k,
        results=[
            RetrievedChunk(
                rank=hit.rank,
                score=hit.score,
                source_file=hit.chunk.source_file,
                chunk_id=hit.chunk.chunk_id,
                chunk_length=hit.chunk.chunk_length,
                text=hit.chunk.text,
            )
            for hit in result.hits
        ],
        timings=RetrievalTimings(
            model_load_ms=round(result.model_load_ms, 3),
            embedding_ms=round(result.embedding_ms, 3),
            search_ms=round(result.search_ms, 3),
        ),
    )
