"""The MedQuery AI web API."""

import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic.alias_generators import to_camel

from medquery_api import config
from medquery_api.answering import answer_question
from medquery_api.dataset import DataFolderNotFoundError, summarise_dataset
from medquery_api.embeddings.query import QueryEmbedder, QueryModelUnavailableError
from medquery_api.evaluation import DEFAULT_DATASET, EvaluationDatasetError, load_dataset
from medquery_api.knowledge_base import (
    IndexFormatError,
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    load_knowledge_base,
)
from medquery_api.llm import LLMProvider, LLMUnavailableError, create_provider
from medquery_api.llm.citations import Grounding
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


class LLMConfigResponse(CamelModel):
    """The configured LLM provider and model, and nothing else (no credentials or settings)."""

    provider: str  # "mock" or "claude"
    model: str  # the model requested; a fallback model may answer instead (see LLMInfo)


class AnswerRequest(RetrieveRequest):
    """The same question, embedding model and topK rules as POST /api/retrieve."""


class ProviderCitation(CamelModel):
    """A citation that passed its checks: the quoted text is exactly that span of the source."""

    document_index: int
    source_label: str  # the cited source ("S1"...); see `sources`
    cited_text: str
    start_char: int
    end_char: int


class DroppedCitationModel(CamelModel):
    """A citation that failed its checks, as the provider returned it."""

    part_index: int  # the answer part that carried it
    document_index: int | None  # None when the provider sent something other than a number
    cited_text: str | None
    start_char: int | None
    end_char: int | None
    reason: Literal["unknown_document", "invalid_range", "text_mismatch"]


class DroppedByReason(CamelModel):
    unknown_document: int
    invalid_range: int
    text_mismatch: int


class CitationCheckModel(CamelModel):
    """Checks, not proof: a valid citation quotes its source exactly; it does not show the
    answer is correct."""

    total: int  # citations the provider returned
    valid: int
    dropped: DroppedByReason
    validity_rate: float | None  # valid / total; None when there were no citations
    sentences: int
    supported_sentences: int  # sentences with at least one valid citation
    grounding_coverage: float | None  # supported_sentences / sentences; None for no sentences


class AnswerSentenceModel(CamelModel):
    # Character positions (code points) in the whole answer, all parts' text joined.
    start: int
    end: int
    supported: bool  # at least one valid citation supports this sentence


class AnswerPartModel(CamelModel):
    text: str
    citations: list[ProviderCitation]


class AnswerSource(CamelModel):
    label: str
    rank: int
    score: float
    source_file: str
    chunk_id: int
    article_title: str
    chunk_count: int
    title: str  # the title the provider saw, e.g. "Asthma (chunk 2 of 14)"
    text: str  # verbatim chunk text


class LLMInfo(CamelModel):
    provider: str
    model: str  # the model that produced the answer
    fell_back: bool  # a fallback model answered after the requested model declined


class TokenUsageModel(CamelModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class CostModel(CamelModel):
    estimated_usd: float | None  # None when no checked price is known
    prices_checked_on: str | None
    note: str


class AnswerTimings(CamelModel):
    model_load_ms: float
    embedding_ms: float
    search_ms: float
    generation_ms: float
    total_ms: float


class AnswerResponse(CamelModel):
    question: str
    embedding_model: str
    top_k: int
    llm: LLMInfo
    parts: list[AnswerPartModel]  # only valid citations are kept
    # How far valid citations support the answer; see llm/citations.py. Only "grounded" means
    # every sentence has a valid citation, and that is still a check, not proof.
    grounding: Literal[
        "grounded", "partly_grounded", "not_grounded", "not_enough_evidence", "refused"
    ]
    sentences: list[AnswerSentenceModel]
    citation_status: Literal["checked"]
    citation_check: CitationCheckModel
    dropped_citations: list[DroppedCitationModel]
    sources: list[AnswerSource]
    stop_reason: str
    refused: bool
    refusal_category: str | None
    usage: TokenUsageModel
    cost: CostModel
    timings: AnswerTimings


class ReferenceAnswerModel(CamelModel):
    text: str
    status: Literal["unverified"]  # historical text, not ground truth
    source: str


class InScopeQuestionModel(CamelModel):
    id: str
    question: str
    relevant_articles: list[str]
    reference_answer: ReferenceAnswerModel | None
    archive_question_id: int | None
    source: str


class OutOfScopeQuestionModel(CamelModel):
    id: str
    question: str
    reason: str
    acceptance_criteria: str
    automatic_pass_statuses: list[Grounding]
    otherwise: Literal["fail", "manual_review"]
    source: str


class EvaluationDatasetResponse(CamelModel):
    identifier: str
    format_version: int
    name: str
    description: str
    relevance_level: Literal["article"]
    in_scope_questions: list[InScopeQuestionModel]
    out_of_scope_questions: list[OutOfScopeQuestionModel]


def get_data_dir() -> Path:
    """Supplied to endpoints by FastAPI; tests replace it with a temporary folder."""
    return config.data_dir()


def get_evaluation_dataset_path() -> Path:
    """The evaluation dataset file; tests replace it."""
    return DEFAULT_DATASET


def get_knowledge_base_loader(
    data_dir: Annotated[Path, Depends(get_data_dir)],
) -> Callable[[], KnowledgeBase]:
    """A function that loads the knowledge base, called only once a request is valid."""
    return lambda: load_knowledge_base(data_dir)


_query_embedder = QueryEmbedder()


def get_query_embedder() -> QueryEmbedder:
    """The process's query embedder (it loads models only when used)."""
    return _query_embedder


_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """The configured LLM provider (MEDQUERY_LLM_PROVIDER; default "mock"), created once."""
    global _llm_provider
    if _llm_provider is None:
        try:
            _llm_provider = create_provider()
        except ValueError as error:
            raise HTTPException(
                status_code=503,
                detail="The LLM provider is not configured correctly. Set MEDQUERY_LLM_PROVIDER "
                "to mock or claude, then restart the backend.",
            ) from error
    return _llm_provider


def open_knowledge_base(load: Callable[[], KnowledgeBase]) -> KnowledgeBase:
    """Load the knowledge base, turning data problems into safe 503 responses."""
    try:
        return load()
    except KnowledgeBaseNotFoundError as error:
        raise HTTPException(status_code=503, detail=DATA_NOT_FOUND) from error
    except (KnowledgeBaseError, IndexFormatError, OSError) as error:
        logger.error("Research data failed its checks: %s", error)
        raise HTTPException(
            status_code=503,
            detail="The research data did not pass its integrity checks, so retrieval is "
            "unavailable. The server log says which check failed.",
        ) from error


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


@app.get("/api/llm", response_model=LLMConfigResponse)
def llm_config(
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> LLMConfigResponse:
    """Which LLM answers questions, so the frontend can show the right privacy notice."""
    return LLMConfigResponse(provider=provider.name, model=provider.model)


@app.get("/api/evaluation/dataset", response_model=EvaluationDatasetResponse)
def evaluation_dataset(
    path: Annotated[Path, Depends(get_evaluation_dataset_path)],
) -> EvaluationDatasetResponse:
    """The evaluation dataset (PRD 4.0), read-only. It needs no research data folder."""
    try:
        dataset = load_dataset(path)
    except EvaluationDatasetError as error:
        logger.error("%s", error)
        raise HTTPException(
            status_code=503,
            detail="The evaluation dataset file is not valid, so it cannot be shown. Run "
            "`uv run python -m medquery_api.evaluation` in the backend folder to see what to fix.",
        ) from error
    return EvaluationDatasetResponse.model_validate(asdict(dataset))


@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    request: RetrieveRequest,
    load: Annotated[Callable[[], KnowledgeBase], Depends(get_knowledge_base_loader)],
    embedder: Annotated[QueryEmbedder, Depends(get_query_embedder)],
) -> RetrieveResponse:
    """The research chunks most similar to a question, using one embedding model."""
    knowledge_base = open_knowledge_base(load)
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


@app.post("/api/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    load: Annotated[Callable[[], KnowledgeBase], Depends(get_knowledge_base_loader)],
    embedder: Annotated[QueryEmbedder, Depends(get_query_embedder)],
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> AnswerResponse:
    """A grounded answer: retrieve chunks, then ask the configured LLM to answer from them.

    Every citation is checked against the chunk it cites; invalid ones are dropped and counted.
    """
    knowledge_base = open_knowledge_base(load)
    try:
        outcome = answer_question(
            request.question,
            request.model,
            request.top_k,
            knowledge_base=knowledge_base,
            embedder=embedder,
            provider=provider,
        )
    except (QueryModelUnavailableError, LLMUnavailableError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    retrieval, context, result = outcome.retrieval, outcome.context, outcome.result
    check = outcome.citations
    return AnswerResponse(
        question=retrieval.question,
        embedding_model=retrieval.model.name,
        top_k=retrieval.top_k,
        llm=LLMInfo(provider=result.provider, model=result.model, fell_back=result.fell_back),
        parts=[
            AnswerPartModel(
                text=part.text,
                citations=[
                    ProviderCitation(
                        document_index=c.document_index,
                        source_label=context.source_for(c.document_index).label,
                        cited_text=c.cited_text,
                        start_char=c.start_char,
                        end_char=c.end_char,
                    )
                    for c in part.citations
                ],
            )
            for part in check.parts
        ],
        grounding=outcome.grounding,
        sentences=[
            AnswerSentenceModel(start=s.start, end=s.end, supported=s.supported)
            for s in check.sentence_list
        ],
        citation_status="checked",
        citation_check=CitationCheckModel(
            total=check.total,
            valid=check.valid,
            dropped=DroppedByReason(**check.dropped_by_reason),
            validity_rate=check.validity_rate,
            sentences=check.sentences,
            supported_sentences=check.supported_sentences,
            grounding_coverage=check.grounding_coverage,
        ),
        dropped_citations=[
            DroppedCitationModel(
                part_index=d.part_index,
                document_index=_int_or_none(d.citation.document_index),
                cited_text=d.citation.cited_text
                if isinstance(d.citation.cited_text, str)
                else None,
                start_char=_int_or_none(d.citation.start_char),
                end_char=_int_or_none(d.citation.end_char),
                reason=d.reason,
            )
            for d in check.dropped
        ],
        sources=[
            AnswerSource(
                label=source.label,
                rank=source.rank,
                score=source.score,
                source_file=source.source_file,
                chunk_id=source.chunk_id,
                article_title=source.article_title,
                chunk_count=source.chunk_count,
                title=document.title,
                text=document.text,
            )
            for source, document in zip(context.sources, context.request.documents, strict=True)
        ],
        stop_reason=result.stop_reason,
        refused=result.refused,
        refusal_category=result.refusal_category,
        usage=TokenUsageModel(**vars(result.usage)),
        cost=CostModel(
            estimated_usd=outcome.cost.usd,
            prices_checked_on=outcome.cost.checked_on,
            note=outcome.cost.note,
        ),
        timings=AnswerTimings(
            model_load_ms=round(retrieval.model_load_ms, 3),
            embedding_ms=round(retrieval.embedding_ms, 3),
            search_ms=round(retrieval.search_ms, 3),
            generation_ms=round(result.latency_ms, 3),
            total_ms=round(outcome.total_ms, 3),
        ),
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
