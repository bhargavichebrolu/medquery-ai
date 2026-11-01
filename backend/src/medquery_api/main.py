"""The MedQuery AI web API."""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from medquery_api import config
from medquery_api.dataset import DataFolderNotFoundError, summarise_dataset

app = FastAPI(title="MedQuery AI API", version="0.1.0")


class HealthResponse(BaseModel):
    status: Literal["ok"]


class DatasetSummaryResponse(BaseModel):
    # Python uses snake_case; the JSON sent to the frontend uses camelCase.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    articles: int
    chunks: int
    embedding_indexes: list[str]
    source: str


def get_data_dir() -> Path:
    """Supplied to endpoints by FastAPI; tests replace it with a temporary folder."""
    return config.data_dir()


@app.get("/api/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/dataset/summary", response_model=DatasetSummaryResponse)
def dataset_summary(data_dir: Annotated[Path, Depends(get_data_dir)]) -> DatasetSummaryResponse:
    try:
        summary = summarise_dataset(data_dir)
    except DataFolderNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Research data folder not found. Restore Healthcare_RAG/data "
            "or set MEDQUERY_DATA_DIR.",
        ) from error

    source = "Healthcare_RAG/data" if data_dir == config.DEFAULT_DATA_DIR else "MEDQUERY_DATA_DIR"
    return DatasetSummaryResponse(
        articles=summary.articles,
        chunks=summary.chunks,
        embedding_indexes=summary.embedding_indexes,
        source=source,
    )
