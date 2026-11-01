"""Count what is in the research data folder. Files are only ever opened for reading."""

import json
from dataclasses import dataclass
from pathlib import Path


class DataFolderNotFoundError(Exception):
    """The research data folder (or its chunks folder) does not exist."""


@dataclass(frozen=True)
class DatasetSummary:
    articles: int
    chunks: int
    embedding_indexes: list[str]


def summarise_dataset(data_dir: Path) -> DatasetSummary:
    """Count articles, chunks and embedding indexes in a research data folder.

    - articles: the JSON files in data/chunks (one per article)
    - chunks: the records inside those files
    - embedding_indexes: model folders in data/faiss that contain a faiss_index.bin
    """
    chunks_dir = data_dir / "chunks"
    if not chunks_dir.is_dir():
        raise DataFolderNotFoundError(f"No chunks folder in {data_dir}")

    chunk_files = sorted(chunks_dir.glob("*.json"))
    chunk_count = 0
    for path in chunk_files:
        with path.open("r", encoding="utf-8") as file:
            chunk_count += len(json.load(file))

    faiss_dir = data_dir / "faiss"
    indexes = (
        sorted(
            folder.name
            for folder in faiss_dir.iterdir()
            if folder.is_dir() and (folder / "faiss_index.bin").is_file()
        )
        if faiss_dir.is_dir()
        else []
    )

    return DatasetSummary(articles=len(chunk_files), chunks=chunk_count, embedding_indexes=indexes)
