"""The research knowledge base: the archive's chunks and its four FAISS indexes, read only.

Row i of every index is the vector for chunk i (PRD 2.0 checked this against metadata.json).
Everything is checked on loading, and a knowledge base is loaded once per data folder and then
kept in memory. Search is exact: every vector is scored against the query.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from medquery_api import config
from medquery_api.embeddings.models import MODELS, EmbeddingModel, get_model
from medquery_api.knowledge_base.flat_index import FlatIndex, read_flat_index
from medquery_api.processing import Chunk

CHUNK_FIELDS = ("chunk_id", "source_file", "chunk_length", "text")
# Query vectors must have length 1 within this tolerance, like the research's queries
# (normalize_embeddings=True), so that scores are cosine similarities.
NORM_TOLERANCE = 1e-4


class KnowledgeBaseError(ValueError):
    """The data folder is missing something, or its files do not line up."""


@dataclass(frozen=True)
class SearchHit:
    rank: int  # 1 for the best match
    row: int  # position in the index (and in KnowledgeBase.chunks)
    score: float  # inner product with the query; cosine similarity for unit vectors
    chunk: Chunk


@dataclass(frozen=True)
class ModelIndex:
    model: EmbeddingModel
    index: FlatIndex


class KnowledgeBase:
    def __init__(self, data_dir: Path, chunks: list[Chunk], indexes: dict[str, ModelIndex]):
        self.data_dir = data_dir
        self.chunks: tuple[Chunk, ...] = tuple(chunks)
        self.indexes = indexes
        self._rows = {(chunk.source_file, chunk.chunk_id): row for row, chunk in enumerate(chunks)}

    @property
    def articles(self) -> list[str]:
        """The articles' file names, in index order."""
        return list(dict.fromkeys(chunk.source_file for chunk in self.chunks))

    def chunk(self, row: int) -> Chunk:
        return self.chunks[row]

    def find_row(self, source_file: str, chunk_id: int) -> int:
        try:
            return self._rows[(source_file, chunk_id)]
        except KeyError:
            raise KeyError(f"no chunk {chunk_id} in {source_file}") from None

    def search(self, model: str, query_vector: Sequence[float], top_k: int = 5) -> list[SearchHit]:
        """The top_k chunks whose vectors have the highest inner product with query_vector.

        Ordered by score, highest first; chunks with exactly equal scores in row order.
        """
        index = self.indexes[get_model(model).name].index
        query = _check_query(query_vector, index.dimension)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= index.count:
            raise ValueError(f"top_k must be a whole number from 1 to {index.count}, got {top_k!r}")
        scores = index.vectors @ query
        # lexsort's last key is the primary one: highest score first, then lowest row.
        order = np.lexsort((np.arange(index.count), -scores))[:top_k]
        return [
            SearchHit(rank=rank, row=int(row), score=float(scores[row]), chunk=self.chunks[row])
            for rank, row in enumerate(order, start=1)
        ]


def _check_query(query_vector: Sequence[float], dimension: int) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float64)
    if query.shape != (dimension,):
        raise ValueError(f"the query vector must have {dimension} numbers, got shape {query.shape}")
    if not np.isfinite(query).all():
        raise ValueError("the query vector contains NaN or infinite values")
    deviation = abs(float(np.linalg.norm(query)) - 1.0)
    if deviation > NORM_TOLERANCE:
        raise ValueError(
            f"the query vector's length differs from 1 by {deviation:.3e} "
            f"(tolerance {NORM_TOLERANCE:g}); normalise it first"
        )
    return query.astype(np.float32)


def load_knowledge_base(data_dir: Path | None = None) -> KnowledgeBase:
    """The knowledge base for a data folder (default: the research archive), loaded once.

    Folders are cached by their resolved path, so the same folder spelled differently gives
    the same instance, and different folders give different ones.
    """
    return _load((data_dir or config.data_dir()).resolve())


@cache
def _load(data_dir: Path) -> KnowledgeBase:
    chunks = _read_chunks(data_dir)
    records = [chunk.as_record() for chunk in chunks]

    faiss_dir = data_dir / "faiss"
    folders = (
        sorted(p.name for p in faiss_dir.iterdir() if p.is_dir()) if faiss_dir.is_dir() else []
    )
    if sorted(folders) != sorted(MODELS):
        raise KnowledgeBaseError(
            f"{faiss_dir} must contain exactly the model folders {sorted(MODELS)}; found {folders}"
        )

    indexes = {}
    for name, model in MODELS.items():
        folder = faiss_dir / name
        index = read_flat_index(folder / "faiss_index.bin")
        if (index.count, index.dimension) != (len(chunks), model.dimension):
            raise KnowledgeBaseError(
                f"{name} index has {index.count} vectors of size {index.dimension}; expected "
                f"{len(chunks)} (one per chunk) of size {model.dimension}"
            )
        with (folder / "metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata != records:
            raise KnowledgeBaseError(f"{name} metadata.json {_first_difference(metadata, records)}")
        indexes[name] = ModelIndex(model=model, index=index)
    return KnowledgeBase(data_dir, chunks, indexes)


def _read_chunks(data_dir: Path) -> list[Chunk]:
    """All chunk records, articles in file-name order (the order the indexes were built in)."""
    chunk_dir = data_dir / "chunks"
    if not chunk_dir.is_dir():
        raise KnowledgeBaseError(f"research data not found: no chunks folder in {data_dir}")
    chunks = []
    for path in sorted(chunk_dir.glob("*.json"), key=lambda p: p.name):
        with path.open("r", encoding="utf-8") as file:
            for record in json.load(file):
                if not isinstance(record, dict) or tuple(record) != CHUNK_FIELDS:
                    raise KnowledgeBaseError(f"{path.name}: unexpected chunk record format")
                chunks.append(Chunk(**record))
    return chunks


def _first_difference(metadata: list, records: list[dict]) -> str:
    if len(metadata) != len(records):
        return f"has {len(metadata)} rows; the chunk files have {len(records)}"
    row = next(i for i, (a, b) in enumerate(zip(metadata, records, strict=True)) if a != b)
    return f"does not match the chunk files at row {row}"
