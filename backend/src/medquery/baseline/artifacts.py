"""Read-only access to the archived corpus and vectors of the research baseline."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from medquery.ingestion import (
    BASELINE_CHUNKING,
    BASELINE_CLEANING,
    Chunk,
    Chunker,
    clean_text,
    load_text_document,
)

# faiss IndexFlat on-disk layout: fourcc, d (int32), ntotal (int64), two dummy int64,
# is_trained (bool), metric_type (int32), then a uint64 float count and float32 data.
_FAISS_HEADER = struct.Struct("<4siqqq?i")
_FAISS_FLAT_CODES = {b"IxFI", b"IxF2"}


def load_baseline_chunks(data_dir: Path) -> list[Chunk]:
    """Re-create the 3,796 baseline chunks from `data/cleaned` with the baseline config.

    Order matches the archived embedding/FAISS rows (articles sorted by file name,
    chunks in document order). Golden tests prove the result equals `data/chunks`.
    """
    chunker = Chunker(BASELINE_CHUNKING)
    chunks: list[Chunk] = []
    for path in sorted((data_dir / "cleaned").glob("*.txt")):
        document = load_text_document(path)
        if clean_text(document.text, BASELINE_CLEANING) != document.text:
            raise ValueError(f"{path.name} is not in baseline-cleaned form")
        chunks.extend(chunker.chunk(document))
    return chunks


def read_faiss_flat_vectors(path: Path) -> np.ndarray:
    """Read the vectors of an archived FAISS `IndexFlat` file without importing faiss."""
    raw = path.read_bytes()
    fourcc, dim, ntotal, _, _, _, _ = _FAISS_HEADER.unpack_from(raw, 0)
    (count,) = struct.unpack_from("<Q", raw, _FAISS_HEADER.size)
    offset = _FAISS_HEADER.size + 8
    if fourcc not in _FAISS_FLAT_CODES or count != dim * ntotal:
        raise ValueError(f"{path} is not a plain FAISS IndexFlat file")
    if len(raw) != offset + count * 4:
        raise ValueError(f"{path} has unexpected size")
    return np.frombuffer(raw, dtype="<f4", count=count, offset=offset).reshape(ntotal, dim)


def read_faiss_metadata_keys(path: Path) -> list[tuple[str, int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [(r["source_file"], r["chunk_id"]) for r in records]
