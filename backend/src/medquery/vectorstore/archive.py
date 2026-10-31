"""Read-only access to the dissertation's FAISS indexes, in place.

`Healthcare_RAG/data/faiss/<Model>/faiss_index.bin` is opened directly (never copied
or written). Chunks are re-created from `data/cleaned` with the baseline cleaning and
chunking configuration, then checked field by field against the archived
`metadata.json`, so every vector row is tied to the exact chunk it was built from.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss

from medquery.baseline.artifacts import load_baseline_chunks
from medquery.config import get_settings
from medquery.embeddings.specs import BASELINE_EMBEDDING_SPECS
from medquery.ingestion import BASELINE_CHUNKING, BASELINE_CLEANING
from medquery.vectorstore.base import VectorStoreError
from medquery.vectorstore.faiss_store import FaissVectorStore

BASELINE_KNOWLEDGE_BASE_ID = "nhs-baseline"


def load_archived_store(model_name: str, archive_dir: Path | None = None) -> FaissVectorStore:
    """Open the archived index of a baseline model as a read-only store."""
    if model_name not in BASELINE_EMBEDDING_SPECS:
        raise KeyError(f"Unknown baseline model {model_name!r}")
    spec = BASELINE_EMBEDDING_SPECS[model_name]
    data_dir = (archive_dir or get_settings().archive_dir) / "data"
    model_dir = data_dir / "faiss" / model_name

    chunks = load_baseline_chunks(data_dir)
    records = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(records) != len(chunks):
        raise VectorStoreError(
            f"{model_name}: metadata has {len(records)} records, expected {len(chunks)}"
        )
    for row, (record, chunk) in enumerate(zip(records, chunks, strict=True)):
        if record != chunk.to_legacy_record():
            raise VectorStoreError(f"{model_name}: metadata row {row} does not match its chunk")

    index = faiss.read_index(str(model_dir / "faiss_index.bin"))
    store = FaissVectorStore(
        spec.fingerprint(),
        BASELINE_KNOWLEDGE_BASE_ID,
        provenance="original_dissertation",
        build_config={
            "source": f"Healthcare_RAG/data/faiss/{model_name}/faiss_index.bin",
            "embedding_model": model_name,
            "cleaning": BASELINE_CLEANING.model_dump(mode="json"),
            "chunking": BASELINE_CHUNKING.model_dump(mode="json"),
        },
    )
    store._attach(index, chunks)
    store._read_only = True
    return store
