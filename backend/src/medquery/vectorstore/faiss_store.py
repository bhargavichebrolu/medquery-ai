"""Exact FAISS vector store (`IndexFlatIP`), the index type used by the dissertation.

On disk a store is a directory with three files:

    index.faiss    FAISS index; row i is the vector of line i of chunks.jsonl
    chunks.jsonl   one serialised `Chunk` per line
    manifest.json  `StoreManifest` with the fingerprint, build config and SHA-256 of
                   the other two files (verified on load)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from medquery.config import get_settings
from medquery.embeddings.specs import EmbeddingFingerprint, ensure_compatible
from medquery.ingestion.models import Chunk
from medquery.vectorstore.base import (
    KNOWLEDGE_BASE_ID_PATTERN,
    NORM_TOLERANCE,
    Provenance,
    ReadOnlyStoreError,
    SearchResult,
    StoreManifest,
    VectorStore,
    VectorStoreError,
)

STORE_TYPE = "faiss_flat_ip"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FaissVectorStore(VectorStore):
    def __init__(
        self,
        fingerprint: EmbeddingFingerprint,
        knowledge_base_id: str,
        *,
        provenance: Provenance = "new_run",
        build_config: dict[str, Any] | None = None,
    ) -> None:
        if not re.fullmatch(KNOWLEDGE_BASE_ID_PATTERN, knowledge_base_id):
            raise ValueError(f"Invalid knowledge_base_id: {knowledge_base_id!r}")
        self._fingerprint = fingerprint
        self._knowledge_base_id = knowledge_base_id
        self._provenance: Provenance = provenance
        self._build_config = dict(build_config or {})
        self._index = faiss.IndexFlatIP(fingerprint.dimension)
        self._chunks: list[Chunk] = []
        self._row_by_uid: dict[str, int] = {}
        self._read_only = False

    # --- Properties -----------------------------------------------------------------

    @property
    def knowledge_base_id(self) -> str:
        return self._knowledge_base_id

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        return self._fingerprint

    @property
    def provenance(self) -> Provenance:
        return self._provenance

    @property
    def read_only(self) -> bool:
        return self._read_only

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def source_ids(self) -> set[str]:
        return {chunk.source_id for chunk in self._chunks}

    # --- Writing --------------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        self._require_writable()
        matrix = self._as_matrix(vectors, "document vectors")
        if len(chunks) != matrix.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {matrix.shape[0]} vectors")
        new_uids = [chunk.chunk_uid for chunk in chunks]
        duplicates = sorted(
            {uid for uid in new_uids if uid in self._row_by_uid}
            | {uid for uid in new_uids if new_uids.count(uid) > 1}
        )
        if duplicates:
            raise VectorStoreError(f"Duplicate chunk_uid(s): {duplicates[:5]}")
        self._append(list(chunks), matrix)

    def save(self, directory: Path, *, overwrite: bool = False) -> StoreManifest:
        self._require_writable()
        target = get_settings().ensure_writable(directory)
        if (target / MANIFEST_FILE).exists() and not overwrite:
            raise VectorStoreError(f"A store already exists at {target}")
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target / INDEX_FILE))
        with (target / CHUNKS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json() + "\n")
        manifest = StoreManifest(
            store_type=STORE_TYPE,
            knowledge_base_id=self._knowledge_base_id,
            fingerprint=self._fingerprint,
            vector_count=len(self._chunks),
            provenance=self._provenance,
            created_at=datetime.now(UTC),
            build_config=self._build_config,
            files={name: _sha256(target / name) for name in (INDEX_FILE, CHUNKS_FILE)},
        )
        (target / MANIFEST_FILE).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return manifest

    @classmethod
    def load(cls, directory: Path, *, read_only: bool = False) -> FaissVectorStore:
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise VectorStoreError(f"No store manifest at {manifest_path}")
        manifest = StoreManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.store_type != STORE_TYPE:
            raise VectorStoreError(f"Unsupported store type: {manifest.store_type!r}")
        for name in (INDEX_FILE, CHUNKS_FILE):
            path = directory / name
            if not path.is_file() or _sha256(path) != manifest.files.get(name):
                raise VectorStoreError(f"{name} is missing or does not match its checksum")
        index = faiss.read_index(str(directory / INDEX_FILE))
        with (directory / CHUNKS_FILE).open(encoding="utf-8") as handle:
            chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
        store = cls(
            manifest.fingerprint,
            manifest.knowledge_base_id,
            provenance=manifest.provenance,
            build_config=manifest.build_config,
        )
        store._attach(index, chunks)
        if len(store) != manifest.vector_count:
            raise VectorStoreError("Vector count does not match the manifest")
        store._read_only = read_only
        return store

    # --- Searching ------------------------------------------------------------------

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        *,
        query_fingerprint: EmbeddingFingerprint,
        source_ids: Collection[str] | None = None,
    ) -> list[list[SearchResult]]:
        ensure_compatible(self._fingerprint, query_fingerprint)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        queries = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        queries = self._as_matrix(queries, "query vectors")
        if queries.shape[0] == 0:
            return []
        if not self._chunks:
            return [[] for _ in range(queries.shape[0])]

        params = None
        if source_ids is not None:
            if isinstance(source_ids, str):
                raise TypeError("source_ids must be a collection of ids, not a string")
            wanted = set(source_ids)
            rows = np.array(
                [i for i, chunk in enumerate(self._chunks) if chunk.source_id in wanted],
                dtype=np.int64,
            )
            if rows.size == 0:
                return [[] for _ in range(queries.shape[0])]
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(rows))

        scores, indices = self._index.search(
            np.ascontiguousarray(queries), min(k, len(self._chunks)), params=params
        )
        return [
            [
                SearchResult(rank=rank, score=float(score), chunk=self._chunks[row])
                for rank, (score, row) in enumerate(
                    ((s, r) for s, r in zip(row_scores, row_ids, strict=True) if r >= 0),
                    start=1,
                )
            ]
            for row_scores, row_ids in zip(scores, indices, strict=True)
        ]

    # --- Internals ------------------------------------------------------------------

    def _require_writable(self) -> None:
        if self._read_only:
            raise ReadOnlyStoreError(f"Store {self._knowledge_base_id!r} is read-only")

    def _as_matrix(self, vectors: np.ndarray, label: str) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._fingerprint.dimension:
            raise ValueError(
                f"{label} must have shape (n, {self._fingerprint.dimension}), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{label} contain NaN or infinite values")
        if self._fingerprint.normalize and matrix.shape[0]:
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
                raise ValueError(f"{label} must be unit-normalised for this store")
        return matrix

    def _append(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if not chunks:
            return
        start = len(self._chunks)
        self._index.add(np.ascontiguousarray(matrix))
        self._chunks.extend(chunks)
        self._row_by_uid.update({c.chunk_uid: start + i for i, c in enumerate(chunks)})

    def _attach(self, index: faiss.Index, chunks: list[Chunk]) -> None:
        """Use an existing index whose row i corresponds to chunks[i]."""
        if not isinstance(index, faiss.IndexFlatIP):
            raise VectorStoreError(f"Expected IndexFlatIP, got {type(index).__name__}")
        if index.d != self._fingerprint.dimension:
            raise VectorStoreError(
                f"Index dimension {index.d} != fingerprint dimension {self._fingerprint.dimension}"
            )
        if index.ntotal != len(chunks):
            raise VectorStoreError(f"Index has {index.ntotal} vectors but {len(chunks)} chunks")
        uids = [chunk.chunk_uid for chunk in chunks]
        if len(set(uids)) != len(uids):
            raise VectorStoreError("Duplicate chunk_uid in stored chunks")
        self._index = index
        self._chunks = list(chunks)
        self._row_by_uid = {uid: i for i, uid in enumerate(uids)}
