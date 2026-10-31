from pathlib import Path

import numpy as np
import pytest

from medquery.config import ArchiveWriteError, get_settings
from medquery.embeddings import (
    BASELINE_EMBEDDING_SPECS,
    EmbeddingCompatibilityError,
    EmbeddingModelSpec,
    FakeEmbedder,
)
from medquery.ingestion.models import Chunk, make_chunk_uid
from medquery.vectorstore import (
    FaissVectorStore,
    ReadOnlyStoreError,
    StoreManifest,
    VectorStoreError,
)
from medquery.vectorstore.faiss_store import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE

DIM = 16
SPEC = EmbeddingModelSpec(name="fake", model_id="test/fake", dimension=DIM)
FP = SPEC.fingerprint()


def make_chunk(source: str, n: int, text: str | None = None) -> Chunk:
    body = text or f"{source} chunk {n}"
    return Chunk(
        chunk_uid=make_chunk_uid(source, n),
        source_id=source,
        chunk_id=n,
        text=body,
        chunk_length=len(body),
        start_index=0,
        metadata={"title": source},
    )


def unit_rows(rng: np.random.Generator, n: int, dim: int = DIM) -> np.ndarray:
    rows = rng.standard_normal((n, dim)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture
def corpus() -> tuple[list[Chunk], np.ndarray]:
    chunks = [make_chunk(f"doc{d}.txt", n) for d in range(5) for n in range(1, 5)]
    return chunks, unit_rows(np.random.default_rng(0), len(chunks))


@pytest.fixture
def store(corpus) -> FaissVectorStore:
    chunks, vectors = corpus
    store = FaissVectorStore(FP, "test-kb", build_config={"note": "unit test"})
    store.add(chunks, vectors)
    return store


def brute_force(vectors: np.ndarray, query: np.ndarray, rows: list[int], k: int) -> list[int]:
    scores = vectors[rows] @ query
    return [rows[i] for i in np.argsort(-scores)[:k]]


# --- Construction and adding ---------------------------------------------------------


def test_new_store_is_empty_and_writable() -> None:
    store = FaissVectorStore(FP, "kb")
    assert len(store) == 0 and not store.read_only and store.provenance == "new_run"


@pytest.mark.parametrize("bad", ["", "Has Upper", "../escape", "a" * 65, "-leading"])
def test_invalid_knowledge_base_ids_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        FaissVectorStore(FP, bad)


def test_add_rejects_dimension_and_count_mismatch(corpus) -> None:
    chunks, vectors = corpus
    store = FaissVectorStore(FP, "kb")
    with pytest.raises(ValueError, match="shape"):
        store.add(chunks, unit_rows(np.random.default_rng(1), len(chunks), DIM + 1))
    with pytest.raises(ValueError, match="chunks but"):
        store.add(chunks[:-1], vectors)
    assert len(store) == 0


def test_add_rejects_non_normalised_and_non_finite_vectors(corpus) -> None:
    chunks, vectors = corpus
    store = FaissVectorStore(FP, "kb")
    with pytest.raises(ValueError, match="unit-normalised"):
        store.add(chunks, vectors * 2)
    broken = vectors.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        store.add(chunks, broken)


def test_unnormalised_fingerprint_accepts_any_finite_vectors(corpus) -> None:
    chunks, vectors = corpus
    raw_fp = SPEC.model_copy(update={"normalize": False}).fingerprint()
    store = FaissVectorStore(raw_fp, "kb")
    store.add(chunks, vectors * 3)
    assert len(store) == len(chunks)


def test_duplicate_chunk_uids_are_rejected(store, corpus) -> None:
    chunks, vectors = corpus
    with pytest.raises(VectorStoreError, match="Duplicate"):
        store.add(chunks[:1], vectors[:1])
    fresh = FaissVectorStore(FP, "kb")
    with pytest.raises(VectorStoreError, match="Duplicate"):
        fresh.add([chunks[0], chunks[0]], vectors[:2])
    assert len(fresh) == 0


def test_adding_in_batches_equals_adding_at_once(store, corpus) -> None:
    chunks, vectors = corpus
    batched = FaissVectorStore(FP, "kb")
    batched.add(chunks[:7], vectors[:7])
    batched.add(chunks[7:], vectors[7:])
    batched.add([], np.zeros((0, DIM), np.float32))
    query = unit_rows(np.random.default_rng(5), 1)
    a = store.search(query, 6, query_fingerprint=FP)[0]
    b = batched.search(query, 6, query_fingerprint=FP)[0]
    assert [r.chunk.chunk_uid for r in a] == [r.chunk.chunk_uid for r in b]


# --- Searching -------------------------------------------------------------------------


def test_search_matches_brute_force_ranking_and_scores(store, corpus) -> None:
    chunks, vectors = corpus
    queries = unit_rows(np.random.default_rng(42), 6)
    results = store.search(queries, 5, query_fingerprint=FP)
    assert len(results) == len(queries)
    for query, hits in zip(queries, results, strict=True):
        expected = brute_force(vectors, query, list(range(len(chunks))), 5)
        assert [h.chunk.chunk_uid for h in hits] == [chunks[i].chunk_uid for i in expected]
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
        assert [h.score for h in hits] == pytest.approx(
            [float(vectors[i] @ query) for i in expected]
        )


def test_single_vector_query_is_accepted(store, corpus) -> None:
    _, vectors = corpus
    [hits] = store.search(vectors[3], 1, query_fingerprint=FP)
    assert hits[0].chunk.chunk_uid == store.chunks[3].chunk_uid
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_store_returns_all_chunks(store) -> None:
    [hits] = store.search(unit_rows(np.random.default_rng(1), 1), 100, query_fingerprint=FP)
    assert len(hits) == len(store)
    assert len({h.chunk.chunk_uid for h in hits}) == len(store)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(store, k) -> None:
    with pytest.raises(ValueError):
        store.search(unit_rows(np.random.default_rng(1), 1), k, query_fingerprint=FP)


def test_empty_store_and_empty_query_batch() -> None:
    empty = FaissVectorStore(FP, "kb")
    assert empty.search(unit_rows(np.random.default_rng(1), 2), 3, query_fingerprint=FP) == [[], []]
    assert empty.search(np.zeros((0, DIM), np.float32), 3, query_fingerprint=FP) == []


def test_query_with_wrong_dimension_or_unnormalised_is_rejected(store) -> None:
    with pytest.raises(ValueError, match="shape"):
        store.search(unit_rows(np.random.default_rng(1), 1, DIM + 1), 3, query_fingerprint=FP)
    with pytest.raises(ValueError, match="unit-normalised"):
        store.search(unit_rows(np.random.default_rng(1), 1) * 5, 3, query_fingerprint=FP)


@pytest.mark.parametrize(
    "update",
    [
        {"model_id": "other/model"},
        {"query_prefix": "query: "},
        {"passage_prefix": "passage: "},
        {"normalize": False},
    ],
)
def test_incompatible_query_fingerprint_is_rejected(store, update) -> None:
    query_fp = SPEC.model_copy(update=update).fingerprint()
    with pytest.raises(EmbeddingCompatibilityError):
        store.search(unit_rows(np.random.default_rng(1), 1), 3, query_fingerprint=query_fp)


def test_different_pinned_revisions_are_rejected() -> None:
    store = FaissVectorStore(SPEC.model_copy(update={"revision": "b" * 40}).fingerprint(), "kb")
    query_fp = SPEC.model_copy(update={"revision": "a" * 40}).fingerprint()
    with pytest.raises(EmbeddingCompatibilityError, match="revision"):
        store.search(unit_rows(np.random.default_rng(1), 1), 3, query_fingerprint=query_fp)


def test_query_from_a_different_baseline_model_is_rejected() -> None:
    """MiniLM and BGE are both 384-dim: only the fingerprint tells them apart."""
    minilm, bge = BASELINE_EMBEDDING_SPECS["MiniLM"], BASELINE_EMBEDDING_SPECS["BGE"]
    store = FaissVectorStore(minilm.fingerprint(), "kb")
    store.add([make_chunk("a.txt", 1)], unit_rows(np.random.default_rng(1), 1, 384))
    with pytest.raises(EmbeddingCompatibilityError, match="model_id"):
        store.search(
            unit_rows(np.random.default_rng(2), 1, 384), 1, query_fingerprint=bge.fingerprint()
        )


# --- Source filters ----------------------------------------------------------------


def test_source_filter_restricts_results_and_keeps_exact_ranking(store, corpus) -> None:
    chunks, vectors = corpus
    allowed = {"doc1.txt", "doc3.txt"}
    rows = [i for i, c in enumerate(chunks) if c.source_id in allowed]
    query = unit_rows(np.random.default_rng(9), 1)
    [hits] = store.search(query, 5, query_fingerprint=FP, source_ids=allowed)
    assert {h.chunk.source_id for h in hits} <= allowed
    assert [h.chunk.chunk_uid for h in hits] == [
        chunks[i].chunk_uid for i in brute_force(vectors, query[0], rows, 5)
    ]


def test_source_filter_returns_fewer_than_k_when_few_chunks_match(store) -> None:
    [hits] = store.search(
        unit_rows(np.random.default_rng(1), 1), 10, query_fingerprint=FP, source_ids=["doc0.txt"]
    )
    assert len(hits) == 4
    assert [h.rank for h in hits] == [1, 2, 3, 4]


def test_source_filter_with_no_matching_documents(store) -> None:
    assert store.search(
        unit_rows(np.random.default_rng(1), 2), 3, query_fingerprint=FP, source_ids={"missing"}
    ) == [[], []]


def test_source_filter_rejects_a_bare_string(store) -> None:
    with pytest.raises(TypeError):
        store.search(
            unit_rows(np.random.default_rng(1), 1), 3, query_fingerprint=FP, source_ids="doc0.txt"
        )


# --- Persistence ---------------------------------------------------------------------


def test_save_and_load_round_trip(store, tmp_path: Path) -> None:
    manifest = store.save(tmp_path / "kb")
    assert isinstance(manifest, StoreManifest)
    assert manifest.vector_count == len(store) and manifest.fingerprint == FP
    assert manifest.build_config == {"note": "unit test"}
    assert set(manifest.files) == {INDEX_FILE, CHUNKS_FILE}

    loaded = FaissVectorStore.load(tmp_path / "kb")
    assert loaded.chunks == store.chunks
    assert loaded.knowledge_base_id == "test-kb" and loaded.fingerprint == FP
    queries = unit_rows(np.random.default_rng(3), 4)
    assert loaded.search(queries, 5, query_fingerprint=FP) == store.search(
        queries, 5, query_fingerprint=FP
    )


def test_save_refuses_to_overwrite_unless_asked(store, tmp_path: Path) -> None:
    store.save(tmp_path / "kb")
    with pytest.raises(VectorStoreError, match="already exists"):
        store.save(tmp_path / "kb")
    store.save(tmp_path / "kb", overwrite=True)


def test_load_detects_tampered_files(store, tmp_path: Path) -> None:
    store.save(tmp_path / "kb")
    chunks_path = tmp_path / "kb" / CHUNKS_FILE
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VectorStoreError, match="checksum"):
        FaissVectorStore.load(tmp_path / "kb")


def test_load_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="manifest"):
        FaissVectorStore.load(tmp_path)


def test_loaded_read_only_store_refuses_changes(store, corpus, tmp_path: Path) -> None:
    chunks, vectors = corpus
    store.save(tmp_path / "kb")
    loaded = FaissVectorStore.load(tmp_path / "kb", read_only=True)
    assert loaded.read_only
    with pytest.raises(ReadOnlyStoreError):
        loaded.add([make_chunk("new.txt", 1)], vectors[:1])
    with pytest.raises(ReadOnlyStoreError):
        loaded.save(tmp_path / "copy")
    assert not (tmp_path / "copy" / MANIFEST_FILE).exists()


def test_save_inside_the_research_archive_is_refused(store) -> None:
    target = get_settings().archive_dir / "data" / "faiss" / "new-store"
    with pytest.raises(ArchiveWriteError):
        store.save(target)
    assert not target.exists()


def test_store_built_from_fake_embedder_retrieves_matching_text() -> None:
    embedder = FakeEmbedder(SPEC)
    chunks = [make_chunk(f"d{i}.txt", 1, text=f"text number {i}") for i in range(10)]
    store = FaissVectorStore(embedder.fingerprint(), "kb")
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    [hits] = store.search(
        embedder.embed_documents(["text number 7"]), 1, query_fingerprint=embedder.fingerprint()
    )
    assert hits[0].chunk.chunk_uid == "d7.txt#1"
