"""The dissertation's FAISS indexes, opened in place as read-only stores."""

from pathlib import Path

import numpy as np
import pytest

from medquery.baseline.artifacts import read_faiss_flat_vectors
from medquery.baseline.manifest import verify_manifest
from medquery.config import get_settings
from medquery.embeddings import BASELINE_EMBEDDING_SPECS, EmbeddingCompatibilityError
from medquery.vectorstore import ReadOnlyStoreError, load_archived_store

pytestmark = pytest.mark.archive
MODELS = list(BASELINE_EMBEDDING_SPECS)


@pytest.fixture(scope="module")
def archive_dir() -> Path:
    path = get_settings().archive_dir
    if not (path / "data" / "faiss").is_dir():
        pytest.skip("Research archive not available on this machine")
    return path


@pytest.fixture(scope="module")
def stores(archive_dir: Path):
    return {name: load_archived_store(name, archive_dir) for name in MODELS}


@pytest.mark.parametrize("name", MODELS)
def test_archived_store_shape_and_provenance(stores, name: str) -> None:
    store = stores[name]
    assert len(store) == 3796
    assert store.fingerprint == BASELINE_EMBEDDING_SPECS[name].fingerprint()
    assert store.read_only and store.provenance == "original_dissertation"
    assert store.knowledge_base_id == "nhs-baseline"
    assert len(store.source_ids()) == 698


@pytest.mark.parametrize("name", MODELS)
def test_each_archived_vector_retrieves_its_own_chunk(stores, archive_dir: Path, name: str) -> None:
    """Row-to-chunk alignment: searching with stored row i returns chunk i first."""
    store = stores[name]
    vectors = read_faiss_flat_vectors(archive_dir / "data" / "faiss" / name / "faiss_index.bin")
    rows = list(range(0, 3796, 97))
    results = store.search(
        vectors[rows], 1, query_fingerprint=BASELINE_EMBEDDING_SPECS[name].fingerprint()
    )
    for row, [top] in zip(rows, results, strict=True):
        if top.chunk.chunk_uid != store.chunks[row].chunk_uid:
            # Only acceptable if another chunk has an identical vector (duplicate text).
            assert top.chunk.text == store.chunks[row].text, (name, row)
        assert top.score == pytest.approx(1.0, abs=1e-4)


def test_archived_stores_refuse_changes(stores) -> None:
    store = stores["BGE"]
    with pytest.raises(ReadOnlyStoreError):
        store.add([store.chunks[0]], np.zeros((1, 384), np.float32))
    with pytest.raises(ReadOnlyStoreError):
        store.save(Path("unused"))


def test_archived_store_rejects_queries_from_another_model(stores) -> None:
    with pytest.raises(EmbeddingCompatibilityError):
        stores["MiniLM"].search(
            np.eye(1, 384, dtype=np.float32),
            1,
            query_fingerprint=BASELINE_EMBEDDING_SPECS["BGE"].fingerprint(),
        )


def test_opening_archived_stores_leaves_the_archive_unchanged(stores, archive_dir: Path) -> None:
    assert verify_manifest(archive_dir, get_settings().baseline_manifest_path).ok
