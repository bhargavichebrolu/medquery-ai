import struct
from pathlib import Path

import numpy as np
import pytest

from medquery.baseline.artifacts import (
    load_baseline_chunks,
    read_faiss_flat_vectors,
    read_faiss_metadata_keys,
)
from medquery.config import get_settings
from medquery.embeddings import BASELINE_EMBEDDING_SPECS


def write_flat_index(path: Path, vectors: np.ndarray, fourcc: bytes = b"IxFI") -> None:
    n, d = vectors.shape
    header = struct.pack("<4siqqq?i", fourcc, d, n, 1 << 20, 1 << 20, True, 0)
    path.write_bytes(header + struct.pack("<Q", n * d) + vectors.astype("<f4").tobytes())


def test_reads_flat_index_vectors(tmp_path: Path) -> None:
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_flat_index(tmp_path / "i.bin", vectors)
    assert np.array_equal(read_faiss_flat_vectors(tmp_path / "i.bin"), vectors)


def test_rejects_non_flat_or_truncated_files(tmp_path: Path) -> None:
    write_flat_index(tmp_path / "hnsw.bin", np.zeros((2, 2), np.float32), fourcc=b"IHNf")
    with pytest.raises(ValueError):
        read_faiss_flat_vectors(tmp_path / "hnsw.bin")
    write_flat_index(tmp_path / "cut.bin", np.zeros((2, 2), np.float32))
    (tmp_path / "cut.bin").write_bytes((tmp_path / "cut.bin").read_bytes()[:-4])
    with pytest.raises(ValueError):
        read_faiss_flat_vectors(tmp_path / "cut.bin")


@pytest.fixture(scope="module")
def data_dir() -> Path:
    path = get_settings().archive_dir / "data"
    if not (path / "faiss").is_dir():
        pytest.skip("Research archive not available on this machine")
    return path


@pytest.mark.archive
def test_archived_indexes_are_compatible_with_baseline_specs(data_dir: Path) -> None:
    """Every archived index has the spec's dimension and the baseline chunk order."""
    keys = [(c.source_id, c.chunk_id) for c in load_baseline_chunks(data_dir)]
    assert len(keys) == 3796
    for name, spec in BASELINE_EMBEDDING_SPECS.items():
        vectors = read_faiss_flat_vectors(data_dir / "faiss" / name / "faiss_index.bin")
        assert vectors.shape == (3796, spec.dimension), name
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4), name
        assert read_faiss_metadata_keys(data_dir / "faiss" / name / "metadata.json") == keys
