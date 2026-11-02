"""Tests for medquery_api.knowledge_base: reading the FAISS indexes, loading, and exact search.

The unit tests build a tiny data folder in a temporary directory and always run. The archive
tests read Healthcare_RAG/data (or MEDQUERY_DATA_DIR) only, and are skipped, with the reason
shown, when it is absent. Run with `-rP` to see the measured values.
"""

import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from medquery_api import config
from medquery_api.embeddings import MODELS
from medquery_api.embeddings.validation import read_archived_embeddings
from medquery_api.knowledge_base import (
    IndexFormatError,
    KnowledgeBaseError,
    load_knowledge_base,
    read_flat_index,
)
from medquery_api.knowledge_base.store import _load

DATA = config.data_dir()

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)


@pytest.fixture(autouse=True)
def fresh_cache():
    """Each test starts with no knowledge base cached."""
    _load.cache_clear()
    yield
    _load.cache_clear()


# --- Building a tiny data folder ---


def index_bytes(
    vectors: np.ndarray,
    *,
    type_code: bytes = b"IxFI",
    metric: int = 0,
    count: int | None = None,
    floats: int | None = None,
) -> bytes:
    """An index file in the layout faiss.write_index uses for IndexFlatIP."""
    rows, dimension = vectors.shape
    count = rows if count is None else count
    floats = rows * dimension if floats is None else floats
    header = type_code + struct.pack("<iqqqbi", dimension, count, 2**20, 2**20, 1, metric)
    return header + struct.pack("<Q", floats) + vectors.astype("<f4").tobytes()


def unit_rows(count: int, dimension: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).standard_normal((count, dimension))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


ARTICLES = {"asthma.txt": ["Asthma one.", "Asthma two.", "Asthma three."], "hives.txt": ["Hives."]}


def make_data_dir(folder: Path) -> dict[str, np.ndarray]:
    """4 chunks in 2 articles, and an index plus metadata.json for each of the four models."""
    records = []
    for source_file, texts in sorted(ARTICLES.items()):
        article = [
            {"chunk_id": i, "source_file": source_file, "chunk_length": len(t), "text": t}
            for i, t in enumerate(texts, start=1)
        ]
        (folder / "chunks").mkdir(parents=True, exist_ok=True)
        (folder / "chunks" / source_file.replace(".txt", ".json")).write_text(
            json.dumps(article), encoding="utf-8"
        )
        records.extend(article)
    vectors = {}
    for seed, (name, model) in enumerate(MODELS.items()):
        model_dir = folder / "faiss" / name
        model_dir.mkdir(parents=True)
        vectors[name] = unit_rows(len(records), model.dimension, seed)
        (model_dir / "faiss_index.bin").write_bytes(index_bytes(vectors[name]))
        (model_dir / "metadata.json").write_text(json.dumps(records), encoding="utf-8")
    return vectors


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


# --- Reading index files ---


def test_a_valid_index_file_is_read_exactly(tmp_path):
    vectors = unit_rows(3, 8, seed=1)
    path = tmp_path / "faiss_index.bin"
    path.write_bytes(index_bytes(vectors))

    index = read_flat_index(path)

    assert (index.count, index.dimension) == (3, 8)
    assert index.vectors.dtype == np.float32
    assert index.vectors.tobytes() == vectors.tobytes()
    assert not index.vectors.flags.writeable


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: index_bytes(v, type_code=b"IxF2"), "is not supported"),
        (lambda v: index_bytes(v, metric=1), "is not inner product"),
        (lambda v: index_bytes(v)[:-4], "the header implies exactly"),
        (lambda v: index_bytes(v) + b"\0", "the header implies exactly"),
        (lambda v: index_bytes(v, floats=7), "need 24"),
        (lambda v: index_bytes(v)[:20], "too short for an index header"),
        (lambda v: b"PK\x03\x04" + index_bytes(v)[4:], "is not supported"),
    ],
    ids=[
        "L2 index",
        "other metric",
        "cut short",
        "extra bytes",
        "inconsistent header",
        "header only",
        "not an index",
    ],
)
def test_unsupported_or_malformed_index_files_are_rejected(tmp_path, change, message):
    path = tmp_path / "faiss_index.bin"
    path.write_bytes(change(unit_rows(3, 8, seed=1)))

    with pytest.raises(IndexFormatError, match=message):
        read_flat_index(path)


def test_the_file_size_is_checked_before_reading_vectors(tmp_path):
    # A header claiming 10**12 vectors must be rejected from its size, not by trying to read
    # (and allocate) terabytes.
    path = tmp_path / "faiss_index.bin"
    path.write_bytes(index_bytes(unit_rows(3, 8, seed=1), count=10**12, floats=8 * 10**12))

    with pytest.raises(IndexFormatError, match="the header implies exactly"):
        read_flat_index(path)


# --- Loading ---


def test_a_valid_data_folder_loads_and_lines_up(tmp_path):
    vectors = make_data_dir(tmp_path)

    kb = load_knowledge_base(tmp_path)

    assert [c.text for c in kb.chunks] == ["Asthma one.", "Asthma two.", "Asthma three.", "Hives."]
    assert kb.articles == ["asthma.txt", "hives.txt"]
    assert kb.find_row("asthma.txt", 3) == 2
    assert kb.chunk(3).source_file == "hives.txt"
    for name in MODELS:
        assert kb.indexes[name].index.vectors.tobytes() == vectors[name].tobytes()
    with pytest.raises(KeyError, match="no chunk 9 in asthma.txt"):
        kb.find_row("asthma.txt", 9)


def damage_bge_index(folder: Path) -> None:
    path = folder / "faiss" / "BGE" / "faiss_index.bin"
    path.write_bytes(index_bytes(unit_rows(4, 768, seed=9)))  # wrong size for BGE (384)


def drop_a_bge_row(folder: Path) -> None:
    path = folder / "faiss" / "BGE" / "faiss_index.bin"
    path.write_bytes(index_bytes(unit_rows(3, 384, seed=9)))


def reorder_metadata(folder: Path) -> None:
    path = folder / "faiss" / "E5" / "metadata.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(records[::-1]), encoding="utf-8")


def change_metadata_text(folder: Path) -> None:
    path = folder / "faiss" / "MPNet" / "metadata.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records[1]["text"] = "Something else."
    path.write_text(json.dumps(records), encoding="utf-8")


def remove_a_model(folder: Path) -> None:
    for file in (folder / "faiss" / "MiniLM").iterdir():
        file.unlink()
    (folder / "faiss" / "MiniLM").rmdir()


def add_an_unknown_model(folder: Path) -> None:
    (folder / "faiss" / "GPT").mkdir()


def change_a_chunk_record(folder: Path) -> None:
    path = folder / "chunks" / "hives.json"
    path.write_text(json.dumps([{"text": "Hives."}]), encoding="utf-8")


@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        (damage_bge_index, KnowledgeBaseError, "BGE index has 4 vectors of size 768"),
        (drop_a_bge_row, KnowledgeBaseError, "BGE index has 3 vectors"),
        (reorder_metadata, KnowledgeBaseError, "E5 metadata.json does not match .* row 0"),
        (change_metadata_text, KnowledgeBaseError, "MPNet metadata.json does not match .* row 1"),
        (remove_a_model, KnowledgeBaseError, "exactly the model folders"),
        (add_an_unknown_model, KnowledgeBaseError, "exactly the model folders"),
        (change_a_chunk_record, KnowledgeBaseError, "unexpected chunk record format"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_folders_that_do_not_line_up_are_rejected(tmp_path, damage, error, message):
    make_data_dir(tmp_path)
    damage(tmp_path)

    with pytest.raises(error, match=message):
        load_knowledge_base(tmp_path)


def test_a_missing_data_folder_is_reported(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="research data not found"):
        load_knowledge_base(tmp_path / "does-not-exist")


def test_files_outside_the_model_folders_are_ignored(tmp_path):
    # Like the archive's leftover faiss/faiss_index.bin and embeddings/*.json, which no
    # notebook wrote: they are never read, so even garbage there does not matter.
    make_data_dir(tmp_path)
    (tmp_path / "faiss" / "faiss_index.bin").write_bytes(b"not an index")
    (tmp_path / "faiss" / "metadata.json").write_text("not json", encoding="utf-8")
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "embeddings" / "asthma.json").write_text("not json", encoding="utf-8")

    assert len(load_knowledge_base(tmp_path).chunks) == 4


def test_loading_and_searching_leave_the_folder_unchanged(tmp_path):
    make_data_dir(tmp_path)
    before = snapshot(tmp_path)

    kb = load_knowledge_base(tmp_path)
    kb.search("BGE", kb.indexes["BGE"].index.vectors[0], top_k=4)

    assert snapshot(tmp_path) == before


def test_each_resolved_folder_is_loaded_once_and_folders_are_kept_apart(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    make_data_dir(first)
    make_data_dir(second)

    kb = load_knowledge_base(first)

    assert load_knowledge_base(first) is kb
    assert load_knowledge_base(tmp_path / "second" / ".." / "first") is kb
    assert load_knowledge_base(second) is not kb
    assert load_knowledge_base(second).data_dir == second.resolve()


# --- Search ---


def searchable(folder: Path, vectors: np.ndarray, model: str = "MiniLM"):
    """A tiny knowledge base whose `model` index holds the given 4 vectors."""
    make_data_dir(folder)
    path = folder / "faiss" / model / "faiss_index.bin"
    path.write_bytes(index_bytes(vectors))
    return load_knowledge_base(folder)


def axis(dimension: int, *weights: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    for position, weight in weights:
        vector[position] = weight
    return vector / np.linalg.norm(vector)


def test_search_returns_the_highest_scores_first_with_their_chunks(tmp_path):
    vectors = np.stack(
        [axis(384, (0, 1)), axis(384, (1, 1)), axis(384, (0, 1), (1, 1)), axis(384, (2, 1))]
    )
    kb = searchable(tmp_path, vectors)

    hits = kb.search("MiniLM", axis(384, (0, 0.8), (1, 0.6)), top_k=4)

    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert [h.row for h in hits] == [2, 0, 1, 3]
    expected = [0.98994946, 0.8, 0.6, 0.0]
    assert [h.score for h in hits] == pytest.approx(expected, abs=1e-6)
    assert hits[0].chunk.text == "Asthma three."


def test_exactly_equal_scores_are_returned_in_row_order(tmp_path):
    same = axis(384, (5, 1))
    vectors = np.stack([axis(384, (1, 1)), same, axis(384, (2, 1)), same])
    kb = searchable(tmp_path, vectors)

    hits = kb.search("MiniLM", same, top_k=3)

    assert [h.row for h in hits] == [1, 3, 0]
    assert hits[0].score == hits[1].score


@pytest.mark.parametrize("top_k", [0, 5, -1, 2.0, True, "3"])
def test_top_k_must_be_a_whole_number_within_the_index(tmp_path, top_k):
    kb = searchable(tmp_path, unit_rows(4, 384, seed=3))

    with pytest.raises(ValueError, match="top_k must be a whole number from 1 to 4"):
        kb.search("MiniLM", unit_rows(1, 384, seed=4)[0], top_k=top_k)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (np.ones(768) / np.sqrt(768), "must have 384 numbers"),
        (np.full(384, np.nan), "NaN or infinite"),
        (np.ones(384), "length differs from 1"),
        (np.zeros(384), "length differs from 1"),
    ],
    ids=["wrong size", "NaN", "not normalised", "zero vector"],
)
def test_query_vectors_must_be_finite_unit_vectors_of_the_right_size(tmp_path, query, message):
    kb = searchable(tmp_path, unit_rows(4, 384, seed=3))

    with pytest.raises(ValueError, match=message):
        kb.search("MiniLM", query)


def test_a_query_just_within_the_length_tolerance_is_accepted_unchanged(tmp_path):
    kb = searchable(tmp_path, unit_rows(4, 384, seed=3))
    query = unit_rows(1, 384, seed=4)[0].astype(np.float64) * (1 + 0.9e-4)

    hits = kb.search("MiniLM", query, top_k=1)
    expected = float(kb.indexes["MiniLM"].index.vectors[hits[0].row] @ query.astype(np.float32))

    assert hits[0].score == pytest.approx(expected, rel=1e-6)  # not rescaled to length 1


def test_an_unknown_model_is_rejected(tmp_path):
    kb = searchable(tmp_path, unit_rows(4, 384, seed=3))

    with pytest.raises(ValueError, match="Unknown embedding model"):
        kb.search("GPT", unit_rows(1, 384, seed=4)[0])


# --- The research archive (read only) ---


@pytest.fixture(scope="module")
def archive():
    _load.cache_clear()
    started = time.perf_counter()
    kb = load_knowledge_base(DATA)
    seconds = time.perf_counter() - started
    return kb, seconds


@needs_data
def test_the_archive_loads_with_every_check_passing(archive):
    kb, seconds = archive
    print(f"measured: loaded {len(kb.chunks)} chunks, 4 indexes, in {seconds:.2f} s")

    assert len(kb.chunks) == 3796
    assert len(kb.articles) == 698
    assert {name: index.index.vectors.shape for name, index in kb.indexes.items()} == {
        "MiniLM": (3796, 384),
        "MPNet": (3796, 768),
        "E5": (3796, 768),
        "BGE": (3796, 384),
    }
    for name, index in kb.indexes.items():
        assert index.index.path == (DATA / "faiss" / name / "faiss_index.bin").resolve()


@needs_data
@pytest.mark.parametrize("name", MODELS)
def test_archive_index_vectors_are_identical_to_the_archived_embeddings(archive, name):
    kb, _ = archive
    records, embeddings = read_archived_embeddings(DATA, name)

    assert [chunk.as_record() for chunk in kb.chunks] == records
    assert kb.indexes[name].index.vectors.tobytes() == embeddings.tobytes()


@needs_data
@pytest.mark.parametrize("name", MODELS)
def test_every_archived_vector_finds_its_own_chunk_first(archive, name):
    kb, _ = archive
    vectors = kb.indexes[name].index.vectors
    texts = [chunk.text for chunk in kb.chunks]
    wrong, score_gap = [], 0.0
    for row, vector in enumerate(vectors):
        (hit,) = kb.search(name, vector, top_k=1)
        if hit.row != row and texts[hit.row] != texts[row]:
            wrong.append((row, hit.row))
        squared_length = float(np.dot(vector.astype(np.float64), vector.astype(np.float64)))
        score_gap = max(score_gap, abs(hit.score - squared_length))
    print(
        f"measured: {name} {len(vectors) - len(wrong)} of {len(vectors)} find their own chunk "
        f"(or an identical text) first; largest |score - squared length| {score_gap:.3e}"
    )

    assert wrong == []
    assert score_gap <= 1e-5


@needs_data
def test_every_archived_chunk_can_be_looked_up(archive):
    kb, _ = archive

    for row, chunk in enumerate(kb.chunks):
        assert kb.find_row(chunk.source_file, chunk.chunk_id) == row
