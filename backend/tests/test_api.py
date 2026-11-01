import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medquery_api import config
from medquery_api.main import app, get_data_dir

client = TestClient(app)


@pytest.fixture
def use_data_dir():
    """Point the API at a given data folder for one test."""

    def _use(folder: Path) -> None:
        app.dependency_overrides[get_data_dir] = lambda: folder

    yield _use
    app.dependency_overrides.clear()


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    """A tiny stand-in for Healthcare_RAG/data: 2 articles, 3 chunks, 1 index."""
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "asthma.json").write_text(json.dumps([{"text": "a"}, {"text": "b"}]), "utf-8")
    (chunks / "hives.json").write_text(json.dumps([{"text": "c"}]), "utf-8")
    (tmp_path / "faiss" / "MPNet").mkdir(parents=True)
    (tmp_path / "faiss" / "MPNet" / "faiss_index.bin").write_bytes(b"index")
    (tmp_path / "faiss" / "NoIndexYet").mkdir()  # a folder without an index is not counted
    (tmp_path / "faiss" / "faiss_index.bin").write_bytes(b"loose file")  # not a model folder
    return tmp_path


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dataset_summary_counts_a_data_folder_without_changing_it(fake_data_dir, use_data_dir):
    use_data_dir(fake_data_dir)
    before = snapshot(fake_data_dir)

    response = client.get("/api/dataset/summary")

    assert response.status_code == 200
    assert response.json() == {
        "articles": 2,
        "chunks": 3,
        "embeddingIndexes": ["MPNet"],
        "source": "MEDQUERY_DATA_DIR",
    }
    assert snapshot(fake_data_dir) == before


def test_dataset_summary_reports_a_missing_folder_instead_of_guessing(tmp_path, use_data_dir):
    use_data_dir(tmp_path / "does-not-exist")

    response = client.get("/api/dataset/summary")

    assert response.status_code == 503
    assert "Research data folder not found" in response.json()["detail"]


@pytest.mark.skipif(
    not (config.DEFAULT_DATA_DIR / "chunks").is_dir(),
    reason="Healthcare_RAG/data not found; the real-archive check runs only where it exists",
)
def test_dataset_summary_matches_the_real_research_archive(use_data_dir):
    use_data_dir(config.DEFAULT_DATA_DIR)

    response = client.get("/api/dataset/summary")

    assert response.status_code == 200
    assert response.json() == {
        "articles": 698,
        "chunks": 3796,
        "embeddingIndexes": ["BGE", "E5", "MPNet", "MiniLM"],
        "source": "Healthcare_RAG/data",
    }
