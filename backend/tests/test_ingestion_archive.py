"""Golden tests: the new cleaner and chunker must reproduce the research artefacts exactly."""

import json
from pathlib import Path

import pytest

from medquery.config import get_settings
from medquery.ingestion import (
    BASELINE_CHUNKING,
    BASELINE_CLEANING,
    Chunker,
    clean_text,
    load_text_document,
)

pytestmark = pytest.mark.archive


@pytest.fixture(scope="module")
def data_dir() -> Path:
    path = get_settings().archive_dir / "data"
    if not (path / "raw").is_dir():
        pytest.skip("Research archive not available on this machine")
    return path


def test_cleaner_reproduces_all_archived_cleaned_articles(data_dir: Path) -> None:
    raw_files = sorted((data_dir / "raw").glob("*.txt"))
    assert len(raw_files) == 698
    mismatches = [
        p.name
        for p in raw_files
        if clean_text(load_text_document(p).text, BASELINE_CLEANING)
        != (data_dir / "cleaned" / p.name).read_text(encoding="utf-8")
    ]
    assert mismatches == []


def test_chunker_reproduces_all_archived_chunks(data_dir: Path) -> None:
    chunker = Chunker(BASELINE_CHUNKING)
    total = 0
    mismatches = []
    for cleaned_path in sorted((data_dir / "cleaned").glob("*.txt")):
        document = load_text_document(cleaned_path)
        chunks = chunker.chunk(document)
        expected = json.loads(
            (data_dir / "chunks" / f"{cleaned_path.stem}.json").read_text(encoding="utf-8")
        )
        if [c.to_legacy_record() for c in chunks] != expected:
            mismatches.append(cleaned_path.name)
        for chunk in chunks:
            start = chunk.start_index
            assert document.text[start : start + chunk.chunk_length] == chunk.text
        total += len(chunks)
    assert mismatches == []
    assert total == 3796
