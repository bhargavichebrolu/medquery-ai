"""The historical baseline configurations are pinned to the notebook values."""

from medquery.ingestion import (
    BASELINE_CHUNKING,
    BASELINE_CLEANING,
    ChunkingConfig,
    CleaningConfig,
)


def test_baseline_cleaning_matches_notebook_02() -> None:
    assert BASELINE_CLEANING.model_dump() == {
        "unicode_normalization": "NFC",
        "remove_control_characters": True,
        "collapse_whitespace": True,
        "deduplicate_lines": True,
        "line_separator": "\n\n",
    }


def test_baseline_chunking_matches_notebook_03() -> None:
    assert BASELINE_CHUNKING.model_dump() == {
        "strategy": "recursive_character",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ("\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""),
    }


def test_experimental_configs_are_separate_objects() -> None:
    experiment = BASELINE_CHUNKING.model_copy(update={"chunk_size": 500, "chunk_overlap": 50})
    assert experiment != BASELINE_CHUNKING
    assert BASELINE_CHUNKING.chunk_size == 1000
    assert isinstance(experiment, ChunkingConfig)
    assert isinstance(BASELINE_CLEANING, CleaningConfig)
