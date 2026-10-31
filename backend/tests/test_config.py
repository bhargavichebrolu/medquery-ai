from pathlib import Path

import pytest

from medquery.config import PROJECT_ROOT, ArchiveWriteError, Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("ARCHIVE_DIR", "DOCS_DIR", "EXPERIMENTS_DIR", "LOG_LEVEL"):
        monkeypatch.delenv(f"MEDQUERY_{name}", raising=False)
    return monkeypatch


def test_defaults_point_inside_project_root(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    assert settings.archive_dir == PROJECT_ROOT / "Healthcare_RAG"
    assert settings.archive_results_dir == PROJECT_ROOT / "Healthcare_RAG" / "results"
    assert settings.baseline_manifest_path == PROJECT_ROOT / "docs" / "baseline" / "MANIFEST.sha256"


def test_relative_env_paths_resolve_against_project_root(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("MEDQUERY_EXPERIMENTS_DIR", "some/runs")
    settings = Settings(_env_file=None)
    assert settings.experiments_dir == (PROJECT_ROOT / "some" / "runs").resolve()


def test_absolute_env_paths_are_kept(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("MEDQUERY_ARCHIVE_DIR", str(tmp_path))
    assert Settings(_env_file=None).archive_dir == tmp_path


def test_log_level_is_normalised_and_validated(clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValueError):
        Settings(_env_file=None, log_level="verbose")


def test_writes_inside_archive_are_refused(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, archive_dir=tmp_path / "archive")
    with pytest.raises(ArchiveWriteError):
        settings.ensure_writable(tmp_path / "archive" / "results" / "new.csv")
    assert settings.ensure_writable(tmp_path / "elsewhere.csv") == (tmp_path / "elsewhere.csv")
