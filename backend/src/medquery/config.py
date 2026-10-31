"""Application settings, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/medquery/config.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ArchiveWriteError(RuntimeError):
    """Raised when code attempts to write inside the read-only research archive."""


class Settings(BaseSettings):
    """Typed configuration. Every field can be overridden with a `MEDQUERY_` env var."""

    model_config = SettingsConfigDict(
        env_prefix="MEDQUERY_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    archive_dir: Path = PROJECT_ROOT / "Healthcare_RAG"
    docs_dir: Path = PROJECT_ROOT / "docs"
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    log_level: str = "INFO"

    @field_validator("archive_dir", "docs_dir", "experiments_dir", mode="after")
    @classmethod
    def _resolve_against_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("log_level", mode="after")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log level: {value!r}")
        return level

    @property
    def archive_results_dir(self) -> Path:
        return self.archive_dir / "results"

    @property
    def baseline_manifest_path(self) -> Path:
        return self.docs_dir / "baseline" / "MANIFEST.sha256"

    def ensure_writable(self, path: Path) -> Path:
        """Return `path` resolved, refusing any location inside the read-only archive."""
        resolved = path.resolve()
        if resolved.is_relative_to(self.archive_dir.resolve()):
            raise ArchiveWriteError(f"Refusing to write inside the research archive: {resolved}")
        return resolved


def get_settings() -> Settings:
    return Settings()
