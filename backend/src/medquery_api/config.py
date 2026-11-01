"""Where the backend finds the research data."""

import os
from pathlib import Path

# backend/src/medquery_api/config.py -> the repository root is three folders up.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "Healthcare_RAG" / "data"


def data_dir() -> Path:
    """The research data folder; MEDQUERY_DATA_DIR overrides the default location."""
    override = os.environ.get("MEDQUERY_DATA_DIR")
    return Path(override) if override else DEFAULT_DATA_DIR
