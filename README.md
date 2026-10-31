# MedQuery AI: RAG Intelligence Platform

A platform for building, evaluating, and comparing Retrieval-Augmented Generation
(RAG) systems. The first knowledge base is a corpus of 698 NHS health articles taken
from earlier dissertation research. The architecture is designed to support other
domains as well.

> **Status: Part 1 (Foundation and Research Migration), milestones 1.1–1.4 complete.**
> Cleaning, chunking, embedding and token analysis are implemented; no vector store,
> retrieval, generation, API, or UI code exists yet.

## Repository layout

```
medquery-ai/
├── Healthcare_RAG/          # Original research archive: READ-ONLY (see below)
├── backend/                 # Python package `medquery` (uv project)
│   ├── src/medquery/
│   │   ├── config.py        # Typed settings (MEDQUERY_* env vars)
│   │   ├── baseline/        # Archive manifest, original-results loader, archived artefacts
│   │   ├── ingestion/       # Text loading, cleaning, configurable chunking
│   │   └── embeddings/      # Model specs, compatibility, embedders, token analysis
│   ├── scripts/             # build_manifest, inspect_legacy_artifacts, analyse_embedding_models
│   └── tests/
├── docs/
│   ├── baseline/            # Provenance, checksum manifest, legacy-artefact report
│   ├── ingestion/           # Cleaning and chunking strategy + limitations
│   ├── embeddings/          # Embedding models, token/reproduction reports, hardware
│   └── LIMITATIONS.md       # Tracked limitations register
└── .env.example
```

## Research baseline

`Healthcare_RAG/` contains the original Colab notebooks, data and results. It is
treated as read-only, and a SHA-256 manifest of all 2,837 files is used to detect any
change. Original results are never overwritten or mixed with new runs. See
[docs/baseline/PROVENANCE.md](docs/baseline/PROVENANCE.md) for the pipeline lineage,
original metrics, evaluation caveats and legacy-file findings, and
[docs/ingestion/CHUNKING.md](docs/ingestion/CHUNKING.md) for the cleaning and
chunking strategy. The new cleaner and chunker reproduce the archived cleaned text
(698 files) and chunks (3,796) exactly.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.11 automatically.

```powershell
cd backend
uv sync                      # create .venv and install dependencies
uv run pytest                # run tests
uv run ruff check .          # lint
uv run ruff format --check . # formatting
```

Tests marked `archive` read the local `Healthcare_RAG/` folder and are skipped when it
is absent (for example in CI, where the data folder is not committed).

Tests never download model weights: Hugging Face offline mode is forced for the whole
test session and unit tests use stand-in models. Real-model tests (marker `models`) are
opt-in and load from the local cache only:

```powershell
uv run python scripts/analyse_embedding_models.py --device cpu   # downloads ~1.8 GB once
$env:MEDQUERY_RUN_MODEL_TESTS = "1"; uv run pytest -m models
```

Verify the archive is unchanged:

```powershell
uv run python scripts/build_manifest.py
```

## Configuration

Copy `.env.example` to `.env` (git-ignored). All variables are optional:

| Variable | Default | Purpose |
|---|---|---|
| `MEDQUERY_ARCHIVE_DIR` | `Healthcare_RAG` | Read-only research archive |
| `MEDQUERY_DOCS_DIR` | `docs` | Documentation and baseline manifest |
| `MEDQUERY_EXPERIMENTS_DIR` | `experiments` | Output folder for new experiment runs |
| `MEDQUERY_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `MEDQUERY_EMBEDDING_DEVICE` | `auto` | `auto` / `cpu` / `cuda` (use `cpu` on the dev machine for now: see EMBEDDINGS.md) |
| `MEDQUERY_RUN_MODEL_TESTS` | unset | `1` enables real-model tests (cache only) |

Relative paths resolve against the repository root. No secrets are needed at this
stage.

## Roadmap

| Part | Scope | Status |
|---|---|---|
| 1 | Foundation and research migration | 1.1–1.4 done; 1.5–1.8 pending |
| 2 | Core RAG and LLM engine (pgvector, local LLM) | Not started |
| 3 | Web application | Not started |
| 4 | Benchmarking and evaluation | Not started |
| 5 | Advanced RAG and production engineering | Not started |
| 6 | Deployment and portfolio | Not started |

Part 1 milestones: 1.1 scaffold · 1.2 baseline freeze · 1.3 cleaner/chunker ·
1.4 embedding registry · 1.5 vector store · 1.6 retriever + metrics ·
1.7 baseline reproduction run · 1.8 prompts, LLM interface (fake backend), pipeline,
FastAPI skeleton.

## Data source

NHS website content (nhs.uk), used under the Open Government Licence v3.0.
