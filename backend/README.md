# MedQuery AI: backend

A small [FastAPI](https://fastapi.tiangolo.com/) service. For now it has two endpoints:

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `{"status": "ok"}` when the service is running |
| `GET /api/dataset/summary` | Article, chunk and embedding-index counts, counted live from `Healthcare_RAG/data` |

The backend only **reads** the research archive; it never writes to it.

## Requirements

[uv](https://docs.astral.sh/uv/), which installs Python 3.11 and the dependencies.

## Commands

Run these from this `backend/` folder.

| Command | What it does |
|---|---|
| `uv sync` | Create `.venv` and install the exact versions in `uv.lock` (first time only) |
| `uv run uvicorn medquery_api.main:app --reload --host 127.0.0.1 --port 8000` | Start the API on this computer only |
| `uv run pytest` | Run the tests |
| `uv run ruff check .` | Lint |

With the server running, http://127.0.0.1:8000/docs shows interactive API documentation.

The frontend's development server forwards every `/api/...` request here, so run both
(see `../frontend/README.md`).

## Why `httpx2` (not `httpx`)

The tests use FastAPI's `TestClient`, which is Starlette's. Starlette 1.10 (installed with
FastAPI 0.162) builds its test client on `httpx2` and names it as the required package.
Plain `httpx` still works, but Starlette then emits a `StarletteDeprecationWarning` on every
run, and the tests fail when warnings are treated as errors (`pytest -W error`). `httpx2` is
by the original `httpx` author and maintained by Pydantic. It is a development-only
dependency: the running API does not use it.

## Research pipeline checks

`tests/test_research_pipeline.py` checks the research archive against
[`docs/research-pipeline.md`](../docs/research-pipeline.md). It only reads files.

- **Notebook checks** confirm the settings the documentation quotes (cleaning function,
  chunk size and overlap, model names, normalisation, index type, query prefixes) are what
  the notebooks contain.
- **Data checks** confirm each stage's files agree with the previous one: cleaning is
  reproduced byte for byte; chunks, embeddings, FAISS indexes and their metadata line up
  row for row; vectors have length 1 within a stated tolerance. They are skipped, with the
  reason shown, when `Healthcare_RAG/data` is absent.

`uv run pytest tests/test_research_pipeline.py -rP` also prints the value each check
measured. The checks take about 20 seconds; the largest embedding file (98 MB) briefly needs
about 300 MiB of memory, because it is one JSON array that has to be read whole.

These checks show the files are consistent with the notebook code. They do not show which
notebook run produced a file.

## Configuration

`MEDQUERY_DATA_DIR` points the backend at a different data folder. By default it uses
`../Healthcare_RAG/data`. Tests that need the real archive are skipped, with the reason
shown, when that folder is absent.
