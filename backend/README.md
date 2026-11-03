# MedQuery AI: backend

A small [FastAPI](https://fastapi.tiangolo.com/) service with four endpoints:

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `{"status": "ok"}` when the service is running |
| `GET /api/dataset/summary` | Article, chunk and embedding-index counts, counted live from `Healthcare_RAG/data` |
| `POST /api/retrieve` | The research chunks most similar to a question, with scores (see [Retrieval](#retrieval)) |
| `GET /api/llm` | The configured LLM provider and model (`{"provider": "mock", "model": "mock"}`); no credentials or other settings |
| `POST /api/answer` | An answer generated from the retrieved chunks, with their sources (see [Grounded answers](#grounded-answers)) |

The backend only **reads** the research archive; it never writes to it.

## Requirements

[uv](https://docs.astral.sh/uv/), which installs Python 3.11 and the dependencies.

## Commands

Run these from this `backend/` folder.

| Command | What it does |
|---|---|
| `uv sync` | Create `.venv` and install the exact versions in `uv.lock` (first time only) |
| `uv run uvicorn medquery_api.main:app --reload --host 127.0.0.1 --port 8000` | Start the API on this computer only |
| `uv run pytest` | Run the tests (without the opt-in real-model tests) |
| `uv run pytest -m models -rP` | Run only the real-model tests (needs the models in the local cache) |
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

## Article processing

`medquery_api.processing` cleans and chunks article text the way the research notebooks did,
in plain Python with no extra libraries:

| Function | Does what |
|---|---|
| `clean_text(text)` | Notebook 02's cleaning |
| `split_text(text, chunk_size=1000, chunk_overlap=200, separators=...)` | Notebook 03's LangChain `RecursiveCharacterTextSplitter`, reimplemented |
| `chunk_article(text, source_file)` | `split_text`, numbered into records shaped like the archive's chunk files |

`tests/test_processing.py` has unit tests on small texts and compares the output with the
archive (read only): all 698 cleaned articles byte for byte, and all 3,796 chunk records
(order, id, file name, length and text). The archive comparisons are skipped, with the reason
shown, when `Healthcare_RAG/data` is absent; `-rP` prints the measured counts.

## Embedding models

`medquery_api.embeddings` turns text into vectors with the research's four models, using
[sentence-transformers](https://www.sbert.net/) on the CPU:

```python
from medquery_api.embeddings import open_model

with open_model("E5") as embedder:  # one model in memory at a time
    vectors = embedder.embed_documents(["chunk text", ...])  # adds "passage: " for E5
    question = embedder.embed_query("What are the symptoms of asthma?")  # "query: " for E5
```

| Model | Hugging Face model | Vector size | Prefixes (document / query) |
|---|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | none |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | 768 | none |
| E5 | `intfloat/multilingual-e5-base` | 768 | `passage: ` / `query: ` |
| BGE | `BAAI/bge-small-en-v1.5` | 384 | none |

- **Pinned versions.** Each model is pinned to one revision (see `embeddings/models.py`).
  These are inferred to be compatible with the research, not recorded as the ones it used.
- **No surprise downloads.** Models load only from the local Hugging Face cache. Download them
  once, explicitly (about 1.7 GB for all four):
  `uv run python -m medquery_api.embeddings download` (or name one model).
- **CPU by default.** `MEDQUERY_EMBEDDING_DEVICE` chooses another device; the PyTorch build
  installed from PyPI on Windows supports only the CPU.
- **Light imports.** PyTorch and sentence-transformers are imported only when a model is
  opened, so the API and the other tests start quickly. Installing them adds about 790 MB to
  `.venv`.
- **Checked outputs.** Every result is checked: one vector per text, in order, of the right
  size, finite, and of length 1 (within 0.0001).

To compare all 3,796 chunks with the archived embeddings (it reads the archive only, and takes
several minutes per model):

```
uv run python -m medquery_api.embeddings validate [--model E5] [--save-dir PATH]
```

A model passes when every chunk's vector has cosine similarity of at least 0.999 with the
archived one and is closest to its own archived vector. `--save-dir` also saves the vectors
and the report, but never inside `Healthcare_RAG/`.

`tests/test_embeddings.py` has fast unit tests with a stand-in model, plus opt-in tests
(`-m models`) that check a sample of 33 chunks per model against the archive.

## Knowledge base

`medquery_api.knowledge_base` serves the research's own chunks and FAISS indexes, read only,
with exact vector search. It needs no extra libraries: the index files are read with NumPy.

```python
from medquery_api.knowledge_base import load_knowledge_base

kb = load_knowledge_base()  # loaded once per data folder, then kept in memory
hits = kb.search("BGE", query_vector, top_k=5)  # a unit-length vector of the model's size
for hit in hits:
    print(hit.rank, round(hit.score, 4), hit.chunk.source_file, hit.chunk.chunk_id)
```

- **What is loaded:** the 3,796 chunks (`data/chunks`) and, for each model,
  `data/faiss/<model>/faiss_index.bin` with its `metadata.json`. Files elsewhere, such as the
  archive's leftover `faiss/faiss_index.bin`, are never read.
- **Checked on loading:** each index file's header and exact size (before any vectors are
  read); only the research's index type (flat, inner product) is accepted; each index has one
  vector per chunk, of the model's size; each `metadata.json` matches the chunks row for row.
  Anything else raises an error that says what is wrong.
- **Search:** scores every chunk (inner product, which is cosine similarity for unit
  vectors) and returns the best `top_k`, highest score first; exactly equal scores come in row
  order. Query vectors must have the model's size, finite values and length 1 within 0.0001;
  they are rejected, not silently normalised.
- **Cost:** loading takes about 0.5 s and about 60 MiB of memory; one search takes under a
  millisecond.

`tests/test_knowledge_base.py` has unit tests on a tiny generated data folder and archive
tests: every index is bit-identical to the archived embeddings, and searching with each of
the 4 × 3,796 archived vectors returns its own chunk first.

## Retrieval

`POST /api/retrieve` embeds a question with one of the four models (adding `query: ` for E5)
and searches that model's research index exactly, as notebook 06 did.

```
request:  {"question": "What are the symptoms of asthma?", "model": "BGE", "topK": 5}
response: {"question", "model", "modelId", "revision", "topK",
           "results": [{"rank", "score", "sourceFile", "chunkId", "chunkLength", "text"}],
           "timings": {"modelLoadMs", "embeddingMs", "searchMs"}}
```

- **Inputs:** `question` is 1 to 1,000 characters after surrounding whitespace is removed;
  `model` is `MiniLM`, `MPNet`, `E5` or `BGE`; `topK` is a whole number from 1 to 20 (default
  5). Anything else is rejected with 422 before any model or data is loaded.
- **Results:** highest score (cosine similarity) first; exactly equal scores in row order.
- **One model in memory:** the last model used stays loaded; asking for another releases it
  and loads the new one. `modelLoadMs` is that load time, and 0 when the model was already
  loaded. Requests use the model one at a time, so a model is never swapped mid-request.
- **Errors (503):** a model missing from the local cache (the message gives the download
  command), or research data that is missing or fails its checks (details go to the server
  log, not the response). Question text is not logged.
- **Speed on this computer's CPU:** after loading, a request takes about 10-30 ms; loading
  takes 0.2-3 s per model, and the very first request about 6 s, mostly loading PyTorch.

`tests/test_retrieval.py` has fast API tests with a stand-in model, and opt-in comparisons
with the research's recorded results (`-m models`).

## LLM providers

`medquery_api.llm` (PRD 3.1) is one interface for generating answers, with two providers:

| Provider | What it does | Needs |
|---|---|---|
| `mock` (default) | Answers by quoting and citing the first document, or says the evidence is not enough; tests can script any result | Nothing: no network, no key, no cost |
| `claude` | Claude through Anthropic's Messages API: each document sent verbatim with citations enabled, the question after them, refusals re-run on Anthropic's recommended fallback model | Anthropic credentials and money per call |

```python
from medquery_api.llm import GenerationRequest, SourceDocument, create_provider

provider = create_provider()  # MEDQUERY_LLM_PROVIDER, default "mock"
result = provider.generate(
    GenerationRequest(
        system="Answer only from the documents.",
        documents=(SourceDocument(title="Asthma (chunk 1)", text=chunk_text),),
        question="What does asthma affect?",
    )
)
result.text, result.parts[0].citations, result.usage, result.refused
```

| Setting | Default | Meaning |
|---|---|---|
| `MEDQUERY_LLM_PROVIDER` | `mock` | `mock` or `claude` |
| `MEDQUERY_LLM_MODEL` | `claude-sonnet-5-5` | Claude model |
| `MEDQUERY_LLM_EFFORT` | `low` | `low`, `medium`, `high`, `xhigh` or `max` |

Credentials are never read or stored by this code: the `anthropic` SDK finds them itself
(`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile). Keep them out of
the repository (`.env` files are already ignored). Every API failure becomes an
`LLMUnavailableError` whose message is safe to show: it contains no credentials or request
text. The SDK is imported only when Claude is actually called.

`tests/test_llm.py` runs without any credentials. Its one real-API test is opt-in and costs
money: `uv run pytest -m llm -rP`.

`medquery_api.llm.context` (PRD 3.2) turns a retrieval (`POST /api/retrieve`'s result) into a
request: the research's answering rules as instructions, each retrieved chunk verbatim as a
document titled "\<article title\> (chunk N of M)" in rank order, then the question.
`build_context(...).source_for(i)` says which chunk document *i* came from, so citations can be
traced back; `render_plain_text` gives the same sources labelled [S1]...[Sn] for providers
without built-in citations.

## Grounded answers

`POST /api/answer` (PRD 3.3) takes the same request as `/api/retrieve` (`question`, `model`,
`topK`), retrieves the chunks, builds the context (`llm.context`) and asks the configured LLM
provider (`MEDQUERY_LLM_PROVIDER`, default `mock`) to answer from them. The response gives:

- `parts`: the answer text, each part with its **valid** citations, labelled with the source
  they cite (`S1`...). `citationStatus: "checked"` (PRD 3.4): `llm.citations` keeps a citation
  only if it names a retrieved source, its character range lies inside that chunk, and the
  chunk's text over that range equals the quoted text exactly.
- `grounding` (PRD 3.6): `refused`; `not_enough_evidence` when the whole answer is exactly the
  sentence the instructions require; otherwise `not_grounded` (no sentence has a valid
  citation, so the answer is never presented as grounded), `grounded` (every sentence has one)
  or `partly_grounded`. `sentences` gives each sentence's `start` and `end` (code points in the
  answer's parts joined, without surrounding whitespace) and whether it is `supported`.
- `droppedCitations`: every citation that failed, as the provider returned it, with the answer
  part it came from and the reason (`unknown_document`, `invalid_range`, `text_mismatch`).
- `citationCheck`: citations returned, valid and dropped by reason; the validity rate; and
  **grounding coverage**, the share of answer sentences with at least one valid citation
  (sentence rule in `llm/citations.py`). These are checks, not proof: a valid citation quotes
  its source exactly, but that does not show the answer is correct.
- `sources`: the retrieved chunks the model saw, verbatim, with label, rank, score, file,
  chunk number and article title.
- `llm` (provider, the model that answered, whether a fallback model ran), `stopReason`,
  `refused`, `refusalCategory`, token `usage`.
- `cost`: an estimate from `pricing.py`'s dated price table (Sonnet 5.5 checked on
  2 November 2026); none for a model without a checked price; $0 for the mock.
- `timings`: model load, embedding, search, generation and total, in milliseconds.

Invalid requests get 422 before anything runs; missing data or models and provider failures
get 503 with safe messages. Question text is never logged.

The research's 10 evaluation questions can be answered for review, next to the archived Llama
answers (mock by default; a paid provider needs `--allow-paid-calls`):

```
uv run python -m medquery_api.answering review [--out answers.json]
```

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
