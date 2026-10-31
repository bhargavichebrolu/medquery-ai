# Research Baseline: Provenance and Integrity

This document records what the original dissertation research produced, how it was
produced, and how its integrity is protected. It is the reference point for every
new experiment in this repository.

## 1. Archive status

`Healthcare_RAG/` is the original research archive. It is **read-only**: nothing in
the new codebase moves, modifies, or deletes it, and `Settings.ensure_writable()`
refuses any write path inside it.

| Item | Value |
|---|---|
| Files covered by manifest | 2,837 |
| Manifest | [`MANIFEST.sha256`](MANIFEST.sha256) (`sha256sum` format) |
| Manifest created | 2026-10-31 |
| Verify | `cd backend && uv run python scripts/build_manifest.py` |
| Automated check | `tests/test_manifest.py::test_real_archive_matches_committed_manifest` |

The archive's `data/` folder (~470 MB) is git-ignored; notebooks and `results/` are
tracked.

## 2. Pipeline lineage

All notebooks ran on Google Colab against `/content/drive/MyDrive/Healthcare_RAG`.

| Notebook | Produces | Key settings |
|---|---|---|
| `01_Data_Collection` | `data/raw/*.txt` (698) | URLs from `data/urls/nhs_urls.txt`; HTTP→HTTPS, de-duplicated; BeautifulSoup on `<main>`: h1–h3, p, top-level `li`; duplicate text removed; 0.5 s delay; 698/698 succeeded |
| `02_Data_Cleaning` | `data/cleaned/*.txt` (698) | Strip lines, drop empty and duplicate lines, collapse whitespace, join with blank lines |
| `03_Text_Chunking` | `data/chunks/*.json` (698 files, 3,796 chunks) | LangChain `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`, separators `["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]`; mean 645.6 chars, min 16, max 999 |
| `04_Embedding_Generation` | `data/embeddings/<Model>/all_embeddings.json`, `data/embeddings/embedding_dimension_comparison.png` | See §3 |
| `05_Vector_Database_FAISS` | `data/faiss/<Model>/{faiss_index.bin, metadata.json}`, `results/faiss_*` | `IndexFlatIP` (exact; inner product = cosine on normalised vectors); vector *i* ↔ metadata record *i* |
| `06_RAG_Pipeline` | All other `results/*` | Retrieval, prompt, generation, evaluation, Gradio UI |

`06_RAG_Pipeline.ipynb` and `06_RAG_Pipeline(1).ipynb` contain **identical code**;
`(1)` holds additional saved outputs from a later execution and is the more complete
record.

**Chunk identity:** `chunk_id` restarts at 1 for every article, so a chunk is uniquely
identified only by `(source_file, chunk_id)`.

## 3. Embedding models

| Name | Hugging Face ID | Dim | Query / passage prefix | Encode batch |
|---|---|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | none | 32 |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | 768 | none | 16 |
| E5 | `intfloat/multilingual-e5-base` | 768 | `query: ` / `passage: ` | 16 |
| BGE | `BAAI/bge-small-en-v1.5` | 384 | none | 32 |

All embeddings use `normalize_embeddings=True` and are stored as float32. Stored text
never includes the E5 prefix. Model *revisions* (commit hashes) were not recorded.

## 4. Generation models

1. `google/flan-t5-base`: initial CPU-compatible generator, later superseded.
2. `meta-llama/Llama-3.1-8B-Instruct`: 4-bit NF4 (bitsandbytes, double quant, fp16
   compute) on a Colab Tesla T4; greedy decoding; `max_new_tokens=180`; input truncated
   at 4,096 tokens. The final chatbot used **MPNet + Llama, top-k 5**.

## 5. Original results (do not overwrite)

These are parsed directly from `Healthcare_RAG/results/` by
`medquery.baseline.loader`, which tags every value
`provenance="original_dissertation"`.

**Retrieval** (`retrieval_model_comparison.csv`, 10 questions, k = 5, source level):

| Model | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| MPNet | 0.20 | 1.0 | 0.95 | 0.9631 |
| BGE | 0.20 | 1.0 | 0.90 | 0.9262 |
| E5 | 0.20 | 1.0 | 0.90 | 0.9262 |
| MiniLM | 0.18 | 0.9 | 0.80 | 0.8262 |

**FAISS index construction** (`faiss_experimental_summary.csv`, Colab):
MiniLM 0.0049 s · MPNet 0.0054 s · E5 0.0071 s · BGE 0.0029 s (3,796 vectors each).

**Embedding generation time** (notebook 04 output only; no CSV saved; Colab):
MiniLM 420.69 s · MPNet 2,813.16 s · E5 2,852.27 s · BGE 934.92 s.

**RAG latency** (`rag_performance_summary.csv`, MPNet + Llama, Colab T4, 10 questions):

| Stage | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| Retrieval | 0.0195 | 0.0033 | 0.0154 | 0.0252 |
| Generation | 10.6982 | 4.7373 | 3.5401 | 18.9132 |
| Total | 10.7247 | 4.7382 | 3.5620 | 18.9433 |

Timings come from Colab hardware and **must not be compared directly** with timings
measured on other hardware.

## 6. Evaluation methodology and caveats

- **Relevance is defined at source-file level.** Each question names the NHS article(s)
  that should answer it; multiple chunks from one article count once.
- **Precision@5 is capped at 0.20.** Every question has exactly one relevant source and
  precision is divided by k = 5, so 0.20 is a perfect score here, not a weak one.
- **The evaluation set is small:** 10 questions over 8 distinct sources (3 about
  asthma). BGE and E5 produce identical scores. The results are a valid baseline but
  are not sufficient to rank the models with statistical confidence.
- **No Ragas scores exist.** Ragas was installed and `ragas_evaluation_dataset.json`
  was prepared, but no saved file contains faithfulness, answer-relevance or other Ragas
  metrics. Generation quality has not been evaluated with automated metrics.
- **Reference answers are human-written** (notebook 06, step 35), derived from the
  known relevant NHS articles and not from Llama or retrieval output. They are not
  independently verified.
- Model revisions, most library versions, and random seeds were not recorded (see §9).
- The `Index_File` column in `faiss_experimental_summary.csv` contains Colab paths.

## 7. Legacy artefacts

Two artefact sets are **not written by any saved notebook version**. Their provenance
was established by content comparison with
[`scripts/inspect_legacy_artifacts.py`](../../backend/scripts/inspect_legacy_artifacts.py);
the full output is in [`legacy_artifacts_report.json`](legacy_artifacts_report.json).
Method: embeddings from different models occupy unrelated vector spaces, so row-aligned
cosine similarity is ≈ 1.0 only if the same model produced both. **All rows** were
compared, not a sample.

### 7.1 `data/faiss/faiss_index.bin` + `data/faiss/metadata.json` (top level)

| Check | Finding |
|---|---|
| Format | FAISS `IndexFlatIP`, 384 dims, 3,796 vectors, all unit-norm |
| Metadata | 3,796 records; order and text identical to every per-model `metadata.json` |
| vs BGE (3,796 rows) | cosine min 1.000000, max abs difference 2.4e-7 |
| vs MiniLM (3,796 rows) | cosine mean 0.207, min 0.0009 (different model) |
| Byte-identical to `faiss/BGE/`? | No: a separate encoding run, equal up to float rounding |

**Conclusion: produced by `BAAI/bge-small-en-v1.5`**, most likely an earlier BGE-only
run of notebooks 04/05 before outputs were split into per-model folders. *(The "earlier
run" part is an inference; the model identity is established by the comparison.)*

### 7.2 `data/embeddings/*.json` (top level)

| Check | Finding |
|---|---|
| Files | 699: 698 per-article files (one per chunk file) **plus** `all_embeddings.json` (3,796 records) |
| Records | 7,592 = 2 × 3,796; only 3,796 unique `(source_file, chunk_id)` keys, so the per-article files and `all_embeddings.json` duplicate each other |
| Fields | `chunk_id, chunk_length, source_file, text, embedding`; no model-identifier field |
| Dimension | 384, all unit-norm |
| vs BGE (7,592 rows) | cosine min 1.000000 |
| vs MiniLM (7,592 rows) | cosine mean 0.207 (different model) |

**Conclusion: produced by `BAAI/bge-small-en-v1.5`**, in the same earlier BGE-only
layout.

### 7.3 Status

Both sets are **identified (BGE) but superseded** by `data/embeddings/BGE/` and
`data/faiss/BGE/`, which are the artefacts the published baseline was evaluated with.
The new pipeline **does not use them**. They remain in the archive unchanged.

`data/embeddings/embedding_dimension_comparison.png` is *not* legacy: notebook 04,
step 25 writes it.

## 8. Results-file loader coverage

All files are preserved as-is and covered by the manifest. Typed loaders
(`medquery.baseline.loader`) are added incrementally as milestones need them.

| File | Loader |
|---|---|
| `retrieval_model_comparison.csv` | `load_retrieval_summary` |
| `corrected_retrieval_results.csv` | `load_retrieval_per_question` |
| `faiss_experimental_summary.csv` | `load_faiss_builds` |
| `rag_performance_summary.csv` | `load_latency` |
| `final_retrieval_results.csv` | Not yet (duplicates the retrieval summary plus generator columns) |
| `retrieval_comparison_report_table.csv` | Not yet (rounded copy of the retrieval summary) |
| `final_experiment_summary.csv` | Not yet |
| `basic_rag_test_results.json` | Not yet |
| `llama_rag_test_results.json` | Not yet |
| `formal_rag_evaluation_data.json` | Not yet |
| `formal_rag_evaluation_summary.csv` | Not yet |
| `response_quality_evaluation.csv` | Not yet |
| `ragas_evaluation_dataset.json` | Not yet |
| `final_chatbot_validation.csv` / `.json` | Not yet |
| `*.png` (6 figures) | Not applicable (images) |

## 9. Historical environment and unknowns

Only values recorded in the notebooks are listed. Anything not recorded is marked
**Unknown** and must not be reconstructed or guessed.

| Item | Value | Evidence |
|---|---|---|
| Runtime | Google Colab | Drive mounts in every notebook |
| Python | 3.12 (notebook 01 session) | pip output path `/usr/local/lib/python3.12/dist-packages` |
| beautifulsoup4 / lxml / requests / tqdm | 4.13.5 / 6.1.1 / 2.32.4 / 4.67.3 (notebook 01 session) | pip "Requirement already satisfied" output |
| Python and package versions in notebooks 02–06 | **Unknown** | Separate sessions; installs used `pip -q` |
| langchain-text-splitters | **Unknown** | `pip install -q` |
| sentence-transformers, transformers, torch | **Unknown** | `pip install -q` |
| faiss-cpu | **Unknown** | `pip install -q` |
| bitsandbytes, ragas, gradio | **Unknown** | `pip install -q` |
| Embedding model revisions (commit hashes) | **Unknown** | Loaded by name only |
| Llama 3.1 8B Instruct / FLAN-T5 revisions | **Unknown** | Loaded by name only |
| Random seeds | **None set** in any notebook; values **Unknown** | No seeding calls in any notebook |
| Hardware | Tesla T4, 14.56 GB VRAM (notebook 06 generation session); other sessions **Unknown** | Notebook 06 GPU check output |

Embedding inference and greedy decoding (`do_sample=False`) are deterministic in
principle, but results may still vary slightly across hardware and library versions.

## 10. Reproduction status

New code is checked against the archive by tests marked `archive`. Newly measured
values are never written into `Healthcare_RAG/`.

| Stage | Status | Evidence |
|---|---|---|
| Cleaning (notebook 02) | Reproduced exactly: 698 / 698 files | `tests/test_ingestion_archive.py` |
| Chunking (notebook 03) | Reproduced exactly: 698 files, 3,796 chunks, using `langchain-text-splitters` 1.1.3 | `tests/test_ingestion_archive.py` |
| Embedding, indexing, retrieval evaluation | Not yet attempted (milestones 1.4–1.7) | n/a |
