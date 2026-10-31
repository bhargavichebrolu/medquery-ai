# Embedding Models and Tokenisation

Status: **milestone 1.4**. Code: [`medquery.embeddings`](../../backend/src/medquery/embeddings/).
Measurements: [`token_length_report.json`](token_length_report.json) and
[`embedding_reproduction_report.json`](embedding_reproduction_report.json), produced by
`backend/scripts/analyse_embedding_models.py` (run 2026-10-31, CPU). All figures below
are **new runs** with current library versions; the dissertation's own results are
unchanged and are documented separately in [PROVENANCE.md](../baseline/PROVENANCE.md).

## 1. Architecture

| Component | Role |
|---|---|
| `EmbeddingModelSpec` (`specs.py`) | Immutable description of how text becomes a vector: model ID, revision, dimension, normalisation, query/passage prefixes, batch size |
| `BASELINE_EMBEDDING_SPECS` | The four dissertation configurations, read-only. Experimental specs are derived with `model_copy(update=...)` and never replace these |
| `EmbeddingFingerprint` + `check_compatibility` | The properties that define a vector space. Batch size is excluded because it does not change the vectors |
| `Embedder` (`base.py`) | Backend-independent interface. Applies prefixes in one place for indexing and querying, and validates every output (shape, finite values, unit norm) |
| `SentenceTransformerEmbedder` | Real backend. Torch is imported only when a model is loaded. Models that define a *default prompt* are rejected, because the library would silently add a second prefix |
| `FakeEmbedder` | Deterministic, hash-based test double. No downloads, no torch |
| `embed_chunks` → `EmbeddingRunRecord` | Embeds chunks and records model ID, requested and resolved revision, dimension, max input length, device, library versions, batch size and timing, with provenance `new_run` |
| `count_tokens` / `summarise_token_lengths` | Untruncated token counts, including special tokens and the prefix, and truncation statistics |

**Batch size** comes from the spec (baseline: MiniLM 32, MPNet 16, E5 16, BGE 32, as in
notebook 04) and can be overridden per call.

**Offline tests.** `tests/conftest.py` sets `HF_HUB_OFFLINE=1` for the whole test
session, so no test can download weights. Real-model tests are opt-in
(`MEDQUERY_RUN_MODEL_TESTS=1`), load from the local cache only, and skip if a model is
not cached.

## 2. Models

| Name | Model ID | Dim | Max input (tokens) | Prefix (query / passage) | Tokeniser | Cache size | Revision used in this run |
|---|---|---|---|---|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | none / none | BERT WordPiece, uncased (30,522) | 88 MB | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | 768 | 384 | none / none | MPNet WordPiece (30,527) | 419 MB | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| E5 | `intfloat/multilingual-e5-base` | 768 | 512 | `query: ` / `passage: ` | XLM-R SentencePiece (250,002) | 1.1 GB | `d128750597153bb5987e10b1c3493a34e5a4502a` |
| BGE | `BAAI/bge-small-en-v1.5` | 384 | 512 | none / none | BERT WordPiece, uncased (30,522) | 129 MB | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |

The revisions are what the Hugging Face cache resolved today. **The revisions used in
the dissertation are unknown**, so baseline specs keep `revision=None`. Libraries in
this run: sentence-transformers 6.2.0, transformers 5.22.1, torch 2.11.0+cu128,
tokenizers 0.23.3.

## 3. Token lengths and truncation (3,796 baseline chunks)

| Model | Mean | Median | p95 | Max | Truncated chunks | Articles affected | Tokens lost |
|---|---|---|---|---|---|---|---|
| MiniLM | 154.6 | 157 | 234 | 592 | **85 (2.24%)** | 40 | 16,407 |
| MPNet | 154.6 | 157 | 234 | 592 | 53 (1.40%) | 25 | 8,986 |
| E5 | 165.2 | 169 | 249 | 525 | 17 (0.45%) | 14 | 158 |
| BGE | 154.6 | 157 | 234 | 592 | 53 (1.40%) | 25 | 2,202 |

Counts include special tokens and, for E5, the `passage: ` prefix. A truncated chunk is
embedded from its first `max_seq_length` tokens only; the rest has no effect on its
vector.

**Findings.**

1. **Truncation is uncommon and mostly affects noise.** 68 of the 85 MiniLM-truncated
   chunks contain image-credit URLs scraped from the NHS pages (sciencephoto.com,
   alamy.com, shutterstock.com, …). A 999-character chunk of URL query strings becomes
   about 590 word-pieces. Across the corpus, 176 chunks in 77 articles contain URLs,
   and 63 chunks are more than half URL characters. This is a baseline data-quality
   issue (L-006). The baseline is not altered.
2. **MiniLM, MPNet and BGE tokenise identically.** All three produce exactly the same
   word-piece sequences for all 3,796 chunks (they share the uncased BERT WordPiece
   vocabulary). Their counts therefore match, and only the limits differ. E5's
   multilingual tokeniser shares no sequence with them.
3. **Truncation touches one evaluation article.** Of the 8 articles in the baseline
   evaluation, only `hives.txt` has truncated chunks: MiniLM #2, #4, #7, #8;
   MPNet/BGE #4, #7; E5 #7. This may slightly affect the hives question for MiniLM.
4. **The dissertation's 1,000-character chunk size is safe for E5 and BGE on normal
   prose.** The tail above 256 tokens is almost entirely URL noise.

## 4. Reproduction of the archived document vectors

All 3,796 chunks were re-embedded with each baseline spec on CPU and compared row by row
with `Healthcare_RAG/data/faiss/<Model>/faiss_index.bin`:

| Model | Min row cosine | Rows < 0.999 | Max abs. difference | Embedding time (CPU) | Model load |
|---|---|---|---|---|---|
| MiniLM | 1.0000000 | 0 | 1e-7 | 96 s | 17 s |
| MPNet | 1.0000000 | 0 | 3e-7 | 821 s | 31 s |
| E5 | 1.0000000 | 0 | 4e-7 | 566 s | 56 s |
| BGE | 1.0000000 | 0 | 2e-7 | 311 s | 9 s |

**The current code and libraries reproduce the dissertation's document embeddings
exactly** (to float32 rounding) for all four models. That the vectors match suggests
the weights are unchanged, but it does not prove which revision was used historically.

**Query side.** Querying the archived indexes through the new embedder and vector store
reproduces notebook 05's recorded top-1 result for *"What are the symptoms of asthma?"*
(`asthma.txt` chunk 1) and its printed similarity for all four models: MiniLM 0.7398,
MPNet 0.7600, E5 0.8932, BGE 0.8265 (opt-in test
`test_query_against_archived_index_reproduces_notebook_05`).

The original Colab timings (MiniLM 420.69 s, MPNet 2,813.16 s, E5 2,852.27 s, BGE
934.92 s) were measured on different, unrecorded hardware and **are not comparable**
with the timings above.

## 5. Query/document compatibility

| Rule | Enforcement |
|---|---|
| Query and document vectors must come from the same model ID, dimension, normalisation and prefixes | `check_compatibility` errors; the vector store calls it on every search |
| Different pinned revisions are incompatible | Error |
| Unknown revision on either side | Warning: identical weights cannot be verified |
| Same dimension is not enough | MiniLM and BGE are both 384-dimensional but incompatible; tested |

**E5 prefixes matter.** For the 10 baseline questions, the cosine similarity between a
question embedded with `query: ` and the same question embedded with `passage: ` has a
mean of 0.945 (min 0.939). Against no prefix it is 0.969 (min 0.950). The model treats
these as noticeably different inputs, so a missing or wrong prefix would silently
degrade retrieval. Prefixes are part of the fingerprint for this reason.

**BGE query instruction.** BGE-small-en-v1.5's authors suggest an optional query
instruction for retrieval. The dissertation did not use one, so the baseline spec has
no prefix. Adding it would be a separate experimental spec with a different
fingerprint, requiring re-indexing.

## 6. Hardware and the known CUDA issue

**Measured on:** Intel Core i7-12650H (16 threads), 15.7 GB RAM, NVIDIA RTX 4060 Laptop
GPU (8 GB), Windows 11, driver 610.62.

| Need | Detail |
|---|---|
| Disk | About 1.8 GB for all four models in the Hugging Face cache, downloaded on first use (no token needed) |
| CPU | Sufficient for this corpus: indexing all 3,796 chunks takes 1.5–14 min per model. Query latency is not yet measured (milestones 1.6–1.7) |
| RAM | Not measured precisely. The largest model (E5, 278M parameters, fp32) needs about 1.1 GB for weights plus activations |
| GPU | Optional. **Currently unusable on the development machine** (below) |

**CUDA failure.** Every transformer encoder forward pass on the RTX 4060 fails with
`CUDA error: an illegal memory access was encountered`. Basic CUDA tensor operations,
embedding lookups, matrix multiplies and PyTorch's native `TransformerEncoder` all
work. Ruled out:

- transformers 5.0.0 → 5.22.1
- torch 2.10.0+cu128 and 2.11.0+cu128
- eager and SDPA attention
- cuBLAS/cuBLASLt selection and TF32
- allocator settings

The suspected cause is the driver (610.62) combined with these CUDA builds. **This is
unresolved.** Set `MEDQUERY_EMBEDDING_DEVICE=cpu` on this machine, because `auto` selects
CUDA when it reports as available. Tracked as L-007.

## 7. Limitations

1. Historical model revisions are unknown (L-004). Exact vector reproduction is strong
   evidence, but not proof, that the weights are the same.
2. Image-credit URLs inflate token counts and add retrieval noise (L-006). A future
   *experimental* cleaning configuration could strip them; the baseline will not be
   changed.
3. GPU inference fails on the development machine (L-007), so embedding runs on CPU only.
4. RAM and CPU memory peaks were not measured; GPU peak memory is not available while
   CUDA fails.
5. Truncation is silent inside the models. The measurement reports it, but nothing
   warns at embedding time.
6. Quantisation, ONNX and other backends are not implemented.
