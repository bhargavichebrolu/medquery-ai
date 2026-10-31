# Vector Store (Initial)

Status: **milestone 1.5**. Code: [`medquery.vectorstore`](../../backend/src/medquery/vectorstore/).

## 1. Design

| Component | Role |
|---|---|
| `VectorStore` (`base.py`) | Backend-independent interface: `add`, `search`, `save`, `len`, `fingerprint`, `knowledge_base_id`, `read_only` |
| `FaissVectorStore` (`faiss_store.py`) | Exact search with FAISS `IndexFlatIP`, the index type used by the dissertation |
| `load_archived_store` (`archive.py`) | Opens a dissertation index in place, read-only |

pgvector (Part 2) will be a second `VectorStore` implementation. Code that uses a store
depends only on the interface.

**One store = one knowledge base + one embedding configuration.** Every store carries
the `EmbeddingFingerprint` of the embedder that produced its vectors (model ID,
revision, dimension, normalisation, query and passage prefixes) and a
`knowledge_base_id` (lowercase, `[a-z0-9_-]`, max 64 characters).

## 2. Guarantees

| Guarantee | How |
|---|---|
| Queries only from a compatible embedder | `search()` requires `query_fingerprint` and calls `ensure_compatible`; for example, MiniLM queries cannot search a BGE store even though both are 384-dimensional |
| Vector validity | Shape, finite values, and unit norm when the fingerprint is normalised (tolerance 1e-3), for both documents and queries |
| Row ↔ chunk alignment | Row *i* of the index is chunk *i*; chunk UIDs must be unique |
| Exact ranking | Flat inner-product search (cosine for normalised vectors); tested against a brute-force NumPy ranking |
| Exact filtering | `source_ids` filtering uses FAISS `IDSelectorBatch` inside the exact search, so filtered results are the true top-k of the matching documents (no over-fetching) |
| Short result lists | If `k` exceeds the available chunks, all available chunks are returned, ranked |
| Integrity on load | `manifest.json` stores SHA-256 checksums of `index.faiss` and `chunks.jsonl`; a mismatch refuses to load |
| Archive protection | `save()` refuses paths inside `Healthcare_RAG/`; archived and `read_only=True` stores refuse `add` and `save` |

## 3. On-disk format (new stores)

```
<store>/
├── index.faiss     # FAISS IndexFlatIP
├── chunks.jsonl    # one Chunk per line, in index row order
└── manifest.json   # StoreManifest: format_version, store_type, knowledge_base_id,
                    # fingerprint, vector_count, provenance, created_at,
                    # build_config, files (SHA-256)
```

`provenance` is `new_run` for stores built by this code and `original_dissertation`
for archived indexes. `build_config` records how the store was produced (cleaning and
chunking settings, source).

## 4. Archived dissertation indexes

`load_archived_store("MPNet")` reads `Healthcare_RAG/data/faiss/MPNet/faiss_index.bin`
directly, with nothing copied or written. Chunks are re-created with
`BASELINE_CLEANING`/`BASELINE_CHUNKING`, and every row is checked field by field against
the archived `metadata.json` before the store is returned. Knowledge base ID:
`nhs-baseline`.

Verified (tests in `test_archived_store.py`):

| Check | Result |
|---|---|
| 4 stores × 3,796 vectors, 698 source documents | Pass |
| Searching with stored row *i* returns chunk *i* first (sampled rows, every model) | Pass |
| Archive checksums unchanged after opening all four stores | Pass |
| Query embedded by the new code reproduces notebook 05's recorded top-1 (`asthma.txt` chunk 1) and similarity | MiniLM 0.7398 ✓, MPNet 0.7600 ✓, E5 0.8932 ✓, BGE 0.8265 ✓ (opt-in model test) |

## 5. Limitations

1. **Exact search is O(n) per query.** It is instant for 3,796 vectors, but will need an
   approximate index (HNSW/IVF) or pgvector indexing at much larger scale. That
   introduces recall trade-offs that would need to be measured.
2. **No update or delete yet.** FR-13 (remove documents, rebuild indexes) needs
   removal by `source_id`. With a flat index this is a rebuild. Planned with pgvector
   in Part 2.
3. **Filtering is by `source_id` only.** Filtering on other metadata fields is not
   implemented.
4. **Whole-store persistence.** `save` rewrites all files, and it is not atomic: an
   interrupted save can leave a store that fails its checksum check on load (it fails
   safely rather than loading corrupt data).
5. **Revision warnings are not surfaced.** Baseline fingerprints have `revision=None`
   (historically unknown), so compatibility checks cannot prove identical weights. The
   check produces a warning, but `search()` currently discards it. The retriever (1.6)
   should report it.
6. **Thread safety is not guaranteed.** Concurrent `add` and `search` on one store
   instance are not synchronised.
