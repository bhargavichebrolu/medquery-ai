# Tracked Limitations

Known limitations that need investigation or a decision. Historical baseline artefacts
are never modified to resolve an item; any fix applies to new configurations only.

| ID | Limitation | Status | Details |
|---|---|---|---|
| L-001 | NHS source URLs are not linked to articles | Open, under investigation | Below |
| L-002 | Chunk quality: 30 chunks < 50 chars, 16 heading-only, no section context | Open, documented | [CHUNKING.md §5](ingestion/CHUNKING.md) items 2–4 |
| L-003 | Chunk size measured in characters; token truncation | Measured (1.4): 85/53/17/53 chunks truncated (MiniLM/MPNet/E5/BGE), mostly image-credit URLs | [EMBEDDINGS.md §3](embeddings/EMBEDDINGS.md) |
| L-004 | Historical library versions, model revisions and seeds unknown | Closed: recorded as Unknown, not reconstructed | [PROVENANCE.md §9](baseline/PROVENANCE.md) |
| L-005 | Vector store: no document removal/update; source-only filters; exact O(n) search | Open, planned for Part 2 (pgvector) | [VECTOR_STORE.md §5](vectorstore/VECTOR_STORE.md) |
| L-006 | Scraped image-credit URLs in 176 chunks (77 articles); 63 chunks are >50% URL | Open, documented; baseline unchanged | [EMBEDDINGS.md §3](embeddings/EMBEDDINGS.md) |
| L-007 | CUDA inference fails on the RTX 4060 dev machine (illegal memory access) | Open, unresolved; CPU used | [EMBEDDINGS.md §6](embeddings/EMBEDDINGS.md) |

## L-001: Missing source URLs

**Facts.**
- `Healthcare_RAG/data/urls/nhs_urls.txt` lists the source URLs.
- Notebook 01 named each saved article after its page `<h1>` title (for example,
  `allopurinol-zyloric_uricto.txt`), not its URL.
- The URL → file mapping was not saved, and file timestamps were lost when the archive
  was copied.

**Impact.** Answers can cite an article file name but cannot link to the nhs.uk page.

**Rules.** No URL may be attached to an article unless the mapping is verified. Guessed
URLs must not appear in citations.

**Candidate approaches** (none attempted yet; each needs approval):
1. Match URL slugs to file names deterministically, accepting only unambiguous one-to-one
   matches and leaving the rest unresolved.
2. Fetch each URL's current `<h1>` and compare it with the archived title. Pages may have
   changed since collection, so a match only shows the page has the same title today.
3. Keep citations at file-name level and record the URL as unknown.
