# Cleaning and Chunking Strategy (Initial)

Status: **milestone 1.3**. The defaults reproduce the dissertation pipeline exactly; all
parameters are configurable for future experiments.

Code: [`medquery.ingestion`](../../backend/src/medquery/ingestion/):
`sources.py` (load) → `cleaner.py` (clean) → `chunker.py` (chunk), with shared models in
`models.py`.

## 1. Input formats

| Format | Status |
|---|---|
| UTF-8 plain text (optionally BOM-prefixed) | Supported (`load_text_document`) |
| PDF, DOCX (FR-02) | Not yet. Future loaders will emit the same `Document` model |

The NHS corpus consists of 698 UTF-8 text files produced by the notebook-01 scraper.
Their structure is Markdown-like: `# Title`, `## Section`, `### Subsection`,
paragraphs, and `• ` bullet lines, separated by blank lines. An inspection of all 698
raw files found no BOMs, no carriage returns, no control characters, and only
NFC-normalised text. Non-ASCII content includes `•`, en dashes, no-break spaces (1,339),
curly quotes, accented Latin, and some Arabic-script text.

Invalid UTF-8 raises `MalformedDocumentError` with the file name and byte offset.
Decoding never silently replaces characters.

## 2. Cleaning (`CleaningConfig`)

| Step | Default | Config field |
|---|---|---|
| Strip BOM and non-whitespace control characters (NUL, ESC, …) | on | `remove_control_characters` |
| Unicode normalisation | NFC | `unicode_normalization` (`NFC`, `NFKC`, `None`) |
| Split on any line break (`\n`, `\r\n`, `\r`, `\f`, U+2028, …), strip lines, drop empty lines | always | n/a |
| Collapse whitespace runs (incl. tabs, no-break spaces) to one space | on | `collapse_whitespace` |
| Drop lines that exactly repeat an earlier line in the same document | on | `deduplicate_lines` |
| Join lines with a blank line | `"\n\n"` | `line_separator` |

Steps 3–6 are the original `clean_text` from notebook 02. The control-character and
Unicode steps are additions for untrusted input. They change nothing on the NHS corpus,
which the golden test confirms. Text with unpaired surrogates is rejected rather than
repaired. Non-string input raises `TypeError`.

## 3. Chunking (`ChunkingConfig`)

| Parameter | Default | Notes |
|---|---|---|
| `strategy` | `recursive_character` | Only strategy implemented |
| `chunk_size` | 1000 | Characters, not tokens; must be > 0 |
| `chunk_overlap` | 200 | Characters; must be ≥ 0 and < `chunk_size` |
| `separators` | `"\n## "`, `"\n### "`, `"\n\n"`, `"\n"`, `". "`, `" "`, `""` | Tried in order |

**How it works.** LangChain's `RecursiveCharacterTextSplitter` (the implementation used
in notebook 03) splits on the first separator that occurs in the text. It then merges
adjacent pieces up to `chunk_size`, carrying up to `chunk_overlap` characters into the
next chunk. Pieces that are still too long are split recursively with the next
separator. Separators are kept at the start of the following piece, so a chunk that
begins a section starts with its `## ` heading.

**Identifiers and metadata.** Every chunk carries:

| Field | Meaning |
|---|---|
| `source_id` | Document identifier (NHS: article file name, e.g. `asthma.txt`) |
| `chunk_id` | 1-based position within the document (as in the research artefacts) |
| `chunk_uid` | `"{source_id}#{chunk_id}"`, unique across a knowledge base |
| `start_index` | Character offset in the cleaned text, so `text[start:start+len] == chunk.text` |
| `chunk_length` | `len(text)`, validated |
| `metadata` | A copy of the document's metadata |

`Chunk.to_legacy_record()` emits the original `data/chunks/*.json` record format.

Empty or whitespace-only documents produce no chunks.

## 4. Reproduction of the research baseline

Verified by `tests/test_ingestion_archive.py` (local archive required):

| Check | Result |
|---|---|
| `clean_text(raw)` vs archived `data/cleaned/*.txt` | 698 / 698 identical |
| Chunker on archived cleaned text vs `data/chunks/*.json` | 698 / 698 files identical, 3,796 chunks |
| Sensitivity check: `chunk_overlap=199` | 18 files differ, so the test detects drift |

Reproduced with `langchain-text-splitters` 1.1.3 (pinned in `backend/uv.lock`). The
version used in the original Colab run is **unknown**.

Corpus statistics (default settings): 3,796 chunks, 5.44 per article; length mean
645.6, min 16, max 999 characters.

## 5. Known limitations

1. **Size is measured in characters, not tokens.** The embedding models have maximum
   input lengths in tokens, and a 1,000-character chunk may exceed the shortest of them
   (MiniLM-L6-v2: 256 word-pieces), which would be silently truncated. This has not
   been measured yet; it should be quantified when the embedding registry is built
   (milestone 1.4).
2. **There are very small and heading-only chunks.** 30 chunks are under 50 characters
   and 16 contain only headings. They add little information and can take up retrieval
   slots.
3. **Section context is lost.** A chunk from the middle of a section does not include
   the article title or section heading (for example, a "Side effects" chunk may not
   name the medicine). This is a candidate improvement (contextual headers); it is not
   implemented, so as not to alter the baseline.
4. **Overlap creates near-duplicates.** Consecutive chunks share up to 200 characters,
   so top-k results often contain several chunks of the same article, which reduces
   the diversity of the context passed to the generator.
5. **Separators assume this corpus's Markdown-like structure.** Text extracted from
   PDF/DOCX will usually lack `## ` headings and fall back to paragraph or line
   splitting.
6. **The size limit is only guaranteed with the `""` fallback separator.** Custom
   separator lists without it can produce chunks longer than `chunk_size` (covered by a
   test).
7. **Line de-duplication is document-wide.** It removed 0 lines from the NHS corpus,
   but for other documents it can drop legitimately repeated lines (for example
   repeated table labels). Set `deduplicate_lines=False` for such knowledge bases.
8. **No source URL is preserved.** The archive's file names are derived from page
   titles and the URL-to-file mapping was not saved, so chunks cannot yet link back to
   nhs.uk pages. The `metadata` field can carry a URL once one is available.
9. **The LangChain dependency is heavy.** `langchain-text-splitters` pulls in
   `langchain-core` and around 28 transitive packages. It is kept because it is the
   original implementation and guarantees identical splitting.
