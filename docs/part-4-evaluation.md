# Part 4: evaluation and experiment analytics

This document is the plan for Part 4. It quotes the project PRD's Part 4 scope word for word,
records the decisions made on 3 November 2026, and proposes the detail of each milestone.
Labels used throughout:

- **PRD:** text from the original project PRD (quoted, not reworded).
- **Decided:** a decision made on 3 November 2026, before any Part 4 code.
- **Proposed:** detail written for this plan. Each milestone's proposed detail is reviewed when
  that milestone starts and may change.

Written on 3 November 2026, after Part 3 was committed (`34fc512`). Nothing in Part 4 is
implemented yet. The full project PRD is not in this repository; only its Part 4 is
reproduced here.

## Scope from the PRD

The PRD's Part 4, quoted in full:

> **Purpose:** Make the platform capable of evaluating retrieval and answer quality through
> repeatable experiments.
>
> **4.0 — Evaluation Dataset Format.** Define a structured format for: Questions. Expected
> answers. Relevant documents or chunks.
>
> **4.1 — Retrieval Metrics.** Implement: Recall@K. Precision@K. MRR. nDCG@K.
>
> **4.2 — Answer Evaluation.** Add suitable evaluation methods for: Faithfulness. Answer
> relevance. Answer correctness. Citation accuracy.
>
> **4.3 — Experiment Runner.** Run experiments with controlled configurations.
>
> **4.4 — Experiment Storage.** Store: Model identifiers. Dataset identifier. Retrieval
> configuration. Evaluation metrics. Timestamp. Relevant execution information.
>
> **4.5 — Comparison Dashboard.** Compare models and retrieval configurations.
>
> **Part 4 Endpoint:** The platform can run repeatable experiments and show evidence-backed
> comparisons.

Other PRD sections that bear on Part 4 (summarised):

- **Analytics (§3.5):** charts show actual results only; answer-quality metrics are shown
  "when implemented".
- **Experiments (§3.6):** select a dataset, an embedding model and retrieval settings; run;
  store; compare. To be introduced "only after the underlying retrieval and evaluation
  components are working".
- **Storage (§4.4):** evaluation datasets, experiment configurations and results; a vector
  database such as PostgreSQL with pgvector "may be introduced later".
- **Rules (§6):** ordinary tests need no paid API (§6.4); historical research results stay
  distinguishable from new application results (§6.5); local, low-cost tools are preferred
  (§6.7).
- **Part 5, not Part 4:** hybrid retrieval, reranking, tuning of chunk size, overlap and top-K,
  and performance analysis including resource use and throughput.

## Decisions (3 November 2026)

| # | Topic | Decision |
|---|---|---|
| 1 | Answer evaluation | Start with a versioned human scoring guide and a mock judge that only exercises the plumbing. Paid judging (Claude or Ragas) waits for an explicitly approved budget. |
| 2 | Reference answers | Keep the 10 historical reference answers, clearly labelled **unverified**. They are not treated as ground truth. A review workflow comes later. |
| 3 | Relevance labels | Article level only, as in the research, so results stay comparable with it. Chunk labels would need new human annotation. |
| 4 | Similarity cut-off | Deferred. 4.1 reports score distributions for the 10 in-scope and 8 out-of-scope questions, applies no cut-off, and does not claim the result generalises. |
| 5 | Storage | SQLite, outside `Healthcare_RAG/`, git-ignored. |
| 6 | Running experiments | Command line only (4.3). The browser shows stored results (4.5) and cannot start a run. |
| 7 | Documentation | This document; the whole project PRD is not rewritten. |

## What Part 4 starts from

| Input | Where | Status |
|---|---|---|
| 10 evaluation questions, one relevant article each | Notebook 06 cell 28; `results/formal_rag_evaluation_data.json` (`relevant_sources`) | Historical; used by the research's retrieval evaluation |
| 10 reference answers | `results/ragas_evaluation_dataset.json` (`reference`), written in notebook 06 cell 35 | Historical, **unverified**: written by hand "from the known relevant NHS source documents"; cells 35 and 36 (its check) have no saved output |
| 10 Llama answers with their retrieved chunks and timings | `results/formal_rag_evaluation_data.json` | Historical; never scored for quality |
| 40 retrieval results (4 models × 10 questions) | `results/corrected_retrieval_results.csv` | Historical; reproduced 40 of 40 by `backend/tests/test_retrieval_metrics.py` (PRD 2.4) |
| Metric formulas (by article) | `backend/src/medquery_api/evaluation/retrieval_metrics.py`, from notebook 06 cell 30 | In the application since PRD 4.1 (before that, in the tests only) |
| 8 out-of-scope questions and pass rules | [`llm-and-rag.md`](llm-and-rag.md), "Out-of-scope questions" | Approved at the PRD 3.6 review; in the evaluation dataset since PRD 4.0 |
| Answering the 10 questions with timings, tokens, cost and grounding | `python -m medquery_api.answering review` | Built (PRD 3.3–3.6); refuses paid providers without `--allow-paid-calls` |
| Embedding model versions | `backend/src/medquery_api/embeddings/models.py` | Pinned; inferred to match the research, not recorded by it |

The 10 questions and their relevant articles:

| # | Question | Relevant article |
|---|---|---|
| 1 | What are the symptoms of asthma? | `asthma.txt` |
| 2 | What are the causes of asthma? | `asthma.txt` |
| 3 | How is asthma diagnosed? | `asthma.txt` |
| 4 | What is aciclovir used to treat? | `aciclovir_zovirax.txt` |
| 5 | What are the symptoms of a food allergy? | `food_allergy.txt` |
| 6 | What are the symptoms of hives? | `hives.txt` |
| 7 | What is amoxicillin used for? | `amoxicillin.txt` |
| 8 | What is amlodipine used to treat? | `amlodipine.txt` |
| 9 | What is allopurinol used to treat? | `allopurinol-zyloric_uricto.txt` |
| 10 | What is adalimumab used to treat? | `adalimumab.txt` |

Known limits of these inputs:

- **Precision@5 is at most 0.20:** each question has one relevant article, and an article
  counts once however many of its chunks are retrieved.
- **Ten questions** are too few for reliable rankings; the models differ on only a few
  questions (see the Analytics page).
- **The reference answers are short paraphrases.** For example, question 2's describes what
  triggers asthma symptoms rather than its causes. This is why they are labelled unverified.
- **Exactly tied chunks:** question 6's MRR@5 and nDCG@5 depend on the order of two chunks
  with identical text (see [`research-pipeline.md`](research-pipeline.md), Stage 6).

## Milestones

The order and titles are the PRD's. Everything under each milestone is **proposed** unless
marked **Decided**. Each milestone is planned and approved separately, as in Part 3.

Rules for every milestone:

- `Healthcare_RAG/` stays read-only.
- No paid API call without explicit approval of its cost.
- Ordinary tests use the mock provider, with no network calls.
- Historical research results and new application results are always labelled and never
  averaged together.
- The ranking metric is written MRR@5 (or MRR@K) because only the top K results are scored.

### 4.0: Evaluation dataset format

- **Objective:** one validated, versioned file format for evaluation questions, expected
  answers or behaviour, and relevance labels.
- **Scope:**
  - a JSON dataset file in the repository (outside `Healthcare_RAG/`) and a backend loader that
    validates it;
  - a dataset identifier computed from the file's content, so any change gives a new
    identifier;
  - first contents: the 10 research questions with their relevant articles, the 10 reference
    answers labelled historical and unverified (**decided**), and the 8 out-of-scope questions
    with their approved pass rules and manual-review flags;
  - article-level labels only (**decided**);
  - a read-only view of the dataset, so the milestone has a visible result.
- **Exclusions:** no metrics, runs or scoring; no new labels or reference answers; no chunk
  labels.
- **Acceptance:**
  - questions and relevant articles match the archive exactly, and every relevant article
    exists among the 698;
  - the pass rules match the approved wording;
  - malformed files are rejected with clear messages;
  - the identifier is unchanged for unchanged content.
- **Tests:** validation of valid and invalid files, identifier stability, and a cross-check
  against the archive (skipped, with the reason, when the data folder is missing).
- **Depends on:** nothing new.

**Implemented in PRD 4.0:**

- **Code and data:**
  - the format, loader and checks: `backend/src/medquery_api/evaluation/`;
  - the dataset: `evaluation/datasets/research_evaluation.json`;
  - the format is described in [`backend/README.md`](../backend/README.md), "Evaluation dataset".
- **Checking a file:** `uv run python -m medquery_api.evaluation [PATH]` prints the identifier
  and checks the relevant articles against the knowledge base.
- **Visible result:** `GET /api/evaluation/dataset`, shown read-only at the end of the
  Analytics page as the "Current application evaluation dataset", labelled separately from the
  historical research results (location confirmed at the PRD 4.0 review). On narrow screens
  each question is shown as a card instead of a table row.
- **Drift check:** `frontend/src/data/researchResults.test.ts` compares the frontend's copy of
  the 10 research questions with the dataset file.
- **How each out-of-scope question is decided:**
  - `automatic_pass_statuses`: the outcomes that pass without review;
  - `otherwise`: what any other outcome means, `fail` or `manual_review`;
  - questions 1, 2, 3 and 5 ("pass only if…"): `otherwise` is `fail`;
  - question 4 (manual review required): `manual_review`;
  - questions 7 and 8: no automatic pass, so always `manual_review`;
  - question 6: passes on `not_enough_evidence` or `refused`; any other outcome goes to
    `manual_review`, not an automatic failure (**confirmed at the PRD 4.0 review**, 3 November
    2026). The Analytics page states this distinction for each question.
- **Bundled dataset identifier at PRD 4.0:**
  `sha256:39294b2b09a5a1c86bb4b5f19eb298a4e86a64e83680905b5a8e875fee8d1489`.

### 4.1: Retrieval metrics

- **Objective:** Precision@K, Recall@K, MRR@K and nDCG@K as part of the application, for any
  embedding model, any K and any dataset in the 4.0 format.
- **Scope:**
  - move notebook 06's formulas (by article) from the tests into the backend, with the tie rule
    from PRD 2.4 documented;
  - reproduce all 40 archived results through the application code;
  - score distributions (**decided**): each model's top scores for the 10 in-scope and 8
    out-of-scope questions, reported without any cut-off.
- **Exclusions:** no cut-off applied anywhere, and no claim that the distributions generalise
  (**decided**); no hybrid retrieval or reranking (Part 5).
- **Acceptance:** 40 of 40 archived results reproduced; the formulas agree with hand-worked
  examples; the distribution report states the number of questions it rests on.
- **Tests:** unit tests for each formula (repeated articles, no relevant result, K larger than
  the results); the archive reproduction as an opt-in model test (`-m models`).
- **Depends on:** 4.0.

**Implemented in PRD 4.1:**

- **Code:** `backend/src/medquery_api/evaluation/retrieval_metrics.py`, run with
  `uv run python -m medquery_api.evaluation.retrieval_metrics [--model MPNet] [--k 5] [--out FILE]`
  (local models only; no API calls). The four formula functions are notebook 06's, copied
  unchanged, and a test compares them with the notebook.
- **Exact ties:** the reported scores use the backend's order (exactly tied chunks lower row
  first). Each question also lists every distinct set of scores obtainable by reordering only
  exactly tied results, and an archived result counts as reproduced when the backend's order or
  such a reordering gives it (PRD 2.4's rule). FAISS defines no order for exact ties, so there is
  no notebook rule to copy beyond this.
- **Reproduction:** all 40 archived results are reproduced through the application code: 36 in
  the backend's order, and question 6 (`research-6`) in every model only with its exactly tied
  pair (`hives.txt` chunk 2 and `rashes_in_babies_and_children.txt` chunk 30) reordered.
- **Why question 6 differs:** the two chunks have identical text, so every model gives them
  exactly the same score. The backend lists the lower row (`hives.txt`, the relevant article)
  first, so the relevant article is at rank 1; the research's FAISS search listed the higher
  row (`rashes_in_babies_and_children.txt`) first, so it was at rank 2. Neither order is more
  correct: FAISS defines no order for exact ties.
- **Consequence:** because of that tie, the application's mean MRR@5 for every model is 0.05
  higher than the research's (for example MPNet 1.000, archived 0.95), and its mean nDCG@5 is
  about 0.037 higher. The archived means are the low ends of the reported ranges. **This is
  ranking-order ambiguity between exactly tied chunks, not an improvement in retrieval:** the
  application does not outperform the research.
- **Reporting rule (approved at the PRD 4.1 review):** the headline score is the backend's
  actual order, always shown with the tie range; results are never reordered to match the
  research.

Measured on 3 November 2026 on this computer's CPU, with the bundled dataset
(`sha256:39294b2b…`) at K = 5. Application results, not historical ones:

| Model | Precision@5 | Recall@5 | MRR@5 (tie range) | nDCG@5 (tie range) | Top score, 10 in-scope | Top score, 8 out-of-scope |
|---|---|---|---|---|---|---|
| MiniLM | 0.180 | 0.900 | 0.850 (0.800–0.850) | 0.863 (0.826–0.863) | 0.4915–0.7596, median 0.7047 | 0.1686–0.6944, median 0.5147 |
| MPNet | 0.200 | 1.000 | 1.000 (0.950–1.000) | 1.000 (0.963–1.000) | 0.5258–0.8385, median 0.7613 | 0.2355–0.6772, median 0.5227 |
| E5 | 0.200 | 1.000 | 0.950 (0.900–0.950) | 0.963 (0.926–0.963) | 0.8351–0.9019, median 0.8864 | 0.7858–0.8985, median 0.8577 |
| BGE | 0.200 | 1.000 | 0.950 (0.900–0.950) | 0.963 (0.926–0.963) | 0.7327–0.8935, median 0.8468 | 0.4860–0.8193, median 0.7308 |

In every model the in-scope and out-of-scope ranges of top scores overlap. These figures
describe these 18 questions only; scores are comparable only within one model, no cut-off is
set or implied (decision 4), and none should be read from this table.

### 4.2: Answer evaluation

- **Objective:** documented methods for faithfulness, answer relevance, answer correctness and
  citation accuracy, with every score saying how it was produced.
- **Scope:**
  - a versioned human scoring guide defining each measure and its scale (**decided**);
  - a mock judge that returns fixed, clearly labelled scores, to test the plumbing only
    (**decided**);
  - **citation validity vs citation accuracy:** citation validity (Part 3) checks that quoted
    text matches its source exactly; citation accuracy asks whether the cited text supports
    the sentence, and is scored by a person under the guide;
  - the approved out-of-scope rules checked automatically where they are rule-based, with
    questions 4, 7 and 8 flagged for manual review;
  - every score records its method (rule, mock or human), the guide's version and who or what
    produced it.
- **Exclusions:**
  - no paid judging with Claude or Ragas until a budget is approved (**decided**);
  - no quality conclusions from mock answers or the mock judge;
  - no changes to Part 3's grounding rules based on mock output.
- **Acceptance:**
  - mock-judge scores can never be shown or stored as quality results;
  - every grounding status is tested against every out-of-scope pass rule;
  - the guide's version is recorded with each score.
- **Tests:** rule-based checks for every status and question; mock-judge plumbing; recording
  and validation of human scores; manual-review flags.
- **Depends on:** 4.0.

### 4.3: Experiment runner (command line)

- **Objective:** run one dataset with one controlled configuration and produce one result.
- **Scope:**
  - a command line runner (**decided**), growing out of `answering review`;
  - configuration: dataset, embedding model, top-k, LLM provider and model;
  - output: retrieval metrics, answer checks and scores, latency, tokens and estimated cost;
  - before a paid run: a cost estimate (which must not itself make a paid call), the
    `--allow-paid-calls` flag, and a budget cap the run cannot exceed.
- **Exclusions:** starting runs from the browser (**decided**); chunk size, overlap, hybrid
  retrieval and reranking (Part 5); job queues.
- **Acceptance:**
  - the same configuration with the mock gives identical metrics twice;
  - a paid provider is refused without the flag;
  - a run whose estimate exceeds the cap does not start, and a run that reaches the cap stops.
- **Tests:** end-to-end with the mock; the paid-call guard and budget cap; configuration
  validation.
- **Depends on:** 4.1; 4.2 for answer scores (the runner can be built for retrieval first).

### 4.4: Experiment storage (SQLite)

- **Objective:** keep every run, unchanged, with the fields the PRD lists.
- **Scope:**
  - SQLite through Python's built-in `sqlite3`, outside `Healthcare_RAG/` and git-ignored
    (**decided**); the location configurable;
  - fields:
    - model identifiers: the embedding model and its pinned version, and the LLM provider and
      model;
    - the dataset identifier;
    - the retrieval configuration;
    - metrics and scores, with their method labels;
    - the timestamp;
    - execution information: git commit, package versions, hardware, durations, tokens,
      estimated cost and the date prices were checked;
  - runs can be added but not changed or deleted;
  - historical research results kept separate from application runs (**decided**);
  - a backend API to list and read runs.
- **Exclusions:** no Postgres or pgvector; no authentication; no secrets or credentials in
  stored runs; nothing written into `Healthcare_RAG/`.
- **Acceptance:**
  - a run reads back exactly as written;
  - attempts to change or delete a run fail;
  - a database path inside the archive is refused;
  - stored runs contain no credentials.
- **Tests:** saving and reading runs; the immutability rules; the archive-path refusal; a check
  that stored runs contain no secrets; the list and read API.
- **Depends on:** 4.3.

### 4.5: Comparison dashboard

- **Objective:** compare stored runs in the browser.
- **Scope:**
  - a list of stored runs and side-by-side comparisons by embedding model, top-k and LLM;
  - charts drawn only from stored runs, with historical research results clearly separate;
  - the Dashboard's "Recent experiments" panel filled from stored runs;
  - no button that starts a run (**decided**).
- **Exclusions:** browser-triggered runs; invented or illustrative values.
- **Acceptance:**
  - every number on the page traces back to a stored run or a labelled historical file;
  - an empty store shows a clear empty state;
  - the page works and is accessible on desktop and phone.
- **Tests:** page tests with fixture runs; API tests; a browser check on desktop and phone.
- **Depends on:** 4.4.

## Open questions, decided later

- **Which page shows the comparisons:** a new Experiments page or the Analytics page (4.5).
  Analytics currently shows only historical results.
- **The scoring guide's scales and wording,** and who scores (4.2).
- **The review workflow** for the reference answers (decision 2: "later").
- **The SQLite file's default location** (4.4).
- **Any paid run:** its budget, and which questions and configurations it covers.
