"""Token-length and reproduction analysis of the four baseline embedding models.

For each model (baseline spec, current library versions, local hardware):
  1. Token lengths: untruncated token counts of all 3,796 baseline chunks, including
     special tokens and the passage prefix, compared with the model's max_seq_length.
  2. Reproduction: re-embed all chunks and compare, row by row, with the archived
     vectors in Healthcare_RAG/data/faiss/<Model>/faiss_index.bin.
  3. Query prefix: for models with a query prefix, how much the prefix changes the
     embeddings of the 10 baseline evaluation questions.

Downloads model weights on first use (~1.8 GB in total) unless --local-files-only.
Reads the archive only; writes JSON reports to docs/embeddings/.

Usage:
    uv run python scripts/analyse_embedding_models.py [--models MPNet E5] [--local-files-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

import numpy as np

from medquery.baseline.artifacts import (
    load_baseline_chunks,
    read_faiss_flat_vectors,
    read_faiss_metadata_keys,
)
from medquery.baseline.loader import load_retrieval_per_question
from medquery.config import get_settings
from medquery.embeddings import BASELINE_EMBEDDING_SPECS, summarise_token_lengths
from medquery.embeddings.sentence_transformer import SentenceTransformerEmbedder


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return (a64 * b64).sum(axis=1) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))


def describe(values: np.ndarray) -> dict:
    return {
        "min": round(float(values.min()), 7),
        "p01": round(float(np.percentile(values, 1)), 7),
        "mean": round(float(values.mean()), 7),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding model analysis.")
    parser.add_argument("--models", nargs="+", default=list(BASELINE_EMBEDDING_SPECS))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-reproduction", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], help="Override MEDQUERY_EMBEDDING_DEVICE."
    )
    args = parser.parse_args()

    import torch

    settings = get_settings()
    data_dir = settings.archive_dir / "data"
    chunks = load_baseline_chunks(data_dir)
    chunk_keys = [(c.source_id, c.chunk_id) for c in chunks]
    texts = [c.text for c in chunks]
    uids = [c.chunk_uid for c in chunks]
    questions = sorted(
        {r.question for r in load_retrieval_per_question(settings.archive_results_dir)}
    )
    print(f"Loaded {len(chunks)} baseline chunks, {len(questions)} evaluation questions")

    created = datetime.now(UTC).isoformat(timespec="seconds")
    token_report: dict = {"created_at": created, "provenance": "new_run", "models": {}}
    repro_report: dict = {"created_at": created, "provenance": "new_run", "models": {}}

    for name in args.models:
        spec = BASELINE_EMBEDDING_SPECS[name]
        print(f"\n=== {name} ({spec.model_id})")
        load_start = time.perf_counter()
        embedder = SentenceTransformerEmbedder(
            spec,
            device=args.device or settings.embedding_device,
            local_files_only=args.local_files_only,
        )
        load_seconds = time.perf_counter() - load_start
        info = embedder.info()
        print(
            f"device={info.device} max_seq_length={info.max_seq_length} "
            f"revision={info.resolved_revision}"
        )

        counts = embedder.count_tokens(texts, kind="document")
        stats = summarise_token_lengths(uids, counts, embedder.max_seq_length)
        token_report["models"][name] = {
            "embedder": info.model_dump(),
            "documents": stats.model_dump(),
        }
        print(
            f"tokens: mean={stats.mean} p95={stats.p95} max={stats.max} "
            f"truncated={stats.truncated_count} ({stats.truncated_fraction:.2%})"
        )

        if args.skip_reproduction:
            continue

        faiss_dir = data_dir / "faiss" / name
        if read_faiss_metadata_keys(faiss_dir / "metadata.json") != chunk_keys:
            raise RuntimeError(f"{name}: archived metadata order differs from baseline chunks")
        archived = read_faiss_flat_vectors(faiss_dir / "faiss_index.bin")

        if info.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        vectors = embedder.embed_documents(texts)
        if info.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak_mb = (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if info.device.startswith("cuda")
            else None
        )

        cos = cosine_rows(vectors, archived)
        entry: dict = {
            "embedder": info.model_dump(),
            "batch_size": spec.batch_size,
            "rows": len(chunks),
            "model_load_seconds": round(load_seconds, 2),
            "embedding_seconds": round(seconds, 2),
            "peak_cuda_memory_mb": peak_mb,
            "row_aligned_cosine_vs_archive": describe(cos),
            "rows_below_0_999": int((cos < 0.999).sum()),
            "rows_below_0_99": int((cos < 0.99).sum()),
            "max_abs_difference": round(float(np.abs(vectors - archived).max()), 7),
        }
        worst = np.argsort(cos)[:5]
        entry["lowest_cosine_rows"] = [
            {"chunk_uid": uids[i], "cosine": round(float(cos[i]), 7), "tokens": counts[i]}
            for i in worst
        ]
        if spec.query_prefix:
            as_query = embedder.embed_queries(questions)
            as_passage = embedder.embed_documents(questions)
            # Deliberately bypass prefixing to measure what a missing prefix would do.
            bare = embedder._embed(list(questions), None)
            entry["query_prefix_effect"] = {
                "questions": len(questions),
                "cosine_query_prefix_vs_passage_prefix": describe(
                    cosine_rows(as_query, as_passage)
                ),
                "cosine_query_prefix_vs_no_prefix": describe(cosine_rows(as_query, bare)),
            }
        repro_report["models"][name] = entry
        print(
            f"reproduction: cosine min={entry['row_aligned_cosine_vs_archive']['min']} "
            f"rows<0.999={entry['rows_below_0_999']} time={seconds:.1f}s peak={peak_mb}MB"
        )

        del embedder, vectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = settings.ensure_writable(settings.docs_dir / "embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token_length_report.json").write_text(
        json.dumps(token_report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if not args.skip_reproduction:
        (out_dir / "embedding_reproduction_report.json").write_text(
            json.dumps(repro_report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(f"\nReports written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
