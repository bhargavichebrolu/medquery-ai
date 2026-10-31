"""Establish the provenance of legacy artefacts in the research archive.

Legacy artefacts are files that no saved notebook writes:
  * data/faiss/faiss_index.bin + data/faiss/metadata.json   (top level, not per model)
  * data/embeddings/*.json                                    (one file per article)

For each, this script records shape/metadata and compares the stored vectors,
row by row, against the four model-specific artefacts produced by notebooks 04/05.
Embeddings from different models live in unrelated vector spaces, so row-aligned
cosine similarity is ~1.0 only when the same model produced both.

Read-only with respect to the archive. Uses only the standard library.

Usage:
    uv run python scripts/inspect_legacy_artifacts.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from array import array
from pathlib import Path

from medquery.config import get_settings

MODELS = ("MiniLM", "MPNet", "E5", "BGE")

# faiss write_index_header for IndexFlat: fourcc, d (int32), ntotal (int64),
# two dummy int64, is_trained (bool), metric_type (int32); then a uint64 float
# count followed by the float32 vectors.
_FAISS_HEADER = struct.Struct("<4siqqq?i")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_faiss_flat(path: Path) -> dict:
    raw = path.read_bytes()
    fourcc, dim, ntotal, _, _, is_trained, metric = _FAISS_HEADER.unpack_from(raw, 0)
    offset = _FAISS_HEADER.size
    (count,) = struct.unpack_from("<Q", raw, offset)
    offset += 8
    if fourcc not in (b"IxFI", b"IxF2") or count != dim * ntotal:
        raise ValueError(f"{path} is not a plain IndexFlat file (fourcc={fourcc!r})")
    vectors = array("f")
    vectors.frombytes(raw[offset : offset + count * 4])
    if len(vectors) != count:
        raise ValueError(f"{path} is truncated")
    return {
        "index_type": "IndexFlatIP" if fourcc == b"IxFI" else "IndexFlatL2",
        "dimension": dim,
        "ntotal": ntotal,
        "is_trained": is_trained,
        "metric_type": metric,
        "vectors": vectors,
    }


def row(vectors: array, dim: int, i: int) -> list[float]:
    return list(vectors[i * dim : (i + 1) * dim])


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def summarise(values: list[float]) -> dict:
    return {"min": round(min(values), 6), "mean": round(sum(values) / len(values), 6)}


def metadata_keys(records: list[dict]) -> list[tuple[str, int]]:
    return [(r["source_file"], r["chunk_id"]) for r in records]


def inspect_top_level_faiss(data_dir: Path) -> dict:
    faiss_dir = data_dir / "faiss"
    legacy = read_faiss_flat(faiss_dir / "faiss_index.bin")
    legacy_meta = json.loads((faiss_dir / "metadata.json").read_text(encoding="utf-8"))
    dim, n = legacy["dimension"], legacy["ntotal"]
    rows = range(n)

    report: dict = {
        "files": {
            "faiss_index.bin": sha256(faiss_dir / "faiss_index.bin"),
            "metadata.json": sha256(faiss_dir / "metadata.json"),
        },
        "index_type": legacy["index_type"],
        "dimension": dim,
        "ntotal": n,
        "metadata_records": len(legacy_meta),
        "metadata_fields": sorted(legacy_meta[0]) if legacy_meta else [],
        "vector_norms": summarise(
            [math.sqrt(sum(x * x for x in row(legacy["vectors"], dim, i))) for i in rows]
        ),
        "comparisons": {},
    }

    for model in MODELS:
        model_dir = faiss_dir / model
        candidate = read_faiss_flat(model_dir / "faiss_index.bin")
        candidate_meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        entry: dict = {
            "dimension": candidate["dimension"],
            "metadata_identical_order": metadata_keys(legacy_meta) == metadata_keys(candidate_meta),
            "metadata_identical_text": [r["text"] for r in legacy_meta]
            == [r["text"] for r in candidate_meta],
        }
        if candidate["dimension"] == dim and candidate["ntotal"] == n:
            sims = [
                cosine(row(legacy["vectors"], dim, i), row(candidate["vectors"], dim, i))
                for i in rows
            ]
            max_abs = max(
                abs(a - b)
                for i in rows
                for a, b in zip(
                    row(legacy["vectors"], dim, i), row(candidate["vectors"], dim, i), strict=True
                )
            )
            entry.update(
                {
                    "row_aligned_cosine": summarise(sims),
                    "max_abs_difference": round(max_abs, 8),
                }
            )
        else:
            entry["row_aligned_cosine"] = "not comparable (different shape)"
        report["comparisons"][model] = entry
    return report


def load_model_embeddings(data_dir: Path, model: str) -> dict[tuple[str, int], list[float]]:
    path = data_dir / "embeddings" / model / "all_embeddings.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {(r["source_file"], r["chunk_id"]): r["embedding"] for r in records}


def inspect_per_article_embeddings(data_dir: Path) -> dict:
    files = sorted((data_dir / "embeddings").glob("*.json"))
    article_stems = {p.stem for p in (data_dir / "chunks").glob("*.json")}
    records: list[dict] = []
    field_sets: set[tuple[str, ...]] = set()
    non_article_files: dict[str, int] = {}
    for path in files:
        content = json.loads(path.read_text(encoding="utf-8"))
        if path.stem not in article_stems:
            non_article_files[path.name] = len(content)
        for record in content:
            field_sets.add(tuple(sorted(record)))
            records.append(record)

    with_vectors = [r for r in records if "embedding" in r]
    dims = sorted({len(r["embedding"]) for r in with_vectors})
    keys = [(r["source_file"], r["chunk_id"]) for r in records]
    report: dict = {
        "file_count": len(files),
        "files_not_matching_an_article": non_article_files,
        "record_count": len(records),
        "unique_keys": len(set(keys)),
        "field_sets": [list(fs) for fs in sorted(field_sets)],
        "records_with_embedding": len(with_vectors),
        "embedding_dimensions": dims,
        "other_model_identifier_fields": sorted(
            {k for fs in field_sets for k in fs}
            - {"chunk_id", "source_file", "chunk_length", "text", "embedding"}
        ),
        "comparisons": {},
    }
    if not with_vectors:
        return report

    report["vector_norms"] = summarise(
        [math.sqrt(sum(x * x for x in r["embedding"])) for r in with_vectors]
    )
    for model in MODELS:
        reference = load_model_embeddings(data_dir, model)
        matched = [r for r in with_vectors if (r["source_file"], r["chunk_id"]) in reference]
        comparable = [
            r
            for r in matched
            if len(reference[(r["source_file"], r["chunk_id"])]) == len(r["embedding"])
        ]
        entry: dict = {"rows_compared": len(with_vectors), "keys_found": len(matched)}
        if comparable:
            sims = [
                cosine(r["embedding"], reference[(r["source_file"], r["chunk_id"])])
                for r in comparable
            ]
            entry["row_aligned_cosine"] = summarise(sims)
        else:
            entry["row_aligned_cosine"] = "not comparable (different dimension or no shared keys)"
        report["comparisons"][model] = entry
        del reference
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = settings.archive_dir / "data"
    report = {
        "top_level_faiss": inspect_top_level_faiss(data_dir),
        "per_article_embeddings": inspect_per_article_embeddings(data_dir),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        out = settings.ensure_writable(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
