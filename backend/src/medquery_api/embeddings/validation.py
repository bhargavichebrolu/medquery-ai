"""Compare the backend's embeddings with the research archive's, without changing the archive.

Vectors are compared by closeness, not bit for bit: library versions and hardware differ from
the research, so tiny numerical differences are expected. A model passes only if, for every one
of the 3,796 chunks:

- its vector has length 1 within NORM_TOLERANCE and contains only finite numbers;
- its cosine similarity with the archived vector for the same chunk is at least
  SIMILARITY_THRESHOLD;
- the archived vector closest to it is that same chunk's (or one for an identical text).
"""

import json
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np

from medquery_api import config
from medquery_api.embeddings.encoder import NORM_TOLERANCE, default_device, open_model
from medquery_api.embeddings.models import get_model
from medquery_api.processing import Chunk, chunk_article, clean_text

SIMILARITY_THRESHOLD = 0.999
LIBRARIES = ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy")


def corpus_chunks(data_dir: Path) -> list[Chunk]:
    """Every chunk, made by PRD 2.1's cleaning and chunking from the raw articles, in the order
    notebook 04 embedded them (articles sorted by file name)."""
    chunks: list[Chunk] = []
    for raw_path in sorted((data_dir / "raw").glob("*.txt"), key=lambda path: path.name):
        with raw_path.open("r", encoding="utf-8") as file:
            chunks.extend(chunk_article(clean_text(file.read()), raw_path.name))
    return chunks


def read_archived_embeddings(data_dir: Path, model_name: str) -> tuple[list[dict], np.ndarray]:
    """The archived chunk records (without vectors) and their vectors, read only."""
    path = data_dir / "embeddings" / model_name / "all_embeddings.json"
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    vectors = np.asarray([record.pop("embedding") for record in records], dtype=np.float32)
    return records, vectors


@dataclass
class Comparison:
    count: int
    dimension: int
    all_finite: bool
    max_length_deviation: float
    cosine: np.ndarray  # one value per chunk
    max_abs_difference: float
    nearest_wrong: list[tuple[int, int]] = field(default_factory=list)  # (chunk, nearest found)

    def failures(self, expected_count: int, expected_dimension: int) -> list[str]:
        problems = []
        if (self.count, self.dimension) != (expected_count, expected_dimension):
            problems.append(
                f"{self.count} vectors of size {self.dimension}; expected "
                f"{expected_count} of size {expected_dimension}"
            )
        if not self.all_finite:
            problems.append("some values are NaN or infinite")
        if self.max_length_deviation > NORM_TOLERANCE:
            problems.append(
                f"a vector's length differs from 1 by {self.max_length_deviation:.3e} "
                f"(tolerance {NORM_TOLERANCE:g})"
            )
        below = np.flatnonzero(self.cosine < SIMILARITY_THRESHOLD)
        if below.size:
            listed = ", ".join(f"chunk {i}: {self.cosine[i]:.6f}" for i in below[:20])
            problems.append(
                f"{below.size} chunks have cosine similarity below {SIMILARITY_THRESHOLD} "
                f"({listed}{', ...' if below.size > 20 else ''})"
            )
        if self.nearest_wrong:
            listed = ", ".join(f"chunk {i} -> {j}" for i, j in self.nearest_wrong[:20])
            problems.append(
                f"{len(self.nearest_wrong)} chunks are closest to another chunk's archived "
                f"vector ({listed})"
            )
        return problems


def compare_with_archive(new: np.ndarray, archived: np.ndarray, texts: list[str]) -> Comparison:
    """Compare new vectors with archived vectors for the same texts, row by row."""
    new64, old64 = new.astype(np.float64), archived.astype(np.float64)
    lengths_new = np.linalg.norm(new64, axis=1)
    cosine = (new64 * old64).sum(axis=1) / (lengths_new * np.linalg.norm(old64, axis=1))
    nearest = np.argmax(new @ archived.T, axis=1)
    nearest_wrong = [(i, int(j)) for i, j in enumerate(nearest) if j != i and texts[j] != texts[i]]
    return Comparison(
        count=new.shape[0],
        dimension=new.shape[1],
        all_finite=bool(np.isfinite(new).all()),
        max_length_deviation=float(np.abs(lengths_new - 1.0).max()),
        cosine=cosine,
        max_abs_difference=float(np.abs(new64 - old64).max()),
        nearest_wrong=nearest_wrong,
    )


def check_save_dir(path: Path) -> Path:
    """Refuse to save anywhere inside the read-only research archive."""
    resolved = path.resolve()
    archive = (config.REPO_ROOT / "Healthcare_RAG").resolve()
    if resolved == archive or archive in resolved.parents:
        raise ValueError(f"refusing to write inside the research archive: {resolved}")
    return resolved


def peak_memory_mib() -> float | None:
    """The largest amount of memory this process has used so far, in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        counters = Counters(cb=ctypes.sizeof(Counters))
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return counters.PeakWorkingSetSize / 2**20
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def validate_model(
    name: str,
    data_dir: Path,
    *,
    device: str | None = None,
    save_dir: Path | None = None,
) -> dict:
    """Embed every chunk with one model and compare with the archive. Returns a report."""
    spec = get_model(name)
    if save_dir is not None:
        save_dir = check_save_dir(save_dir)
    chunks = corpus_chunks(data_dir)
    texts = [chunk.text for chunk in chunks]
    archived_records, archived = read_archived_embeddings(data_dir, name)

    with open_model(name, device) as embedder:
        started = time.perf_counter()
        vectors = embedder.embed_documents(texts, show_progress=True)
        seconds = time.perf_counter() - started
        tokenizer, limit = embedder.model.tokenizer, embedder.model.max_seq_length
        token_counts = [
            len(ids)
            for ids in tokenizer([spec.document_prefix + text for text in texts])["input_ids"]
        ]

    comparison = compare_with_archive(vectors, archived, texts)
    failures = comparison.failures(len(archived_records), spec.dimension)
    order_matches = [chunk.as_record() for chunk in chunks] == archived_records
    if not order_matches:
        failures.insert(0, "the chunks do not match the archived records in content or order")
    cosine = comparison.cosine
    report = {
        "model": name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "device": device or default_device(),
        "libraries": {library: version(library) for library in LIBRARIES},
        "chunks": len(chunks),
        "chunks_match_archive": order_matches,
        "dimension": comparison.dimension,
        "all_finite": comparison.all_finite,
        "max_length_deviation": comparison.max_length_deviation,
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "cosine_percentiles": {str(p): float(np.percentile(cosine, p)) for p in (0.1, 1, 5, 50)},
        "max_one_minus_cosine": float((1.0 - cosine).max()),
        "max_abs_difference": comparison.max_abs_difference,
        "nearest_correct": comparison.count - len(comparison.nearest_wrong),
        "token_limit": limit,
        "chunks_over_token_limit": sum(1 for count in token_counts if count > limit),
        "longest_chunk_tokens": max(token_counts),
        "embedding_seconds": round(seconds, 1),
        "peak_memory_mib": peak_memory_mib(),
        "failures": failures,
        "passed": not failures,
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{name}.npy", vectors)
        (save_dir / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
