"""Integration tests with the real embedding models (opt-in, cache-only, CPU).

Enable with MEDQUERY_RUN_MODEL_TESTS=1. Models are loaded with local_files_only=True
(and conftest forces HF offline mode), so these tests never download weights; a model
missing from the cache is skipped. Run scripts/analyse_embedding_models.py once to
populate the cache. CPU is used because CUDA inference currently fails on the
development machine (see docs/embeddings/EMBEDDINGS.md).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from medquery.baseline.artifacts import load_baseline_chunks, read_faiss_flat_vectors
from medquery.config import get_settings
from medquery.embeddings import BASELINE_EMBEDDING_SPECS

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("MEDQUERY_RUN_MODEL_TESTS") != "1",
        reason="set MEDQUERY_RUN_MODEL_TESTS=1 to run real-model tests",
    ),
]

# Every 200th baseline chunk: a spread of 19 chunks across the corpus.
SAMPLE = slice(0, None, 200)
MIN_COSINE_VS_ARCHIVE = 0.999


def load(name: str):
    from medquery.embeddings.sentence_transformer import SentenceTransformerEmbedder

    try:
        return SentenceTransformerEmbedder(
            BASELINE_EMBEDDING_SPECS[name], device="cpu", local_files_only=True
        )
    except OSError as error:
        pytest.skip(f"{name} not in local Hugging Face cache: {error}")


@pytest.fixture(scope="module")
def data_dir() -> Path:
    path = get_settings().archive_dir / "data"
    if not (path / "faiss").is_dir():
        pytest.skip("Research archive not available on this machine")
    return path


@pytest.mark.parametrize("name", list(BASELINE_EMBEDDING_SPECS))
def test_model_loads_with_expected_configuration(name: str) -> None:
    embedder = load(name)
    info = embedder.info()
    assert info.dimension == BASELINE_EMBEDDING_SPECS[name].dimension
    assert info.max_seq_length and info.max_seq_length > 0
    assert info.resolved_revision is not None and len(info.resolved_revision) == 40
    assert not any(info.model_prompts.values()) or name == "E5"


@pytest.mark.archive
@pytest.mark.parametrize("name", list(BASELINE_EMBEDDING_SPECS))
def test_document_embeddings_reproduce_archived_vectors(name: str, data_dir: Path) -> None:
    chunks = load_baseline_chunks(data_dir)[SAMPLE]
    archived = read_faiss_flat_vectors(data_dir / "faiss" / name / "faiss_index.bin")[SAMPLE]
    vectors = load(name).embed_documents([c.text for c in chunks])
    cosine = (vectors.astype(np.float64) * archived).sum(axis=1)
    assert cosine.min() >= MIN_COSINE_VS_ARCHIVE, (name, float(cosine.min()))


def test_e5_query_and_passage_prefixes_change_the_embedding() -> None:
    embedder = load("E5")
    text = ["What are the symptoms of asthma?"]
    query, passage = embedder.embed_queries(text), embedder.embed_documents(text)
    assert float((query * passage).sum()) < 0.999


# Notebook 05, step 11 ("FINAL FAISS RETRIEVAL TEST"): top-1 result for this question
# with each model, as printed in the saved notebook output (rounded to 4 decimals).
NOTEBOOK_05_QUESTION = "What are the symptoms of asthma?"
NOTEBOOK_05_TOP1 = {"MiniLM": 0.7398, "MPNet": 0.7600, "E5": 0.8932, "BGE": 0.8265}


@pytest.mark.archive
@pytest.mark.parametrize("name", list(BASELINE_EMBEDDING_SPECS))
def test_query_against_archived_index_reproduces_notebook_05(name: str, data_dir: Path) -> None:
    from medquery.vectorstore import load_archived_store

    embedder = load(name)
    store = load_archived_store(name, data_dir.parent)
    [hits] = store.search(
        embedder.embed_queries([NOTEBOOK_05_QUESTION]),
        3,
        query_fingerprint=embedder.fingerprint(),
    )
    assert hits[0].chunk.chunk_uid == "asthma.txt#1"
    assert hits[0].score == pytest.approx(NOTEBOOK_05_TOP1[name], abs=5e-5)
