"""Behaviour of the `Embedder` base class, exercised through `FakeEmbedder`."""

import numpy as np
import pytest

from medquery.embeddings import (
    BASELINE_EMBEDDING_SPECS,
    Embedder,
    EmbedderInfo,
    FakeEmbedder,
    embed_chunks,
)
from medquery.ingestion import Document, chunk_document

E5 = BASELINE_EMBEDDING_SPECS["E5"]
MINILM = BASELINE_EMBEDDING_SPECS["MiniLM"]


def test_outputs_are_float32_unit_norm_with_spec_dimension() -> None:
    vectors = FakeEmbedder(MINILM).embed_documents(["a", "b", "c"])
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_deterministic() -> None:
    a = FakeEmbedder(MINILM).embed_documents(["same text"])
    b = FakeEmbedder(MINILM).embed_documents(["same text"])
    assert np.array_equal(a, b)


def test_prefixes_follow_the_spec() -> None:
    embedder = FakeEmbedder(E5)
    assert embedder.prepare(["x"], "query") == ["query: x"]
    assert embedder.prepare(["x"], "document") == ["passage: x"]
    q, d = embedder.embed_queries(["asthma"]), embedder.embed_documents(["asthma"])
    assert not np.array_equal(q, d)


def test_symmetric_models_embed_queries_and_documents_identically() -> None:
    embedder = FakeEmbedder(MINILM)
    assert np.array_equal(embedder.embed_queries(["asthma"]), embedder.embed_documents(["asthma"]))


def test_batch_size_defaults_to_spec_and_can_be_overridden() -> None:
    embedder = FakeEmbedder(E5)
    embedder.embed_documents(["a", "b"])
    embedder.embed_documents(["a", "b"], batch_size=4)
    assert embedder.encode_calls == [(2, 16), (2, 4)]


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_batch_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(MINILM).embed_documents(["a"], batch_size=size)


def test_empty_input_returns_empty_matrix_without_calling_the_model() -> None:
    embedder = FakeEmbedder(MINILM)
    assert embedder.embed_documents([]).shape == (0, 384)
    assert embedder.count_tokens([]) == []
    assert embedder.encode_calls == []


def test_a_bare_string_or_non_string_items_are_rejected() -> None:
    embedder = FakeEmbedder(MINILM)
    with pytest.raises(TypeError):
        embedder.embed_documents("not a list")
    with pytest.raises(TypeError):
        embedder.embed_documents(["ok", 3])  # type: ignore[list-item]


def test_token_counts_include_the_prefix() -> None:
    assert FakeEmbedder(MINILM).count_tokens(["two words"]) == [4]
    assert FakeEmbedder(E5).count_tokens(["two words"]) == [5]  # + "passage:"
    assert FakeEmbedder(E5).count_tokens(["two words"], kind="query") == [5]


# --- Output validation ------------------------------------------------------------------


class BrokenEmbedder(FakeEmbedder):
    def __init__(self, output: np.ndarray) -> None:
        super().__init__(MINILM)
        self.output = output

    def _encode(self, texts, batch_size):
        return self.output


@pytest.mark.parametrize(
    "output",
    [
        np.ones((1, 383), dtype=np.float32),  # wrong dimension
        np.ones((2, 384), dtype=np.float32),  # wrong row count
        np.full((1, 384), np.nan, dtype=np.float32),  # NaN
        np.ones((1, 384), dtype=np.float32),  # not unit-norm
    ],
)
def test_invalid_model_output_is_rejected(output: np.ndarray) -> None:
    with pytest.raises(ValueError):
        BrokenEmbedder(output).embed_documents(["a"])


def test_embedder_is_abstract() -> None:
    with pytest.raises(TypeError):
        Embedder(MINILM)  # type: ignore[abstract]


# --- Embedding runs -------------------------------------------------------------------


def test_embed_chunks_records_configuration_and_preserves_order() -> None:
    chunks = chunk_document(Document(source_id="a.txt", text="## A\n\n" + "word " * 400))
    embedder = FakeEmbedder(E5)
    result = embed_chunks(embedder, chunks, batch_size=8)

    assert result.vectors.shape == (len(chunks), 768)
    assert result.chunk_uids == tuple(c.chunk_uid for c in chunks)
    assert np.array_equal(result.vectors, embedder.embed_documents([c.text for c in chunks]))
    record = result.record
    assert record.provenance == "new_run"
    assert (record.batch_size, record.num_texts) == (8, len(chunks))
    assert record.fingerprint == E5.fingerprint()
    assert isinstance(record.embedder, EmbedderInfo) and record.embedder.backend == "fake"
    assert record.seconds >= 0
