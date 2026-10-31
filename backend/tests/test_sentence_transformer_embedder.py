"""SentenceTransformerEmbedder with a stand-in model object (no weights, no downloads)."""

from pathlib import Path

import numpy as np
import pytest

from medquery.embeddings import BASELINE_EMBEDDING_SPECS, EmbeddingConfigError
from medquery.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
    cached_revision,
    resolve_device,
)

E5 = BASELINE_EMBEDDING_SPECS["E5"]


class StandInTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class StandInModel:
    """Mimics the parts of `SentenceTransformer` the embedder uses."""

    def __init__(self, dimension=768, default_prompt_name=None, prompts=None) -> None:
        self.dimension = dimension
        self.default_prompt_name = default_prompt_name
        self.prompts = prompts if prompts is not None else {"query": "", "document": ""}
        self.max_seq_length = 512
        self.device = "cpu"
        self.tokenizer = StandInTokenizer()
        self.calls: list[tuple[list[str], dict]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        vectors = np.ones((len(texts), self.dimension), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make(
    model: StandInModel | None = None, spec=E5
) -> tuple[SentenceTransformerEmbedder, StandInModel]:
    model = model or StandInModel()
    return SentenceTransformerEmbedder(spec, model=model), model


def test_encode_receives_prefixed_texts_and_spec_settings() -> None:
    embedder, model = make()
    embedder.embed_queries(["what is asthma?"])
    embedder.embed_documents(["Asthma is..."], batch_size=4)
    (q_texts, q_kwargs), (d_texts, d_kwargs) = model.calls
    assert q_texts == ["query: what is asthma?"] and d_texts == ["passage: Asthma is..."]
    assert q_kwargs["batch_size"] == 16 and d_kwargs["batch_size"] == 4
    for kwargs in (q_kwargs, d_kwargs):
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["show_progress_bar"] is False
        assert "prompt" not in kwargs and "prompt_name" not in kwargs


def test_dimension_mismatch_is_rejected_at_load() -> None:
    with pytest.raises(EmbeddingConfigError, match="dimension"):
        make(StandInModel(dimension=384))


def test_models_with_a_default_prompt_are_rejected() -> None:
    with pytest.raises(EmbeddingConfigError, match="default prompt"):
        make(StandInModel(default_prompt_name="query", prompts={"query": "query: "}))


def test_token_counts_are_untruncated_and_include_special_tokens() -> None:
    embedder, model = make()
    assert embedder.count_tokens(["a b c"]) == [6]  # "passage:" + 3 words + 2 specials
    assert model.tokenizer.kwargs["add_special_tokens"] is True
    assert model.tokenizer.kwargs["truncation"] is False


def test_info_records_what_ran() -> None:
    embedder, _ = make()
    info = embedder.info()
    assert (info.model_id, info.dimension, info.max_seq_length) == (E5.model_id, 768, 512)
    assert (info.device, info.backend, info.device_name) == ("cpu", "sentence-transformers", None)
    assert (info.query_prefix, info.passage_prefix) == ("query: ", "passage: ")
    assert info.requested_revision is None
    assert info.resolved_revision is None  # injected model: no hub snapshot to inspect
    assert info.model_prompts == {"query": "", "document": ""}
    assert "sentence-transformers" in info.library_versions


def test_explicit_device_is_passed_through() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


# --- Revision resolution from the Hugging Face cache ----------------------------------


def write_ref(cache: Path, model_id: str, ref: str, commit: str) -> None:
    refs = cache / f"models--{model_id.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True)
    (refs / ref).write_text(commit, encoding="utf-8")


def test_cached_revision_reads_main_ref(tmp_path: Path) -> None:
    write_ref(tmp_path, "org/model", "main", "a" * 40)
    assert cached_revision("org/model", None, str(tmp_path)) == "a" * 40


def test_cached_revision_passes_through_a_pinned_commit(tmp_path: Path) -> None:
    assert cached_revision("org/model", "b" * 40, str(tmp_path)) == "b" * 40


def test_cached_revision_is_none_when_unknown(tmp_path: Path) -> None:
    assert cached_revision("org/missing", None, str(tmp_path)) is None
    write_ref(tmp_path, "org/bad", "main", "not-a-hash")
    assert cached_revision("org/bad", None, str(tmp_path)) is None
