"""Tests for medquery_api.embeddings.

The unit tests use a stand-in model: they need no downloads, no PyTorch and run in well under a
second. The tests marked `models` load the real models from the local cache (never
downloading) and compare a sample of chunks with the archive; they are opt-in:

    uv run pytest -m models -rP

The full comparison of all 3,796 chunks is a separate command:
`uv run python -m medquery_api.embeddings validate`.
"""

import ast
import json
import subprocess
import sys
import zlib

import numpy as np
import pytest

from medquery_api import config
from medquery_api.embeddings import (
    MODELS,
    InvalidEmbeddingsError,
    ModelNotCachedError,
    get_model,
    open_model,
)
from medquery_api.embeddings.encoder import DEVICE_ENV, default_device, load_sentence_transformer
from medquery_api.embeddings.validation import (
    SIMILARITY_THRESHOLD,
    check_save_dir,
    compare_with_archive,
    corpus_chunks,
    read_archived_embeddings,
)

DATA = config.data_dir()
NOTEBOOK_04 = config.REPO_ROOT / "Healthcare_RAG" / "notebooks" / "04_Embedding_Generation.ipynb"

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)


class FakeModel:
    """Stands in for a SentenceTransformer: records each call and returns unit vectors that
    depend only on the text."""

    def __init__(self, dimension: int, transform=None):
        self.dimension = dimension
        self.transform = transform
        self.calls: list[tuple[list[str], dict]] = []

    def vector(self, text: str) -> np.ndarray:
        values = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(self.dimension)
        return (values / np.linalg.norm(values)).astype(np.float32)

    def encode(self, texts, **options):
        self.calls.append((list(texts), options))
        vectors = np.array([self.vector(text) for text in texts])
        return self.transform(vectors) if self.transform else vectors


def fake_loader(opened: list | None = None, transform=None):
    def load(spec, device):
        model = FakeModel(spec.dimension, transform)
        if opened is not None:
            opened.append((spec.name, device, model))
        return model

    return load


# --- The model table ---


def test_model_table_matches_notebook_04():
    with NOTEBOOK_04.open("r", encoding="utf-8") as file:
        cells = ["".join(cell["source"]) for cell in json.load(file)["cells"]]
    model_ids = next(
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(cells[4]))
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "EMBEDDING_MODELS"
    )
    batch_sizes = {}
    for name, cell in [("MiniLM", 7), ("MPNet", 11), ("E5", 16), ("BGE", 20)]:
        (call,) = [
            node
            for node in ast.walk(ast.parse(cells[cell]))
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "encode"
        ]
        keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        batch_sizes[name] = keywords["batch_size"]

    assert {name: model.model_id for name, model in MODELS.items()} == model_ids
    assert {name: model.batch_size for name, model in MODELS.items()} == batch_sizes
    assert {name: model.dimension for name, model in MODELS.items()} == {
        "MiniLM": 384,
        "MPNet": 768,
        "E5": 768,
        "BGE": 384,
    }
    assert "f\"passage: {chunk['text']}\"" in cells[15]
    for model in MODELS.values():
        assert len(model.revision) == 40 and all(c in "0123456789abcdef" for c in model.revision)


def test_an_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Choose from: MiniLM, MPNet, E5, BGE"):
        get_model("GPT")


# --- Embedding with a stand-in model ---


@pytest.mark.parametrize("name", MODELS)
def test_only_e5_adds_prefixes_to_documents_and_queries(name):
    opened = []
    with open_model(name, loader=fake_loader(opened)) as embedder:
        embedder.embed_documents(["first chunk", "second chunk"])
        embedder.embed_query("a question")
    model = opened[0][2]

    if name == "E5":
        assert model.calls[0][0] == ["passage: first chunk", "passage: second chunk"]
        assert model.calls[1][0] == ["query: a question"]
    else:
        assert model.calls[0][0] == ["first chunk", "second chunk"]
        assert model.calls[1][0] == ["a question"]


@pytest.mark.parametrize("name", MODELS)
def test_the_research_batch_size_and_normalisation_are_used(name):
    opened = []
    with open_model(name, loader=fake_loader(opened)) as embedder:
        embedder.embed_documents(["text"])
    options = opened[0][2].calls[0][1]

    assert options["batch_size"] == MODELS[name].batch_size
    assert options["normalize_embeddings"] is True


def test_vectors_come_back_one_per_text_in_input_order():
    opened = []
    texts = ["c", "a", "b", "a"]
    with open_model("MPNet", loader=fake_loader(opened)) as embedder:
        vectors = embedder.embed_documents(texts)
        query = embedder.embed_query("q")
    model = opened[0][2]

    assert vectors.shape == (4, 768) and vectors.dtype == np.float32
    for row, text in zip(vectors, texts, strict=True):
        assert np.array_equal(row, model.vector(text))
    assert query.shape == (768,)


def test_no_texts_give_no_vectors_without_calling_the_model():
    opened = []
    with open_model("BGE", loader=fake_loader(opened)) as embedder:
        vectors = embedder.embed_documents([])

    assert vectors.shape == (0, 384)
    assert opened[0][2].calls == []


@pytest.mark.parametrize(
    "transform",
    [
        lambda v: v[:, :-1],  # wrong size
        lambda v: v[:-1],  # one vector missing
        lambda v: np.where(np.arange(v.shape[1]) == 0, np.nan, v),  # NaN
        lambda v: v * 2,  # length 2, not 1
    ],
    ids=["wrong size", "missing vector", "NaN", "not unit length"],
)
def test_bad_model_output_is_rejected(transform):
    with (
        open_model("MiniLM", loader=fake_loader(transform=transform)) as embedder,
        pytest.raises(InvalidEmbeddingsError),
    ):
        embedder.embed_documents(["one", "two"])


def test_only_one_model_is_open_at_a_time_and_it_is_released():
    # Opening BGE while MiniLM is still open fails; MiniLM is released afterwards.
    with (
        open_model("MiniLM", loader=fake_loader()) as first,
        pytest.raises(RuntimeError, match="already open"),
        open_model("BGE", loader=fake_loader()),
    ):
        pass
    assert first.model is None

    with pytest.raises(KeyError), open_model("MiniLM", loader=fake_loader()):
        raise KeyError("an error inside the block still releases the model")
    with open_model("BGE", loader=fake_loader()) as second:
        assert second.spec.name == "BGE"


# --- Loading: cache only, pinned revision, CPU ---


def test_loading_uses_the_pinned_revision_and_the_local_cache_only():
    received = {}

    def factory(model_id, **options):
        received.update(model_id=model_id, **options)
        return "a model"

    spec = get_model("E5")
    assert load_sentence_transformer(spec, "cpu", factory=factory) == "a model"
    assert received == {
        "model_id": "intfloat/multilingual-e5-base",
        "revision": spec.revision,
        "device": "cpu",
        "local_files_only": True,
    }


def test_a_model_missing_from_the_cache_names_the_download_command():
    def factory(model_id, **options):
        raise OSError("not found in the local cache")

    with pytest.raises(ModelNotCachedError, match="python -m medquery_api.embeddings download E5"):
        load_sentence_transformer(get_model("E5"), "cpu", factory=factory)


def test_the_cpu_is_used_unless_configured_otherwise(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    assert default_device() == "cpu"

    monkeypatch.setenv(DEVICE_ENV, "cuda:0")
    assert default_device() == "cuda:0"


def test_importing_the_package_does_not_import_pytorch():
    code = (
        "import sys, medquery_api.embeddings, medquery_api.embeddings.validation, "
        "medquery_api.main; print(sorted(m for m in ('torch', 'sentence_transformers', "
        "'transformers') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "[]"


# --- Validation helpers ---


def test_saving_inside_the_research_archive_is_refused(tmp_path):
    with pytest.raises(ValueError, match="refusing to write inside the research archive"):
        check_save_dir(config.REPO_ROOT / "Healthcare_RAG" / "data" / "new")
    assert check_save_dir(tmp_path) == tmp_path.resolve()


def unit_rows(count: int, dimension: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).standard_normal((count, dimension))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_comparison_accepts_tiny_differences():
    archived = unit_rows(6, 16, seed=1)
    noisy = archived + np.float32(1e-6) * unit_rows(6, 16, seed=2)
    new = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)

    comparison = compare_with_archive(new, archived, list("abcdef"))

    assert comparison.failures(6, 16) == []
    assert comparison.cosine.min() > 0.999999


def test_comparison_reports_a_changed_vector_and_a_wrong_neighbour():
    archived = unit_rows(6, 16, seed=1)
    new = archived.copy()
    new[2] = archived[4]  # chunk 2 now looks like chunk 4

    failures = compare_with_archive(new, archived, list("abcdef")).failures(6, 16)

    assert any("1 chunks have cosine similarity below 0.999 (chunk 2:" in f for f in failures)
    assert any("chunk 2 -> 4" in f for f in failures)


def test_comparison_accepts_a_neighbour_with_identical_text():
    archived = unit_rows(4, 16, seed=1)
    archived[3] = archived[1]  # chunks 1 and 3 have the same text, so the same vector
    texts = ["a", "same", "b", "same"]

    comparison = compare_with_archive(archived.copy(), archived, texts)

    assert comparison.nearest_wrong == []


# --- Real models (opt-in: pytest -m models) ---


@pytest.fixture(scope="module")
def sample():
    """Every 120th chunk plus the longest: (positions, texts), from PRD 2.1's pipeline."""
    chunks = corpus_chunks(DATA)
    longest = max(range(len(chunks)), key=lambda i: len(chunks[i].text))
    positions = sorted(set(range(0, len(chunks), 120)) | {longest})
    return positions, [chunks[i].text for i in positions]


@pytest.mark.models
@needs_data
@pytest.mark.parametrize("name", MODELS)
def test_real_model_reproduces_archived_vectors_for_a_sample(name, sample):
    positions, texts = sample
    _, archived_all = read_archived_embeddings(DATA, name)
    archived = archived_all[positions]
    try:
        with open_model(name) as embedder:
            vectors = embedder.embed_documents(texts)
            again = embedder.embed_documents(texts)
    except ModelNotCachedError as error:
        pytest.skip(str(error))
    comparison = compare_with_archive(vectors, archived, texts)
    print(
        f"measured: {name} {len(texts)} chunks, cosine min {comparison.cosine.min():.9f}, "
        f"largest |length - 1| {comparison.max_length_deviation:.3e}, "
        f"repeat difference {np.abs(vectors - again).max():.3e}"
    )

    assert vectors.shape == (len(positions), MODELS[name].dimension)
    assert comparison.failures(len(positions), MODELS[name].dimension) == []
    assert comparison.cosine.min() >= SIMILARITY_THRESHOLD
    assert np.abs(vectors - again).max() <= 1e-6


@pytest.mark.models
@needs_data
def test_real_e5_needs_its_prefixes_and_applies_them(sample):
    positions, texts = sample
    _, archived_all = read_archived_embeddings(DATA, "E5")
    archived = archived_all[positions]
    question = "What are the symptoms of asthma?"
    try:
        with open_model("E5") as embedder:
            with_prefix = embedder.embed_documents(texts)
            without_prefix = np.asarray(
                embedder.model.encode(texts, batch_size=16, normalize_embeddings=True)
            )
            query = embedder.embed_query(question)
            expected_query = embedder.model.encode(
                [f"query: {question}"], normalize_embeddings=True
            )
    except ModelNotCachedError as error:
        pytest.skip(str(error))
    cosine_with = compare_with_archive(with_prefix, archived, texts).cosine
    cosine_without = (without_prefix * archived).sum(axis=1)
    print(
        f"measured: E5 cosine with archive: with 'passage: ' min {cosine_with.min():.6f}; "
        f"without it min {cosine_without.min():.6f}, mean {cosine_without.mean():.6f}"
    )

    assert cosine_with.min() >= SIMILARITY_THRESHOLD
    # Without the prefix, the archive could not be reproduced: validation would fail.
    assert cosine_without.min() < SIMILARITY_THRESHOLD
    assert np.abs(query - expected_query[0]).max() <= 1e-6
