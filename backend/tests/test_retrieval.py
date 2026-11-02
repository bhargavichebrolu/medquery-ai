"""Tests for the retrieval API, POST /api/retrieve.

The ordinary tests use a stand-in model and a tiny generated knowledge base: no PyTorch, no
model files, no research data. The tests marked `models` use the real cached models and the
research archive, and compare retrieval with the research's recorded results; they are
opt-in (never downloading):

    uv run pytest -m models -rP
"""

import ast
import csv
import gc
import itertools
import json
import math
import threading
import time
import weakref
import zlib
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
from fastapi.testclient import TestClient

from medquery_api import config
from medquery_api.embeddings import MODELS, ModelNotCachedError
from medquery_api.embeddings.query import QueryEmbedder
from medquery_api.knowledge_base.store import _load
from medquery_api.main import (
    RetrieveRequest,
    app,
    get_data_dir,
    get_knowledge_base_loader,
    get_query_embedder,
)

DATA = config.data_dir()
RESULTS = config.REPO_ROOT / "Healthcare_RAG" / "results"
NOTEBOOK_06 = config.REPO_ROOT / "Healthcare_RAG" / "notebooks" / "06_RAG_Pipeline(1).ipynb"

client = TestClient(app)

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)


# --- A tiny knowledge base and a stand-in model ---

ARTICLES = {"asthma.txt": 10, "hives.txt": 8, "migraine.txt": 6}  # 24 chunks in all


def index_bytes(vectors: np.ndarray) -> bytes:
    """An index file in the layout faiss.write_index uses for IndexFlatIP."""
    rows, dimension = vectors.shape
    header = b"IxFI" + __import__("struct").pack(
        "<iqqqbiQ", dimension, rows, 2**20, 2**20, 1, 0, rows * dimension
    )
    return header + vectors.astype("<f4").tobytes()


def unit_vector(seed: str, dimension: int) -> np.ndarray:
    values = np.random.default_rng(zlib.crc32(seed.encode())).standard_normal(dimension)
    return (values / np.linalg.norm(values)).astype(np.float32)


def make_data_dir(folder: Path) -> list[dict]:
    records = []
    (folder / "chunks").mkdir(parents=True)
    for source_file, count in sorted(ARTICLES.items()):
        article = [
            {
                "chunk_id": i,
                "source_file": source_file,
                "chunk_length": len(f"{source_file} part {i}"),
                "text": f"{source_file} part {i}",
            }
            for i in range(1, count + 1)
        ]
        (folder / "chunks" / source_file.replace(".txt", ".json")).write_text(
            json.dumps(article), encoding="utf-8"
        )
        records.extend(article)
    for name, model in MODELS.items():
        model_dir = folder / "faiss" / name
        model_dir.mkdir(parents=True)
        vectors = np.stack([unit_vector(f"{name}:{r['text']}", model.dimension) for r in records])
        (model_dir / "faiss_index.bin").write_bytes(index_bytes(vectors))
        (model_dir / "metadata.json").write_text(json.dumps(records), encoding="utf-8")
    return records


class FakeModel:
    """Stands in for a SentenceTransformer. Vectors depend on the model and the text."""

    def __init__(self, name: str, dimension: int, holder: "Holder"):
        self.name, self.dimension, self.holder = name, dimension, holder

    def encode(self, texts, **options):
        self.holder.encoded.append((self.name, list(texts)))
        if self.holder.encode_delay:
            time.sleep(self.holder.encode_delay)
            # The lock must stop another request swapping the model during this embedding.
            if self.holder.embedder.loaded_model != self.name:
                self.holder.swapped_during_use.append(self.name)
        return np.stack([unit_vector(f"{self.name}:{text}", self.dimension) for text in texts])


class Holder:
    """A QueryEmbedder with a stand-in loader, recording what happens."""

    def __init__(self, *, load_delay: float = 0.0, encode_delay: float = 0.0, missing=()):
        self.loaded: list[str] = []
        self.encoded: list[tuple[str, list[str]]] = []
        self.swapped_during_use: list[str] = []
        self.models: list[weakref.ref] = []
        self.load_delay, self.encode_delay, self.missing = load_delay, encode_delay, missing
        self.embedder = QueryEmbedder(loader=self.load)

    def load(self, spec, device):
        if spec.name in self.missing:
            raise ModelNotCachedError(
                f"{spec.name} is not in the local model cache. Download it with: uv run python "
                f"-m medquery_api.embeddings download {spec.name}"
            )
        time.sleep(self.load_delay)
        self.loaded.append(spec.name)
        model = FakeModel(spec.name, spec.dimension, self)
        self.models.append(weakref.ref(model))
        return model


@pytest.fixture
def api(tmp_path):
    """The API with a tiny data folder and a stand-in model loader."""
    make_data_dir(tmp_path)
    holder = Holder()
    knowledge_base_loads = []

    def counting_loader():
        load = get_knowledge_base_loader(tmp_path)

        def wrapped():
            knowledge_base_loads.append(1)
            return load()

        return wrapped

    app.dependency_overrides[get_data_dir] = lambda: tmp_path
    app.dependency_overrides[get_knowledge_base_loader] = counting_loader
    app.dependency_overrides[get_query_embedder] = lambda: holder.embedder
    _load.cache_clear()
    holder.knowledge_base_loads = knowledge_base_loads
    yield holder
    app.dependency_overrides.clear()
    _load.cache_clear()


def post(question="What are the symptoms of asthma?", model="BGE", **extra):
    return client.post("/api/retrieve", json={"question": question, "model": model, **extra})


# --- Responses ---


def test_a_valid_request_returns_ranked_chunks_with_metadata_and_timings(api, tmp_path):
    response = post(topK=3)
    body = response.json()

    assert response.status_code == 200
    assert list(body) == [
        "question",
        "model",
        "modelId",
        "revision",
        "topK",
        "results",
        "timings",
    ]
    assert body["model"] == "BGE"
    assert body["modelId"] == "BAAI/bge-small-en-v1.5"
    assert body["revision"] == MODELS["BGE"].revision
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert list(body["results"][0]) == [
        "rank",
        "score",
        "sourceFile",
        "chunkId",
        "chunkLength",
        "text",
    ]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    for result in body["results"]:
        assert result["text"] == f"{result['sourceFile']} part {result['chunkId']}"
        assert result["chunkLength"] == len(result["text"])
    assert list(body["timings"]) == ["modelLoadMs", "embeddingMs", "searchMs"]


def test_results_are_the_exact_search_of_the_embedded_question(api, tmp_path):
    body = post("Tell me about hives", model="MPNet", topK=20).json()

    records = json.loads((tmp_path / "faiss" / "MPNet" / "metadata.json").read_text("utf-8"))
    vectors = np.stack([unit_vector(f"MPNet:{r['text']}", 768) for r in records])
    scores = vectors @ unit_vector("MPNet:Tell me about hives", 768)
    expected = sorted(range(len(records)), key=lambda row: (-scores[row], row))[:20]

    assert [(r["sourceFile"], r["chunkId"]) for r in body["results"]] == [
        (records[row]["source_file"], records[row]["chunk_id"]) for row in expected
    ]
    assert [r["score"] for r in body["results"]] == pytest.approx(
        [float(scores[row]) for row in expected], abs=1e-6
    )


def test_top_k_defaults_to_five(api):
    body = post().json()

    assert body["topK"] == 5
    assert len(body["results"]) == 5


@pytest.mark.parametrize("name", MODELS)
def test_only_e5_adds_its_query_prefix_and_the_question_is_trimmed(api, name):
    body = post("  What causes migraines?  ", model=name).json()

    expected = "query: What causes migraines?" if name == "E5" else "What causes migraines?"
    assert api.encoded == [(name, [expected])]
    assert body["question"] == "What causes migraines?"


def test_the_supported_models_are_exactly_the_four_research_models():
    model_field = RetrieveRequest.model_fields["model"].annotation

    assert set(get_args(model_field)) == set(MODELS)


# --- Input checks, before anything is loaded ---


@pytest.mark.parametrize(
    "body",
    [
        {"model": "BGE"},
        {"question": "", "model": "BGE"},
        {"question": "   \n\t ", "model": "BGE"},
        {"question": "x" * 1001, "model": "BGE"},
        {"question": "q"},
        {"question": "q", "model": "GPT"},
        {"question": "q", "model": "bge"},
        {"question": "q", "model": "BGE", "topK": 0},
        {"question": "q", "model": "BGE", "topK": 21},
        {"question": "q", "model": "BGE", "topK": 2.5},
        {"question": "q", "model": "BGE", "topK": 2.0},
        {"question": "q", "model": "BGE", "topK": "3"},
        {"question": "q", "model": "BGE", "topK": True},
        {"question": "q", "model": "BGE", "topK": None},
        {"question": "q", "model": "BGE", "unexpected": 1},
    ],
    ids=[
        "no question",
        "empty question",
        "whitespace question",
        "1001 characters",
        "no model",
        "unknown model",
        "wrong case",
        "topK 0",
        "topK 21",
        "topK 2.5",
        "topK 2.0",
        "topK text",
        "topK true",
        "topK null",
        "unknown field",
    ],
)
def test_invalid_requests_are_rejected_before_loading_anything(api, body):
    response = client.post("/api/retrieve", json=body)

    assert response.status_code == 422
    assert api.loaded == []
    assert api.knowledge_base_loads == []


def test_the_question_limit_applies_after_trimming(api):
    assert post(" " * 50 + "x" * 1000 + " " * 50).status_code == 200


# --- Keeping one model loaded ---


def test_model_load_time_is_zero_when_the_model_is_already_loaded(api):
    api.load_delay = 0.05
    first = post(model="BGE").json()["timings"]
    second = post(model="BGE").json()["timings"]

    assert first["modelLoadMs"] >= 40
    assert second["modelLoadMs"] == 0
    assert api.loaded == ["BGE"]
    assert second["embeddingMs"] > 0 and second["searchMs"] > 0


def test_switching_models_loads_the_new_one_and_releases_the_old_one(api):
    post(model="BGE")
    old = api.models[0]
    switched = post(model="E5").json()["timings"]
    gc.collect()

    assert api.loaded == ["BGE", "E5"]
    assert api.embedder.loaded_model == "E5"
    assert old() is None  # the BGE stand-in is no longer held anywhere
    assert switched["modelLoadMs"] > 0


def test_concurrent_requests_never_swap_a_model_in_use(api):
    api.encode_delay = 0.05
    models = ["BGE", "E5", "BGE", "MiniLM", "E5", "MPNet"]
    responses = [None] * len(models)

    def call(position, name):
        responses[position] = post(model=name)

    threads = [threading.Thread(target=call, args=item) for item in enumerate(models)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [r.status_code for r in responses] == [200] * len(models)
    assert [r.json()["model"] for r in responses] == models
    assert api.swapped_during_use == []


def test_concurrent_first_requests_for_one_model_load_it_once(api):
    api.load_delay = 0.05
    threads = [threading.Thread(target=post, kwargs={"model": "MPNet"}) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert api.loaded == ["MPNet"]


# --- Errors ---


def assert_safe_error(response, tmp_path):
    text = response.text
    assert "Traceback" not in text
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
    assert str(config.REPO_ROOT) not in text


def test_a_model_missing_from_the_cache_gives_503_naming_the_download_command(api, tmp_path):
    api.missing = ("E5",)
    response = post(model="E5")

    assert response.status_code == 503
    assert "uv run python -m medquery_api.embeddings download E5" in response.json()["detail"]
    assert_safe_error(response, tmp_path)
    assert post(model="BGE").status_code == 200  # other models still work


def test_missing_research_data_gives_503(api, tmp_path):
    app.dependency_overrides[get_knowledge_base_loader] = lambda: get_knowledge_base_loader(
        tmp_path / "nothing-here"
    )
    response = post()

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Research data folder not found.")
    assert_safe_error(response, tmp_path)
    assert api.loaded == []


def test_damaged_research_data_gives_503_and_logs_the_reason(api, tmp_path, caplog):
    index = tmp_path / "faiss" / "BGE" / "faiss_index.bin"
    index.write_bytes(index.read_bytes()[:-4])
    response = post()

    assert response.status_code == 503
    assert "did not pass its integrity checks" in response.json()["detail"]
    assert_safe_error(response, tmp_path)
    assert "the header implies exactly" in caplog.text


def test_requests_leave_the_data_folder_unchanged(api, tmp_path):
    def snapshot():
        return {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = snapshot()
    for name in MODELS:
        post(model=name, topK=20)

    assert snapshot() == before


# --- Comparison with the research's recorded results (opt-in) ---

TIE_TOLERANCE = 1e-5  # archived scores this close count as tied; their order may differ
SCORE_TOLERANCE = 1e-4  # our scores must match archived scores this closely
METRIC_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def real_api():
    """The real API (cached models only, the research archive)."""
    app.dependency_overrides.clear()
    _load.cache_clear()
    yield
    get_query_embedder().close()
    _load.cache_clear()


def retrieve_real(question: str, model: str, top_k: int = 5) -> list[dict]:
    response = post(question, model=model, topK=top_k)
    if response.status_code == 503 and "download" in response.json()["detail"]:
        pytest.skip(response.json()["detail"])
    assert response.status_code == 200, response.text
    return response.json()["results"]


def tie_groups(scores: list[float]) -> list[list[int]]:
    """Positions grouped into runs of adjacent scores within TIE_TOLERANCE of each other."""
    groups = [[0]]
    for position in range(1, len(scores)):
        if abs(scores[position - 1] - scores[position]) <= TIE_TOLERANCE:
            groups[-1].append(position)
        else:
            groups.append([position])
    return groups


@pytest.mark.models
@needs_data
def test_bge_reproduces_the_archived_chunks_and_scores(real_api):
    with (RESULTS / "llama_rag_test_results.json").open("r", encoding="utf-8") as file:
        archived = json.load(file)
    problems = []
    for item in archived:
        assert item["embedding_model"] == "BGE" and item["top_k"] == 5
        results = retrieve_real(item["question"], "BGE")
        ours = {(r["sourceFile"], r["chunkId"]): r["score"] for r in results}
        theirs = [(s["source_file"], s["chunk_id"]) for s in item["sources"]]
        their_scores = [s["similarity"] for s in item["sources"]]
        our_order = [(r["sourceFile"], r["chunkId"]) for r in results]
        for group in tie_groups(their_scores):
            if {our_order[p] for p in group} != {theirs[p] for p in group}:
                problems.append(f"{item['question']!r} ranks {[p + 1 for p in group]} differ")
        for chunk, score in zip(theirs, their_scores, strict=True):
            if chunk in ours and abs(ours[chunk] - score) > SCORE_TOLERANCE:
                problems.append(f"{item['question']!r} {chunk}: {ours[chunk]} vs {score}")
        largest = max(
            abs(ours.get(c, math.inf) - s) for c, s in zip(theirs, their_scores, strict=True)
        )
        print(f"measured: BGE {item['question']!r}: largest score difference {largest:.3e}")

    assert problems == []


@pytest.mark.models
@needs_data
def test_mpnet_reproduces_the_archived_articles_and_chunk_texts(real_api):
    with (RESULTS / "formal_rag_evaluation_data.json").open("r", encoding="utf-8") as file:
        archived = json.load(file)
    problems = []
    for item in archived:
        assert item["embedding_model"] == "MPNet" and item["top_k"] == 5
        results = retrieve_real(item["question"], "MPNet")
        ours = [(r["sourceFile"], r["text"]) for r in results]
        theirs = list(zip(item["retrieved_sources"], item["contexts"], strict=True))
        # No scores were archived here, so ties are judged from our own scores.
        for group in tie_groups([r["score"] for r in results]):
            if sorted(ours[p] for p in group) != sorted(theirs[p] for p in group):
                problems.append(f"question {item['question_id']} ranks {[p + 1 for p in group]}")
    print(f"measured: MPNet {len(archived) - len(problems)} of {len(archived)} questions match")

    assert problems == []


# The research's source-level metrics, copied from notebook 06 (cell 30); a test below checks
# the copies are unchanged. Used only to compare with the archived results.


def precision_at_k_source_level(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]
    unique_retrieved = list(dict.fromkeys(retrieved))
    relevant_retrieved = sum(1 for source in unique_retrieved if source in relevant_sources)
    return relevant_retrieved / k


def recall_at_k_source_level(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]
    unique_retrieved = set(retrieved)
    relevant_retrieved = len(unique_retrieved.intersection(relevant_sources))
    return relevant_retrieved / len(relevant_sources)


def reciprocal_rank_source_level(retrieved_sources, relevant_sources):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def ndcg_at_k_source_level(retrieved_sources, relevant_sources, k):
    retrieved = retrieved_sources[:k]
    seen_sources = set()
    relevance_scores = []
    for source in retrieved:
        if source in relevant_sources and source not in seen_sources:
            relevance_scores.append(1)
            seen_sources.add(source)
        else:
            relevance_scores.append(0)
    dcg = 0.0
    for rank, relevance in enumerate(relevance_scores, start=1):
        dcg += relevance / math.log2(rank + 1)
    ideal_relevant = min(len(relevant_sources), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


METRIC_FUNCTIONS = (
    precision_at_k_source_level,
    recall_at_k_source_level,
    reciprocal_rank_source_level,
    ndcg_at_k_source_level,
)


def test_metric_copies_match_notebook_06():
    with NOTEBOOK_06.open("r", encoding="utf-8") as file:
        cell = "".join(json.load(file)["cells"][30]["source"])
    notebook = {n.name: ast.dump(n) for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef)}
    source = Path(__file__).read_text(encoding="utf-8")
    copies = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}

    for function in METRIC_FUNCTIONS:
        assert copies[function.__name__] == notebook[function.__name__]


def research_metrics(sources: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "precision_at_5": precision_at_k_source_level(sources, relevant, 5),
        "recall_at_5": recall_at_k_source_level(sources, relevant, 5),
        "mrr": reciprocal_rank_source_level(sources, relevant),
        "ndcg_at_5": ndcg_at_k_source_level(sources, relevant, 5),
    }


def exact_tie_orderings(results: list[dict], k: int = 5) -> tuple[list[list[dict]], set]:
    """The tie groups touching the top k, and every top-k list of source files obtainable by
    reordering results whose scores are exactly equal. Without exact ties there is just one:
    our own order."""
    groups: list[list[dict]] = []
    for result in results:
        if groups and groups[-1][-1]["score"] == result["score"]:
            groups[-1].append(result)
        else:
            groups.append([result])
    touching, position = [], 0
    for group in groups:
        if position >= k:
            break
        touching.append(group)
        position += len(group)
    # A group reaching the end of what was retrieved might continue beyond it.
    assert position < len(results), "an exact tie group may extend past the retrieved results"
    orderings = {
        tuple(r["sourceFile"] for group in choice for r in group)[:k]
        for choice in itertools.product(*(itertools.permutations(g) for g in touching))
    }
    return touching, orderings


def metrics_match(ours: dict[str, float], row: dict[str, str]) -> bool:
    return all(
        abs(value - float(row[metric])) <= METRIC_TOLERANCE for metric, value in ours.items()
    )


@pytest.mark.models
@needs_data
@pytest.mark.parametrize("name", ["MiniLM", "BGE", "MPNet", "E5"])
def test_research_metrics_are_reproduced_for_every_question(real_api, name):
    # Our order puts exactly tied chunks lower row first (PRD 2.3); the research's FAISS order
    # may differ there. A row passes if our own order, or a reordering within exactly tied
    # scores only, gives the archived values. Without exact ties only our own order counts.
    with (RESULTS / "formal_rag_evaluation_data.json").open("r", encoding="utf-8") as file:
        relevant = {item["question_id"]: set(item["relevant_sources"]) for item in json.load(file)}
    with (RESULTS / "corrected_retrieval_results.csv").open("r", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["model"] == name]
    problems, reordered = [], []
    for row in rows:
        question_id = int(row["question_id"])
        results = retrieve_real(row["question"], name, top_k=20)
        own_order = [r["sourceFile"] for r in results[:5]]
        if metrics_match(research_metrics(own_order, relevant[question_id]), row):
            continue
        groups, orderings = exact_tie_orderings(results)
        matching = [
            o
            for o in orderings
            if metrics_match(research_metrics(list(o), relevant[question_id]), row)
        ]
        if not matching:
            problems.append(
                f"question {question_id}: no order within exact ties gives the archived values; "
                f"ours gives {research_metrics(own_order, relevant[question_id])}"
            )
            continue
        tied = [
            [(r["rank"], r["sourceFile"], r["chunkId"], r["score"]) for r in group]
            for group in groups
            if len(group) > 1
        ]
        reordered.append(
            f"question {question_id}: archived values need an exact tie reordered: {tied}; "
            f"{len(matching)} of {len(orderings)} orders within the ties match, "
            f"e.g. {list(sorted(matching)[0])}"
        )
    print(f"measured: {name} {len(rows) - len(problems)} of {len(rows)} questions reproduce")
    for line in reordered:
        print(f"measured: {name} {line}")

    assert len(rows) == 10
    assert problems == []
