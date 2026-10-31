import pytest
from pydantic import ValidationError

from medquery.embeddings import (
    BASELINE_EMBEDDING_SPECS,
    EmbeddingCompatibilityError,
    EmbeddingModelSpec,
    check_compatibility,
    ensure_compatible,
)

EXPECTED_BASELINE = {
    "MiniLM": ("sentence-transformers/all-MiniLM-L6-v2", 384, "", "", 32),
    "MPNet": ("sentence-transformers/all-mpnet-base-v2", 768, "", "", 16),
    "E5": ("intfloat/multilingual-e5-base", 768, "query: ", "passage: ", 16),
    "BGE": ("BAAI/bge-small-en-v1.5", 384, "", "", 32),
}


def test_baseline_specs_match_notebooks() -> None:
    assert set(BASELINE_EMBEDDING_SPECS) == set(EXPECTED_BASELINE)
    for name, (model_id, dim, query_prefix, passage_prefix, batch) in EXPECTED_BASELINE.items():
        spec = BASELINE_EMBEDDING_SPECS[name]
        assert (spec.name, spec.model_id, spec.dimension) == (name, model_id, dim)
        assert (spec.query_prefix, spec.passage_prefix) == (query_prefix, passage_prefix)
        assert spec.batch_size == batch
        assert spec.normalize is True


def test_historical_revisions_are_unknown_not_invented() -> None:
    assert all(spec.revision is None for spec in BASELINE_EMBEDDING_SPECS.values())


def test_baseline_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        BASELINE_EMBEDDING_SPECS["X"] = BASELINE_EMBEDDING_SPECS["BGE"]  # type: ignore[index]
    with pytest.raises(ValidationError):
        BASELINE_EMBEDDING_SPECS["BGE"].dimension = 1  # type: ignore[misc]


def test_experimental_spec_is_derived_without_touching_baseline() -> None:
    baseline = BASELINE_EMBEDDING_SPECS["BGE"]
    experiment = baseline.model_copy(
        update={"query_prefix": "Represent this sentence for searching relevant passages: "}
    )
    assert experiment.fingerprint() != baseline.fingerprint()
    assert baseline.query_prefix == ""


@pytest.mark.parametrize("kwargs", [{"dimension": 0}, {"batch_size": 0}, {"model_id": ""}])
def test_invalid_specs_are_rejected(kwargs: dict) -> None:
    base = {"name": "M", "model_id": "org/m", "dimension": 8}
    with pytest.raises(ValidationError):
        EmbeddingModelSpec(**{**base, **kwargs})


def test_batch_size_is_not_part_of_the_fingerprint() -> None:
    spec = BASELINE_EMBEDDING_SPECS["MPNet"]
    assert spec.model_copy(update={"batch_size": 128}).fingerprint() == spec.fingerprint()


# --- Compatibility --------------------------------------------------------------------


def fp(**changes):
    return BASELINE_EMBEDDING_SPECS["E5"].model_copy(update=changes).fingerprint()


def test_same_spec_is_compatible_but_unknown_revision_is_flagged() -> None:
    report = check_compatibility(fp(), fp())
    assert report.ok
    assert any("revision unknown" in w for w in report.warnings)


def test_pinned_identical_revisions_give_no_warning() -> None:
    report = check_compatibility(fp(revision="a" * 40), fp(revision="a" * 40))
    assert report.ok and report.warnings == ()


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "intfloat/multilingual-e5-large"},
        {"dimension": 1024},
        {"normalize": False},
        {"query_prefix": ""},
        {"passage_prefix": ""},
    ],
)
def test_incompatible_properties_are_errors(change: dict) -> None:
    report = check_compatibility(fp(), fp(**change))
    assert not report.ok
    assert next(iter(change)) in report.errors[0]


def test_different_pinned_revisions_are_an_error() -> None:
    assert not check_compatibility(fp(revision="a" * 40), fp(revision="b" * 40)).ok


def test_ensure_compatible_raises_on_mismatch() -> None:
    with pytest.raises(EmbeddingCompatibilityError, match="dimension"):
        ensure_compatible(fp(), fp(dimension=384))
    assert ensure_compatible(fp(), fp()).ok


def test_different_baseline_models_are_never_compatible() -> None:
    minilm = BASELINE_EMBEDDING_SPECS["MiniLM"].fingerprint()
    bge = BASELINE_EMBEDDING_SPECS["BGE"].fingerprint()  # same dimension (384)
    assert not check_compatibility(minilm, bge).ok
