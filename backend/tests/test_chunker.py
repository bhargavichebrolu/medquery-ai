import pytest
from pydantic import ValidationError

from medquery.ingestion import Chunk, ChunkingConfig, Document, chunk_document
from medquery.ingestion.chunker import DEFAULT_SEPARATORS


def make_doc(text: str, **metadata: str) -> Document:
    return Document(source_id="article.txt", text=text, metadata=metadata)


def section(title: str, sentences: int) -> str:
    body = " ".join(f"{title} sentence number {i} has some words." for i in range(sentences))
    return f"## {title}\n\n{body}"


# --- Configuration --------------------------------------------------------------------


def test_defaults_match_research_notebook() -> None:
    config = ChunkingConfig()
    assert (config.chunk_size, config.chunk_overlap) == (1000, 200)
    assert config.separators == ("\n## ", "\n### ", "\n\n", "\n", ". ", " ", "")
    assert config.strategy == "recursive_character"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -5},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 150},
        {"separators": ()},
        {"strategy": "semantic"},
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(**kwargs)


# --- Empty input ----------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_empty_or_whitespace_documents_produce_no_chunks(text: str) -> None:
    assert chunk_document(make_doc(text)) == []


# --- Boundaries -----------------------------------------------------------------------


def test_short_document_is_a_single_chunk() -> None:
    [chunk] = chunk_document(make_doc("# Title\n\nShort body."))
    assert (chunk.text, chunk.chunk_id, chunk.start_index) == ("# Title\n\nShort body.", 1, 0)


@pytest.mark.parametrize(("size", "overlap"), [(1000, 200), (300, 50), (120, 0), (80, 30)])
def test_chunks_respect_size_and_map_back_to_source(size: int, overlap: int) -> None:
    text = "\n\n".join(section(f"S{i}", 12) for i in range(6))
    chunks = chunk_document(make_doc(text), ChunkingConfig(chunk_size=size, chunk_overlap=overlap))

    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == list(range(1, len(chunks) + 1))
    for chunk in chunks:
        assert chunk.chunk_length <= size
        assert text[chunk.start_index : chunk.start_index + chunk.chunk_length] == chunk.text
    # Offsets are monotonic and every word of the source is covered by some chunk.
    starts = [c.start_index for c in chunks]
    assert starts == sorted(starts)
    covered = {word for c in chunks for word in c.text.split()}
    assert set(text.split()) <= covered


def test_overlap_is_bounded_and_zero_overlap_means_disjoint_spans() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    for overlap in (0, 40):
        chunks = chunk_document(
            make_doc(text), ChunkingConfig(chunk_size=150, chunk_overlap=overlap)
        )
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            shared = prev.start_index + prev.chunk_length - nxt.start_index
            assert shared <= overlap


def test_splits_prefer_section_headings() -> None:
    text = section("Symptoms", 8) + "\n\n" + section("Causes", 8)
    assert len(section("Symptoms", 8)) < 600 < len(text)
    chunks = chunk_document(make_doc(text), ChunkingConfig(chunk_size=600, chunk_overlap=100))
    assert [c.text.splitlines()[0] for c in chunks] == ["## Symptoms", "## Causes"]


def test_unbreakable_text_falls_back_to_character_splitting() -> None:
    chunks = chunk_document(make_doc("x" * 250), ChunkingConfig(chunk_size=100, chunk_overlap=0))
    assert [c.chunk_length for c in chunks] == [100, 100, 50]


def test_without_character_fallback_chunks_may_exceed_size() -> None:
    """Documented limitation: custom separators without "" cannot guarantee chunk_size."""
    config = ChunkingConfig(chunk_size=100, chunk_overlap=0, separators=("\n\n",))
    [chunk] = chunk_document(make_doc("y" * 250), config)
    assert chunk.chunk_length == 250


def test_smaller_chunk_size_produces_more_chunks() -> None:
    doc = make_doc("\n\n".join(section(f"S{i}", 10) for i in range(5)))
    big = chunk_document(doc, ChunkingConfig(chunk_size=1000, chunk_overlap=200))
    small = chunk_document(doc, ChunkingConfig(chunk_size=300, chunk_overlap=50))
    assert len(small) > len(big)


# --- Identifiers and metadata ---------------------------------------------------------


def test_identifiers_and_metadata_are_preserved_on_every_chunk() -> None:
    doc = make_doc(
        "\n\n".join(section(f"S{i}", 10) for i in range(4)),
        title="Asthma",
        url="https://www.nhs.uk/conditions/asthma/",
    )
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=300, chunk_overlap=50))
    assert len({c.chunk_uid for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.source_id == "article.txt"
        assert chunk.chunk_uid == f"article.txt#{chunk.chunk_id}"
        assert chunk.metadata == doc.metadata


def test_chunk_metadata_is_a_copy_not_shared() -> None:
    doc = make_doc(section("S", 30), title="T")
    first, second = chunk_document(doc, ChunkingConfig(chunk_size=200, chunk_overlap=0))[:2]
    assert first.metadata is not second.metadata
    assert first.metadata is not doc.metadata


def test_legacy_record_matches_research_artifact_format() -> None:
    [chunk] = chunk_document(Document(source_id="asthma.txt", text="# Asthma"))
    assert chunk.to_legacy_record() == {
        "chunk_id": 1,
        "source_file": "asthma.txt",
        "chunk_length": 8,
        "text": "# Asthma",
    }


def test_chunk_model_rejects_inconsistent_fields() -> None:
    with pytest.raises(ValidationError):
        Chunk(chunk_uid="a#1", source_id="a", chunk_id=1, text="abc", chunk_length=2, start_index=0)
    with pytest.raises(ValidationError):
        Chunk(chunk_uid="b#1", source_id="a", chunk_id=1, text="abc", chunk_length=3, start_index=0)


def test_default_separators_constant_is_immutable() -> None:
    assert isinstance(DEFAULT_SEPARATORS, tuple)
