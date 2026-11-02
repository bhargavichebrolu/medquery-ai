"""Tests for medquery_api.processing: cleaning and chunking articles.

The unit tests use small hand-written texts and always run. The archive tests compare the
backend's output with Healthcare_RAG/data (or MEDQUERY_DATA_DIR), only reading it, and are
skipped, with the reason shown, when it is absent. Run with `-rP` to see the measured counts.
"""

import json
from pathlib import Path

import pytest

from medquery_api import config
from medquery_api.processing import Chunk, chunk_article, clean_text, split_text

DATA = config.data_dir()
CHUNK_FIELDS = ("chunk_id", "source_file", "chunk_length", "text")

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)


# --- Cleaning ---


def test_cleaning_trims_lines_collapses_whitespace_and_separates_lines():
    text = "  # Title  \n\n\nSome   text\twith  gaps \n"

    assert clean_text(text) == "# Title\n\nSome text with gaps"


def test_cleaning_drops_a_repeated_line_anywhere_in_the_article():
    assert clean_text("A\nB\nC\nA\nD") == "A\n\nB\n\nC\n\nD"


def test_cleaning_compares_lines_after_collapsing_whitespace():
    assert clean_text("x  y\nx y") == "x y"


def test_cleaning_uses_python_line_breaks_and_whitespace():
    assert clean_text("A\r\nB\r\n") == "A\n\nB"  # Windows line endings
    assert clean_text("A B") == "A\n\nB"  # Unicode line separator
    assert clean_text("a  b") == "a b"  # non-breaking spaces


def test_cleaning_an_empty_article_gives_empty_text():
    assert clean_text("") == ""
    assert clean_text("   \n\n \t ") == ""


# --- Splitting ---


def test_a_short_text_is_one_stripped_chunk():
    assert split_text("  Hello world.  ") == ["Hello world."]


def test_headings_are_preferred_break_points_and_start_the_next_chunk():
    text = "# T\n\nIntro para.\n## A\n\nAaa.\n## B\n\nBbb."

    assert split_text(text, chunk_size=20, chunk_overlap=0) == [
        "# T\n\nIntro para.",
        "## A\n\nAaa.",
        "## B\n\nBbb.",
    ]


def test_a_new_chunk_repeats_whole_pieces_that_fit_in_the_overlap():
    chunks = split_text("aaa bbb ccc ddd eee", chunk_size=11, chunk_overlap=4)

    assert chunks == ["aaa bbb ccc", "ccc ddd", "ddd eee"]


def test_nothing_is_repeated_when_the_last_piece_is_longer_than_the_overlap():
    chunks = split_text("aaaaaa bbbbbb cccccc", chunk_size=13, chunk_overlap=4)

    assert chunks == ["aaaaaa bbbbbb", "cccccc"]


def test_a_word_longer_than_a_chunk_is_cut_into_characters():
    assert split_text("x" * 25, chunk_size=10, chunk_overlap=0) == ["x" * 10, "x" * 10, "x" * 5]


def test_chunks_never_exceed_the_size_or_start_or_end_with_whitespace():
    text = "\n\n".join(
        f"## Section {n}\n\n" + " ".join(f"Sentence {n}.{m} has some words." for m in range(40))
        for n in range(5)
    )

    chunks = split_text(text, chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 5
    for chunk in chunks:
        assert 0 < len(chunk) <= 300
        assert chunk == chunk.strip()


def test_empty_or_blank_text_gives_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


@pytest.mark.parametrize(
    "settings",
    [
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 101},
        {"separators": ()},
    ],
)
def test_settings_that_make_no_sense_are_rejected(settings):
    with pytest.raises(ValueError):
        split_text("some text", **settings)


def test_an_overlap_equal_to_the_chunk_size_is_allowed_as_in_langchain():
    assert split_text("aaa bbb", chunk_size=4, chunk_overlap=4) == ["aaa", "bbb"]


# --- Chunk records ---


def test_chunk_records_are_numbered_from_one_in_the_archive_field_order():
    chunks = chunk_article("aaa bbb ccc ddd eee", "x.txt", chunk_size=11, chunk_overlap=4)

    assert chunks == [
        Chunk(chunk_id=1, source_file="x.txt", chunk_length=11, text="aaa bbb ccc"),
        Chunk(chunk_id=2, source_file="x.txt", chunk_length=7, text="ccc ddd"),
        Chunk(chunk_id=3, source_file="x.txt", chunk_length=7, text="ddd eee"),
    ]
    assert tuple(chunks[0].as_record()) == CHUNK_FIELDS


# --- Archive comparisons (read only) ---


def read_text(path: Path, *, exactly: bool) -> str:
    """Read a text file: as notebooks 02/03 did (text mode), or exactly as stored."""
    with path.open("r", encoding="utf-8", newline="" if exactly else None) as file:
        return file.read()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def chunk_files() -> list[Path]:
    return sorted((DATA / "chunks").glob("*.json"), key=lambda path: path.name)


def describe_difference(ours: list[dict], archived: list[dict]) -> str:
    """Where two lists of chunk records first differ, without printing whole texts."""
    for i, (mine, theirs) in enumerate(zip(ours, archived, strict=False), start=1):
        if mine != theirs:
            fields = [field for field in CHUNK_FIELDS if mine.get(field) != theirs.get(field)]
            a, b = mine.get("text", ""), theirs.get("text", "")
            pairs = enumerate(zip(a, b, strict=False))
            position = next((j for j, (x, y) in pairs if x != y), min(len(a), len(b)))
            return (
                f"chunk {i}: fields {fields} differ; backend length {len(a)}, archive {len(b)}; "
                f"texts first differ at character {position}"
            )
    return f"backend made {len(ours)} chunks, the archive has {len(archived)}"


@needs_data
def test_cleaning_reproduces_every_archived_cleaned_article():
    raw_files = sorted((DATA / "raw").glob("*.txt"))
    different = [
        path.name
        for path in raw_files
        if clean_text(read_text(path, exactly=False))
        != read_text(DATA / "cleaned" / path.name, exactly=True)
    ]
    print(f"measured: {len(raw_files) - len(different)} of {len(raw_files)} identical")

    assert len(raw_files) == 698
    assert different == []


@needs_data
def test_chunking_reproduces_every_archived_chunk_record():
    articles = chunk_files()
    matched_articles, matched_chunks, total_chunks, mismatches = 0, 0, 0, []
    for path in articles:
        archived = read_json(path)
        cleaned = read_text(DATA / "cleaned" / f"{path.stem}.txt", exactly=True)
        ours = [chunk.as_record() for chunk in chunk_article(cleaned, f"{path.stem}.txt")]
        total_chunks += len(archived)
        matched_chunks += sum(1 for a, b in zip(ours, archived, strict=False) if a == b)
        if ours == archived:
            matched_articles += 1
        else:
            mismatches.append(f"{path.name}: {describe_difference(ours, archived)}")
    print(
        f"measured: {matched_articles} of {len(articles)} articles and {matched_chunks} of "
        f"{total_chunks} chunk records identical (order, id, file, length, text)"
    )

    assert mismatches == []
    assert (matched_articles, matched_chunks) == (698, 3796)


@needs_data
def test_raw_articles_become_the_archived_chunks_end_to_end():
    mismatches = []
    for path in chunk_files():
        raw = read_text(DATA / "raw" / f"{path.stem}.txt", exactly=False)
        ours = [chunk.as_record() for chunk in chunk_article(clean_text(raw), f"{path.stem}.txt")]
        archived = read_json(path)
        if ours != archived:
            mismatches.append(f"{path.name}: {describe_difference(ours, archived)}")

    assert mismatches == []


@needs_data
def test_diagnostic_chunk_records_serialise_to_the_archived_json_files():
    # Diagnostic only: the chunking criterion is the record comparison above. This checks that
    # writing the records the way notebook 03 did (json.dump, indent=4, ensure_ascii=False)
    # also reproduces each file's bytes.
    different = []
    for path in chunk_files():
        cleaned = read_text(DATA / "cleaned" / f"{path.stem}.txt", exactly=True)
        records = [chunk.as_record() for chunk in chunk_article(cleaned, f"{path.stem}.txt")]
        serialised = json.dumps(records, indent=4, ensure_ascii=False).encode("utf-8")
        if serialised != path.read_bytes():
            different.append(path.name)
    print(f"measured: {698 - len(different)} of 698 chunk files identical byte for byte")

    assert different == []
