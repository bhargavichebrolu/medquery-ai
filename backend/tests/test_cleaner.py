import pytest

from medquery.ingestion import CleaningConfig, Document, MalformedDocumentError
from medquery.ingestion.cleaner import clean_document, clean_text

# --- Empty and whitespace-only input -------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\n\n", " \t \r\n     ", "﻿"])
def test_empty_or_whitespace_only_input_yields_empty_string(text: str) -> None:
    assert clean_text(text) == ""


# --- Whitespace handling --------------------------------------------------------------


def test_lines_are_stripped_blank_lines_dropped_and_rejoined() -> None:
    assert clean_text("  first line  \n\n\n   \nsecond line\n") == "first line\n\nsecond line"


def test_internal_whitespace_runs_collapse_including_tabs_and_no_break_spaces() -> None:
    assert clean_text("a \t b  c d") == "a b c d"


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", "\f", " "])
def test_all_line_break_conventions_split_lines(newline: str) -> None:
    assert clean_text(f"one{newline}two") == "one\n\ntwo"


def test_whitespace_collapsing_can_be_disabled() -> None:
    assert clean_text("a  b", CleaningConfig(collapse_whitespace=False)) == "a  b"


# --- Structure preservation -----------------------------------------------------------


def test_headings_and_bullets_are_preserved() -> None:
    raw = "# Asthma\n\nIntro.\n\n\n## Symptoms\n\n• wheezing\n\n• coughing"
    assert clean_text(raw) == "# Asthma\n\nIntro.\n\n## Symptoms\n\n• wheezing\n\n• coughing"


def test_duplicate_lines_removed_by_default_and_optional() -> None:
    raw = "## A\n• fever\n## B\n• fever"
    assert clean_text(raw) == "## A\n\n• fever\n\n## B"
    assert clean_text(raw, CleaningConfig(deduplicate_lines=False)).count("• fever") == 2


def test_duplicates_are_detected_after_whitespace_normalisation() -> None:
    assert clean_text("same  line\nsame line") == "same line"


def test_cleaning_is_idempotent() -> None:
    raw = "# T\n\n  body \t text \n\n## H\n\n• x\n\n• x\n "
    once = clean_text(raw)
    assert clean_text(once) == once


# --- Malformed / untrusted input ------------------------------------------------------


def test_byte_order_mark_and_control_characters_are_removed() -> None:
    assert clean_text("﻿abc\x00def\x07\x1b") == "abcdef"


def test_control_character_removal_can_be_disabled_but_bom_is_still_stripped() -> None:
    config = CleaningConfig(remove_control_characters=False)
    assert clean_text("﻿a\x00b", config) == "a\x00b"


def test_unicode_is_nfc_normalised_by_default() -> None:
    decomposed = "café"
    assert clean_text(decomposed) == "café"
    assert clean_text(decomposed, CleaningConfig(unicode_normalization=None)) == decomposed


def test_non_latin_text_is_preserved() -> None:
    assert clean_text("  درمان  \n\nçà") == "درمان\n\nçà"


@pytest.mark.parametrize("value", [None, b"bytes", 42, ["list"]])
def test_non_string_input_raises_type_error(value: object) -> None:
    with pytest.raises(TypeError):
        clean_text(value)  # type: ignore[arg-type]


def test_unpaired_surrogates_are_rejected() -> None:
    with pytest.raises(MalformedDocumentError):
        clean_text("bad \udc80 text")


# --- Document wrapper -----------------------------------------------------------------


def test_clean_document_preserves_identifier_and_metadata() -> None:
    doc = Document(source_id="asthma.txt", text="  a \n\n b ", metadata={"title": "Asthma"})
    cleaned = clean_document(doc)
    assert cleaned.text == "a\n\nb"
    assert cleaned.source_id == "asthma.txt"
    assert cleaned.metadata == {"title": "Asthma"}
    assert doc.text == "  a \n\n b "  # original untouched
