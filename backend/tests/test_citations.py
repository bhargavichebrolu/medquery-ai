"""Tests for the citation checks and grounding coverage (PRD 3.4) and grounding status (PRD 3.6).

Pure functions over hand-written answers and documents: no data, models, network or cost.
"""

import pytest

from medquery_api.llm import (
    AnswerPart,
    Citation,
    GenerationRequest,
    GenerationResult,
    MockProvider,
    SourceDocument,
    TokenUsage,
)
from medquery_api.llm.citations import (
    INVALID_RANGE,
    TEXT_MISMATCH,
    UNKNOWN_DOCUMENT,
    check_citations,
    citation_problem,
    grounding_status,
    sentence_spans,
)
from medquery_api.llm.mock import NOT_ENOUGH_EVIDENCE

ASTHMA = "Asthma is a common lung condition.\nSymptoms include wheezing and breathlessness."
HIVES = "Hives is a raised, itchy rash."
DOCUMENTS = (SourceDocument("Asthma (chunk 1 of 2)", ASTHMA), SourceDocument("Hives", HIVES))


def cite(passage, document_index=0, text=ASTHMA):
    start = text.index(passage)
    return Citation(document_index, start, start + len(passage), passage)


def sentences(text):
    return [text[start:end].strip() for start, end in sentence_spans(text)]


# --- Each citation ---


def test_a_citation_quoting_its_span_exactly_is_valid():
    assert citation_problem(cite("wheezing and breathlessness"), DOCUMENTS) is None
    assert citation_problem(cite("raised, itchy", 1, HIVES), DOCUMENTS) is None


def test_the_whole_document_is_a_valid_span():
    whole = Citation(1, 0, len(HIVES), HIVES)

    assert citation_problem(whole, DOCUMENTS) is None


@pytest.mark.parametrize(
    "document_index",
    [2, -1, -2, 99, None, True, 0.0, "0"],
    ids=["one past the end", "-1", "-2", "99", "None", "True", "float", "text"],
)
def test_a_citation_must_name_a_supplied_document(document_index):
    citation = Citation(document_index, 0, 6, "Asthma")

    assert citation_problem(citation, DOCUMENTS) == UNKNOWN_DOCUMENT


def test_no_documents_means_every_citation_is_unknown():
    assert citation_problem(Citation(0, 0, 6, "Asthma"), ()) == UNKNOWN_DOCUMENT


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0, len(HIVES) + 1),
        (-1, 5),
        (5, 5),
        (6, 2),
        (len(HIVES), len(HIVES) + 3),
        (None, 5),
        (0, None),
        (0.0, 5),
        (False, 5),
        ("0", "5"),
    ],
    ids=[
        "end past the text",
        "negative start",
        "empty",
        "reversed",
        "starts at the end",
        "no start",
        "no end",
        "float",
        "bool",
        "text",
    ],
)
def test_a_citation_range_must_lie_inside_the_document(start, end):
    citation = Citation(1, start, end, "Hives")

    assert citation_problem(citation, DOCUMENTS) == INVALID_RANGE


@pytest.mark.parametrize(
    "cited_text",
    ["Hive", "Hives ", "hives", "Hives  is", "", None],
    ids=["off by one", "extra space", "case", "doubled space", "empty", "None"],
)
def test_the_quoted_text_must_match_the_span_exactly(cited_text):
    citation = Citation(1, 0, 5, cited_text)

    assert citation_problem(citation, DOCUMENTS) == TEXT_MISMATCH


def test_the_quoted_text_must_come_from_the_cited_document_not_another():
    elsewhere = Citation(1, 0, 6, "Asthma")  # true of document 0, not of document 1

    assert citation_problem(elsewhere, DOCUMENTS) == TEXT_MISMATCH


def test_the_first_failing_check_gives_the_reason():
    assert citation_problem(Citation(5, -1, 0, "x"), DOCUMENTS) == UNKNOWN_DOCUMENT
    assert citation_problem(Citation(0, -1, 0, "x"), DOCUMENTS) == INVALID_RANGE


def test_positions_count_characters_even_beyond_the_basic_plane():
    text = "Rash 🙂 spreads."
    documents = (SourceDocument("Rash", text),)

    assert citation_problem(cite("spreads", text=text), documents) is None
    assert text.index("spreads") == 7  # one position per character, as Python counts


# --- A whole answer ---


def test_invalid_citations_are_dropped_and_counted_and_valid_ones_kept_in_order():
    valid_a, valid_b = cite("Asthma is"), cite("wheezing")
    unknown, mismatch = Citation(4, 0, 6, "Asthma"), Citation(0, 0, 6, "asthma")
    parts = (
        AnswerPart("Asthma is a lung condition.", (valid_a, unknown)),
        AnswerPart(" It causes wheezing.", (mismatch, valid_b)),
    )

    check = check_citations(parts, DOCUMENTS)

    assert [p.citations for p in check.parts] == [(valid_a,), (valid_b,)]
    assert [p.text for p in check.parts] == [p.text for p in parts]
    assert [(d.part_index, d.citation, d.reason) for d in check.dropped] == [
        (0, unknown, UNKNOWN_DOCUMENT),
        (1, mismatch, TEXT_MISMATCH),
    ]
    assert (check.total, check.valid, check.validity_rate) == (4, 2, 0.5)
    assert check.dropped_by_reason == {UNKNOWN_DOCUMENT: 1, INVALID_RANGE: 0, TEXT_MISMATCH: 1}


def test_duplicate_and_overlapping_citations_are_each_checked_and_kept():
    wide, narrow = cite("wheezing and breathlessness"), cite("wheezing")
    parts = (AnswerPart("Wheezing and breathlessness.", (wide, wide, narrow)),)

    check = check_citations(parts, DOCUMENTS)

    assert check.parts[0].citations == (wide, wide, narrow)
    assert (check.total, check.valid) == (3, 3)


def test_an_answer_without_citations_has_no_validity_rate_and_no_coverage():
    check = check_citations((AnswerPart("Asthma is common. It is treatable."),), DOCUMENTS)

    assert (check.total, check.validity_rate) == (0, None)
    assert (check.sentences, check.supported_sentences, check.grounding_coverage) == (2, 0, 0.0)


def test_an_empty_answer_has_no_coverage():
    check = check_citations((), DOCUMENTS)

    assert (check.sentences, check.grounding_coverage, check.validity_rate) == (0, None, None)


def test_the_mock_providers_answer_passes_every_check():
    request = GenerationRequest(system="", documents=DOCUMENTS, question="What is asthma?")
    result = MockProvider().generate(request)

    check = check_citations(result.parts, DOCUMENTS)

    assert (check.valid, check.total, check.grounding_coverage) == (1, 1, 1.0)


# --- Grounding coverage ---


def test_a_sentence_is_supported_when_a_cited_part_falls_inside_it():
    # Claude often cites a phrase in the middle of a sentence.
    parts = (
        AnswerPart("Asthma "),
        AnswerPart("causes wheezing and breathlessness", (cite("wheezing and breathlessness"),)),
        AnswerPart(". It can be serious."),
    )

    check = check_citations(parts, DOCUMENTS)

    assert (check.sentences, check.supported_sentences, check.grounding_coverage) == (2, 1, 0.5)


def test_a_cited_part_does_not_support_the_next_sentence_through_whitespace():
    parts = (
        AnswerPart("Asthma is a common lung condition. ", (cite("Asthma is a common"),)),
        AnswerPart("It is caused by cats."),
    )

    check = check_citations(parts, DOCUMENTS)

    assert (check.sentences, check.supported_sentences) == (2, 1)


def test_one_cited_part_can_support_several_sentences():
    cited = AnswerPart("Asthma is common. It causes wheezing.", (cite("wheezing"),))

    assert check_citations((cited,), DOCUMENTS).grounding_coverage == 1.0


def test_dropped_citations_do_not_support_a_sentence():
    parts = (AnswerPart("Asthma is caused by cats.", (Citation(0, 0, 6, "Cats!!"),)),)

    check = check_citations(parts, DOCUMENTS)

    assert (check.sentences, check.supported_sentences, check.grounding_coverage) == (1, 0, 0.0)


def test_bullet_items_are_sentences_and_bullet_markers_are_not():
    parts = (
        AnswerPart("Symptoms include:\n- "),
        AnswerPart("wheezing", (cite("wheezing"),)),
        AnswerPart("\n- a tight chest\n1. Coughing at night"),
    )

    check = check_citations(parts, DOCUMENTS)

    assert sentences("".join(part.text for part in parts)) == [
        "Symptoms include:",
        "- wheezing",
        "- a tight chest",
        "Coughing at night",  # "1." alone has no letter, so it is not a sentence
    ]
    assert (check.sentences, check.supported_sentences) == (4, 1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three? Four", ["One.", "Two!", "Three?", "Four"]),
        ('Call "999." Then wait.', ['Call "999."', "Then wait."]),
        ("Take 2.5 ml (see leaflet.) Then rest.", ["Take 2.5 ml (see leaflet.)", "Then rest."]),
        ("Rest...  Drink water.", ["Rest...", "Drink water."]),
        ("## Symptoms\nWheezing.", ["Wheezing."]),
        ("1.\n-\n**\nWheezing.", ["Wheezing."]),
        ("Symptoms, e.g. wheezing.", ["Symptoms, e.g.", "wheezing."]),  # a known limitation
        ("", []),
        ("   \n\n  ", []),
    ],
    ids=[
        "punctuation",
        "closing quote",
        "decimal and bracket",
        "ellipsis",
        "heading",
        "no letters",
        "abbreviation",
        "empty",
        "blank",
    ],
)
def test_sentences_are_split_deterministically(text, expected):
    assert sentences(text) == expected


# --- Sentence positions (PRD 3.6) ---


def test_each_sentence_has_its_position_in_the_answer_and_whether_it_is_supported():
    parts = (
        AnswerPart("Asthma "),
        AnswerPart("causes wheezing", (cite("wheezing"),)),
        AnswerPart(".  It is caused by cats. "),
    )
    text = "".join(part.text for part in parts)

    check = check_citations(parts, DOCUMENTS)

    assert [(text[s.start : s.end], s.supported) for s in check.sentence_list] == [
        ("Asthma causes wheezing.", True),
        ("It is caused by cats.", False),
    ]


def test_sentence_positions_leave_out_surrounding_whitespace_and_bullet_lines_keep_markers():
    text = "  Symptoms include:\n\n- wheezing  \n"

    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "Symptoms include:",
        "- wheezing",
    ]


# --- Grounding status (PRD 3.6) ---


def answer(*parts, stop_reason="end_turn"):
    return GenerationResult(
        provider="mock",
        model="mock",
        parts=parts,
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        latency_ms=0.0,
    )


def status(*parts, stop_reason="end_turn"):
    result = answer(*parts, stop_reason=stop_reason)
    return grounding_status(result, check_citations(result.parts, DOCUMENTS))


def test_an_answer_whose_every_sentence_is_cited_is_grounded():
    assert status(AnswerPart("Asthma causes wheezing.", (cite("wheezing"),))) == "grounded"


def test_an_answer_with_some_uncited_sentences_is_partly_grounded():
    parts = (
        AnswerPart("Asthma causes wheezing.", (cite("wheezing"),)),
        AnswerPart(" Cats cure it."),
    )

    assert status(*parts) == "partly_grounded"


def test_an_answer_without_valid_citations_is_never_grounded():
    assert status(AnswerPart("Asthma is caused by cats.")) == "not_grounded"
    invalid_only = AnswerPart("Asthma is caused by cats.", (Citation(0, 0, 6, "Cats!!"),))
    assert status(invalid_only) == "not_grounded"
    assert status() == "not_grounded"  # no text at all
    assert status(AnswerPart("   ")) == "not_grounded"


@pytest.mark.parametrize(
    "text",
    [NOT_ENOUGH_EVIDENCE, f"\n  {NOT_ENOUGH_EVIDENCE}  \n"],
    ids=["exact", "surrounding whitespace"],
)
def test_the_required_not_enough_evidence_sentence_is_recognised(text):
    assert status(AnswerPart(text)) == "not_enough_evidence"


def test_the_sentence_split_across_parts_is_still_recognised():
    half = len(NOT_ENOUGH_EVIDENCE) // 2
    parts = (AnswerPart(NOT_ENOUGH_EVIDENCE[:half]), AnswerPart(NOT_ENOUGH_EVIDENCE[half:]))

    assert status(*parts) == "not_enough_evidence"


@pytest.mark.parametrize(
    "text",
    [
        f'"{NOT_ENOUGH_EVIDENCE}"',
        NOT_ENOUGH_EVIDENCE.lower(),
        NOT_ENOUGH_EVIDENCE.rstrip("."),
        f"{NOT_ENOUGH_EVIDENCE} Asthma is caused by cats.",
    ],
    ids=["quoted", "lower case", "no full stop", "with more text"],
)
def test_anything_other_than_the_exact_sentence_is_judged_by_its_citations(text):
    assert status(AnswerPart(text)) == "not_grounded"


def test_a_refusal_is_reported_as_refused_whatever_text_came_before_it():
    cited = AnswerPart("Asthma causes wheezing.", (cite("wheezing"),))

    assert status(cited, stop_reason="refusal") == "refused"
    assert status(AnswerPart(NOT_ENOUGH_EVIDENCE), stop_reason="refusal") == "refused"
    assert status(stop_reason="refusal") == "refused"
