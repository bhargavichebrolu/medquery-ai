"""Tests for medquery_api.llm.context (PRD 3.2): retrieval results into an LLM request.

The unit tests build a tiny knowledge base in memory: no data folder, models or network. The
archive tests read Healthcare_RAG/data (or MEDQUERY_DATA_DIR) only, and are skipped, with the
reason shown, when it is absent.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from medquery_api import config
from medquery_api.embeddings import get_model
from medquery_api.knowledge_base import KnowledgeBase, SearchHit, load_knowledge_base
from medquery_api.llm import MockProvider
from medquery_api.llm.context import (
    SYSTEM_INSTRUCTIONS,
    article_catalogue,
    build_context,
    document_title,
    render_plain_text,
)
from medquery_api.llm.mock import NOT_ENOUGH_EVIDENCE
from medquery_api.processing import Chunk
from medquery_api.retrieval import Retrieval

DATA = config.data_dir()

needs_data = pytest.mark.skipif(
    not (DATA / "chunks").is_dir(),
    reason=f"research data not found at {DATA}; the archive checks run only where it exists "
    "(MEDQUERY_DATA_DIR can point to another folder)",
)

ASTHMA_1 = "# Asthma\n\nAsthma is a common condition that affects your breathing."
ASTHMA_2 = "## Symptoms of asthma\n\n• wheezing\n\nSee https://www.example.org/help for more."
ASTHMA_3 = "## Treatment\n\nUse your reliever inhaler."
HIVES_1 = "# Hives\n\nHives is a raised, itchy rash."


def chunk(source_file, chunk_id, text):
    return Chunk(chunk_id=chunk_id, source_file=source_file, chunk_length=len(text), text=text)


CHUNKS = [
    chunk("asthma.txt", 1, ASTHMA_1),
    chunk("asthma.txt", 2, ASTHMA_2),
    chunk("asthma.txt", 3, ASTHMA_3),
    chunk("hives.txt", 1, HIVES_1),
]


@pytest.fixture
def kb():
    return KnowledgeBase(Path("in-memory"), CHUNKS, indexes={})


def retrieval(*hits, question="What are the symptoms of asthma?"):
    return Retrieval(
        question=question,
        model=get_model("BGE"),
        top_k=len(hits) or 5,
        hits=list(hits),
        model_load_ms=0.0,
        embedding_ms=1.0,
        search_ms=0.1,
    )


def hit(rank, row, score):
    return SearchHit(rank=rank, row=row, score=score, chunk=CHUNKS[row])


# --- The request ---


def test_chunks_become_verbatim_documents_in_rank_order_with_titles(kb):
    context = build_context(retrieval(hit(1, 1, 0.83), hit(2, 3, 0.71), hit(3, 0, 0.70)), kb)
    documents = context.request.documents

    assert [d.text for d in documents] == [ASTHMA_2, HIVES_1, ASTHMA_1]
    assert [d.title for d in documents] == [
        "Asthma (chunk 2 of 3)",
        "Hives (chunk 1 of 1)",
        "Asthma (chunk 1 of 3)",
    ]
    assert context.request.question == "What are the symptoms of asthma?"
    assert context.request.system == SYSTEM_INSTRUCTIONS
    assert context.request.max_tokens == 4096


def test_results_are_ordered_by_rank_whatever_order_they_arrive_in(kb):
    context = build_context(retrieval(hit(2, 3, 0.71), hit(1, 1, 0.83)), kb)

    assert [s.rank for s in context.sources] == [1, 2]
    assert [d.text for d in context.request.documents] == [ASTHMA_2, HIVES_1]


def test_the_source_map_traces_each_document_to_its_chunk(kb):
    context = build_context(retrieval(hit(1, 1, 0.83), hit(2, 3, 0.71)), kb)

    first = context.source_for(0)
    assert (first.index, first.label, first.source_file, first.chunk_id) == (
        0,
        "S1",
        "asthma.txt",
        2,
    )
    assert (first.rank, first.score, first.article_title, first.chunk_count) == (
        1,
        0.83,
        "Asthma",
        3,
    )
    assert context.source_for(1).source_file == "hives.txt"
    with pytest.raises(IndexError, match="no source document at position 2"):
        context.source_for(2)
    with pytest.raises(IndexError):
        context.source_for(-1)


def test_the_instructions_carry_the_research_rules():
    for rule in (
        "using only the NHS documents provided",
        "cite the passages you rely on",
        "Do not add facts that the documents do not support",
        "Do not diagnose, and do not prescribe",
        "Do not present yourself as a doctor",
        "emergency or urgent-care guidance",
        "repeat it accurately and do not change it",
        f'reply with exactly this sentence and nothing else: "{NOT_ENOUGH_EVIDENCE}"',
    ):
        assert rule in SYSTEM_INSTRUCTIONS


def test_the_answer_length_can_be_set(kb):
    assert build_context(retrieval(hit(1, 0, 0.9)), kb, max_tokens=512).request.max_tokens == 512


def test_no_retrieved_chunks_give_a_request_without_documents(kb):
    context = build_context(retrieval(), kb)

    assert context.request.documents == () and context.sources == ()
    assert MockProvider().generate(context.request).text == NOT_ENOUGH_EVIDENCE


def test_a_citation_from_a_provider_maps_back_to_the_retrieved_chunk(kb):
    context = build_context(retrieval(hit(1, 3, 0.9), hit(2, 0, 0.8)), kb)

    result = MockProvider().generate(context.request)
    citation = result.parts[1].citations[0]
    source = context.source_for(citation.document_index)

    assert (source.source_file, source.chunk_id, source.rank) == ("hives.txt", 1, 1)
    assert HIVES_1[citation.start_char : citation.end_char] == citation.cited_text


# --- Titles ---


def test_titles_are_the_first_line_without_its_heading_marker(kb):
    assert article_catalogue(kb)["asthma.txt"].title == "Asthma"
    assert article_catalogue(kb)["asthma.txt"].chunk_count == 3
    assert document_title("Asthma", 2, 3) == "Asthma (chunk 2 of 3)"


def test_an_article_without_a_first_chunk_falls_back_to_its_file_name():
    kb = KnowledgeBase(Path("in-memory"), [chunk("mystery_article.txt", 2, "Text.")], indexes={})

    assert article_catalogue(kb)["mystery_article.txt"].title == "mystery_article"


# --- Plain-text rendering ---


def test_plain_text_labels_sources_and_keeps_their_text_verbatim(kb):
    context = build_context(retrieval(hit(1, 1, 0.83), hit(2, 3, 0.71)), kb)

    text = render_plain_text(context)

    assert text == (
        "SOURCES\n\n"
        f"[S1] Asthma (chunk 2 of 3)\n{ASTHMA_2}\n\n"
        f"[S2] Hives (chunk 1 of 1)\n{HIVES_1}\n\n"
        "QUESTION\n\nWhat are the symptoms of asthma?\n\n"
        "Cite the sources you use by their labels, for example [S1]."
    )


def test_plain_text_says_when_nothing_was_retrieved(kb):
    assert "(No documents were retrieved.)" in render_plain_text(build_context(retrieval(), kb))


def test_building_a_context_loads_no_model_or_sdk():
    code = (
        "import sys, medquery_api.llm.context\n"
        "print(sorted(m for m in ('anthropic', 'torch', 'sentence_transformers') "
        "if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "[]"


# --- The research archive (read only) ---


@needs_data
def test_every_archive_title_is_the_raw_files_first_line():
    kb = load_knowledge_base(DATA)
    catalogue = article_catalogue(kb)
    different = []
    for path in sorted((DATA / "raw").glob("*.txt")):
        with path.open("r", encoding="utf-8") as file:
            first_line = file.readline().rstrip("\n")
        if not first_line.startswith("# ") or catalogue[path.name].title != first_line[2:]:
            different.append(path.name)

    assert len(catalogue) == 698
    assert different == []
    assert sum(info.chunk_count for info in catalogue.values()) == 3796


@needs_data
def test_a_context_from_the_real_archive_keeps_chunks_verbatim():
    kb = load_knowledge_base(DATA)
    vectors = kb.indexes["BGE"].index.vectors
    row = kb.find_row("asthma.txt", 1)
    hits = kb.search("BGE", np.asarray(vectors[row]), top_k=5)

    context = build_context(retrieval(*hits, question="What is asthma?"), kb)

    assert [d.text for d in context.request.documents] == [h.chunk.text for h in hits]
    assert context.request.documents[0].title.startswith("Asthma (chunk 1 of ")
    assert [(s.source_file, s.chunk_id, s.rank) for s in context.sources] == [
        (h.chunk.source_file, h.chunk.chunk_id, h.rank) for h in hits
    ]
