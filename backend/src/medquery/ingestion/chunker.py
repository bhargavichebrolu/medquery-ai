"""Configurable document chunking.

The initial strategy is recursive character splitting (LangChain's
`RecursiveCharacterTextSplitter`), the same implementation and defaults used in
notebook `03_Text_Chunking`: 1,000-character chunks, 200-character overlap, splitting
preferentially at section headings, then subsection headings, paragraphs, lines,
sentences, words, and finally characters. See docs/ingestion/CHUNKING.md.
"""

from __future__ import annotations

from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from medquery.ingestion.models import Chunk, Document, make_chunk_uid

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ", "")


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["recursive_character"] = "recursive_character"
    chunk_size: int = Field(default=1000, gt=0, description="Maximum characters per chunk.")
    chunk_overlap: int = Field(default=200, ge=0, description="Maximum overlap in characters.")
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    @model_validator(mode="after")
    def _check(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.separators:
            raise ValueError("at least one separator is required")
        return self


# Reproduces notebook 03 (dissertation baseline). Values are spelled out so that later
# changes to `ChunkingConfig` defaults cannot alter the baseline. Never modify; define
# new configurations for experiments instead.
BASELINE_CHUNKING = ChunkingConfig(
    strategy="recursive_character",
    chunk_size=1000,
    chunk_overlap=200,
    separators=("\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""),
)

DEFAULT_CHUNKING = BASELINE_CHUNKING


class Chunker:
    """Splits cleaned documents into `Chunk`s, preserving identifiers and metadata."""

    def __init__(self, config: ChunkingConfig = DEFAULT_CHUNKING) -> None:
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=list(config.separators),
            add_start_index=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        pieces = self._splitter.create_documents([document.text])
        return [
            Chunk(
                chunk_uid=make_chunk_uid(document.source_id, chunk_id),
                source_id=document.source_id,
                chunk_id=chunk_id,
                text=piece.page_content,
                chunk_length=len(piece.page_content),
                start_index=piece.metadata["start_index"],
                metadata=dict(document.metadata),
            )
            for chunk_id, piece in enumerate(pieces, start=1)
        ]


def chunk_document(document: Document, config: ChunkingConfig = DEFAULT_CHUNKING) -> list[Chunk]:
    return Chunker(config).chunk(document)
