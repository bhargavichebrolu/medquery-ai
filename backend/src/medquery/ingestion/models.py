"""Document and chunk data models shared by the ingestion stages."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetadataValue = str | int | float | bool | None


class MalformedDocumentError(ValueError):
    """Raised when document content cannot be decoded or processed safely."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Document(_Frozen):
    """A source document as plain text, before or after cleaning.

    `source_id` is the stable identifier of the document within its knowledge base.
    For the NHS corpus it is the article filename (e.g. ``asthma.txt``), matching the
    ``source_file`` field of the original research artefacts.
    """

    source_id: str = Field(min_length=1)
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class Chunk(_Frozen):
    """A contiguous span of a cleaned document.

    `chunk_id` is 1-based and restarts for every document (as in the research
    artefacts); `chunk_uid` is unique across a knowledge base. `start_index` is the
    character offset of the chunk within the cleaned document text.
    """

    chunk_uid: str
    source_id: str = Field(min_length=1)
    chunk_id: int = Field(ge=1)
    text: str = Field(min_length=1)
    chunk_length: int = Field(ge=1)
    start_index: int = Field(ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> Chunk:
        if self.chunk_length != len(self.text):
            raise ValueError("chunk_length must equal len(text)")
        if self.chunk_uid != make_chunk_uid(self.source_id, self.chunk_id):
            raise ValueError("chunk_uid does not match source_id and chunk_id")
        return self

    def to_legacy_record(self) -> dict[str, Any]:
        """Return the record in the format of `Healthcare_RAG/data/chunks/*.json`."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_id,
            "chunk_length": self.chunk_length,
            "text": self.text,
        }


def make_chunk_uid(source_id: str, chunk_id: int) -> str:
    return f"{source_id}#{chunk_id}"
