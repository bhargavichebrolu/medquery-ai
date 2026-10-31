"""Loading plain-text source documents.

Only UTF-8 text is supported at this stage; PDF and DOCX extraction (FR-02) will add
further loaders that produce the same `Document` model.
"""

from __future__ import annotations

from pathlib import Path

from medquery.ingestion.models import Document, MalformedDocumentError, MetadataValue


def load_text_document(
    path: Path,
    *,
    source_id: str | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Document:
    """Read a UTF-8 (optionally BOM-prefixed) text file as a `Document`.

    `source_id` defaults to the file name, matching the research artefacts.
    """
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise MalformedDocumentError(
            f"{path.name}: not valid UTF-8 (byte offset {error.start})"
        ) from error
    return Document(source_id=source_id or path.name, text=text, metadata=metadata or {})
