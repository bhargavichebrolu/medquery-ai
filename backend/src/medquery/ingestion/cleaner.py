"""Text cleaning and normalisation.

The default configuration reproduces notebook `02_Data_Cleaning` exactly (verified
against all 698 archived articles) while adding defensive steps for untrusted input
that are no-ops on the NHS corpus:

1. Strip a leading byte-order mark and non-whitespace control characters (e.g. NUL).
2. Unicode-normalise (NFC).
3. Split into lines (any line-break convention), strip each line, drop empty lines.
4. Collapse internal whitespace runs (including tabs and no-break spaces) to one space.
5. Drop lines that exactly repeat an earlier line in the same document.
6. Re-join lines separated by a blank line.

Markdown-style headings (``#``, ``##``, ``###``) and ``•`` bullets are preserved as
ordinary lines; the chunker relies on the headings as split points.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from medquery.ingestion.models import Document, MalformedDocumentError

_WHITESPACE_RUN = re.compile(r"\s+")
_BOM = "﻿"


class CleaningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    unicode_normalization: Literal["NFC", "NFKC"] | None = "NFC"
    remove_control_characters: bool = True
    collapse_whitespace: bool = True
    deduplicate_lines: bool = True
    line_separator: str = "\n\n"


# Reproduces notebook 02 (dissertation baseline). Values are spelled out so that later
# changes to `CleaningConfig` defaults cannot alter the baseline. Never modify; define
# new configurations for experiments instead.
BASELINE_CLEANING = CleaningConfig(
    unicode_normalization="NFC",
    remove_control_characters=True,
    collapse_whitespace=True,
    deduplicate_lines=True,
    line_separator="\n\n",
)

DEFAULT_CLEANING = BASELINE_CLEANING


def _remove_control_characters(text: str) -> str:
    # Whitespace controls (\t, \n, \r, \f, \v, ...) are kept: they act as line breaks
    # or spaces and are handled by line splitting and whitespace collapsing.
    return "".join(
        ch
        for ch in text
        if ch != _BOM and not (unicodedata.category(ch) == "Cc" and not ch.isspace())
    )


def clean_text(text: str, config: CleaningConfig = DEFAULT_CLEANING) -> str:
    """Clean raw document text. Returns an empty string for empty/whitespace-only input."""
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text).__name__}")
    if any(unicodedata.category(ch) == "Cs" for ch in text):
        raise MalformedDocumentError("Text contains unpaired surrogate code points")

    if config.remove_control_characters:
        text = _remove_control_characters(text)
    elif text.startswith(_BOM):
        text = text[1:]
    if config.unicode_normalization:
        text = unicodedata.normalize(config.unicode_normalization, text)

    cleaned_lines: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if config.collapse_whitespace:
            line = _WHITESPACE_RUN.sub(" ", line)
        if config.deduplicate_lines:
            if line in seen:
                continue
            seen.add(line)
        cleaned_lines.append(line)

    return config.line_separator.join(cleaned_lines)


def clean_document(document: Document, config: CleaningConfig = DEFAULT_CLEANING) -> Document:
    """Return a copy of `document` with cleaned text; identifier and metadata unchanged."""
    return document.model_copy(update={"text": clean_text(document.text, config)})
