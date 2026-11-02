"""Split an article into chunks the way the research pipeline did (notebook 03, cells 5-6).

The notebook used LangChain's RecursiveCharacterTextSplitter with chunk_size=1000,
chunk_overlap=200, the separators below, and every other option at its default (separators
kept at the start of the following piece, whitespace stripped from each chunk, length counted
in characters). This module reimplements that algorithm in plain Python, without LangChain.
Applied to the archive's 698 cleaned articles, it is checked to reproduce all 3,796 archived
chunks exactly (see tests/test_processing.py).
"""

from dataclasses import asdict, dataclass

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Tried in order: section headings, subsection headings, paragraphs, lines, sentences, words,
# and finally single characters.
SEPARATORS = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class Chunk:
    """One chunk, with the same fields, in the same order, as the archive's chunk files."""

    chunk_id: int  # position within the article, counting from 1
    source_file: str  # the cleaned article it came from, e.g. "asthma.txt"
    chunk_length: int  # number of characters in text
    text: str

    def as_record(self) -> dict:
        return asdict(self)


def chunk_article(
    text: str,
    source_file: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: tuple[str, ...] = SEPARATORS,
) -> list[Chunk]:
    """Split a cleaned article into numbered chunks, as notebook 03 recorded them."""
    pieces = split_text(text, chunk_size, chunk_overlap, separators)
    return [
        Chunk(chunk_id=i, source_file=source_file, chunk_length=len(piece), text=piece)
        for i, piece in enumerate(pieces, start=1)
    ]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: tuple[str, ...] = SEPARATORS,
) -> list[str]:
    """Split text into chunks of at most chunk_size characters.

    Chunks are made of whole pieces of the text, broken at the first separator (in order)
    that the text contains. A new chunk starts with whole pieces carried over from the end of
    the previous chunk, as many as fit in chunk_overlap characters.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must not be negative, got {chunk_overlap}")
    if chunk_overlap > chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) is larger than chunk_size ({chunk_size})"
        )
    if not separators:
        raise ValueError("at least one separator is needed")
    return _split(text, list(separators), chunk_size, chunk_overlap)


def _split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    # Use the first separator the text contains; "" (single characters) always applies.
    separator, finer_separators = separators[-1], []
    for i, candidate in enumerate(separators):
        if candidate == "":
            separator = candidate
            break
        if candidate in text:
            separator, finer_separators = candidate, separators[i + 1 :]
            break

    chunks: list[str] = []
    short_pieces: list[str] = []
    for piece in _split_keeping_separator(text, separator):
        if len(piece) < chunk_size:
            short_pieces.append(piece)
            continue
        # A piece too long to fit: merge what came before it, then split it more finely.
        if short_pieces:
            chunks.extend(_merge(short_pieces, chunk_size, chunk_overlap))
            short_pieces = []
        if finer_separators:
            chunks.extend(_split(piece, finer_separators, chunk_size, chunk_overlap))
        else:
            chunks.append(piece)
    if short_pieces:
        chunks.extend(_merge(short_pieces, chunk_size, chunk_overlap))
    return chunks


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split text at each separator, keeping the separator at the start of the next piece."""
    if separator == "":
        pieces = list(text)
    else:
        first, *rest = text.split(separator)
        pieces = [first] + [separator + part for part in rest]
    return [piece for piece in pieces if piece]


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Join consecutive pieces into chunks of at most chunk_size characters, with overlap."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0
    for piece in pieces:
        if current and total + len(piece) > chunk_size:
            _add_chunk(chunks, current)
            # Keep only trailing pieces that fit in the overlap and leave room for this piece.
            while total > chunk_overlap or (total + len(piece) > chunk_size and total > 0):
                total -= len(current.pop(0))
        current.append(piece)
        total += len(piece)
    _add_chunk(chunks, current)
    return chunks


def _add_chunk(chunks: list[str], pieces: list[str]) -> None:
    chunk = "".join(pieces).strip()
    if chunk:
        chunks.append(chunk)
