"""Clean an article's text the way the research pipeline did (notebook 02, cell 4).

Applied to the archive's 698 raw articles, this reproduces the 698 cleaned articles byte for
byte (see tests/test_processing.py).
"""

import re

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return the cleaned version of an article.

    1. Split into lines (Python's str.splitlines) and trim each line.
    2. Drop empty lines.
    3. Replace each run of whitespace inside a line with one space.
    4. Drop a line if the same line appeared anywhere earlier in the article.
    5. Join the remaining lines with one blank line between them.

    Lines and whitespace are as Python defines them (str.splitlines, str.strip and the regex
    \\s), so this exact behaviour depends on using those functions.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _WHITESPACE.sub(" ", line)
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return "\n\n".join(kept)
