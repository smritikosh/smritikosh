"""Language-neutral tokenization for lexical code retrieval."""

from __future__ import annotations

import re
from typing import Final

__all__ = ["tokenize_code"]

_RAW_TOKEN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_PART: Final[re.Pattern[str]] = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+"
)


def _is_indexable(token: str) -> bool:
    """Reject tokens that barely distinguish one chunk from another.

    A single character or a bare number costs a posting and counts toward the
    document length the score is normalised by, while matching almost anything
    it is asked about — 7% of the postings on a 10k-chunk index.  Digits
    survive inside the identifier that produced them, so ``sha256`` stays
    findable where the lone ``256`` does not.
    """
    return len(token) > 1 and not token.isdigit()


def tokenize_code(text: str) -> list[str]:
    """Tokenize prose, paths, snake_case, and camelCase identifiers."""
    tokens: list[str] = []
    for raw in _RAW_TOKEN.findall(text):
        exact: str = raw.lower()
        if _is_indexable(exact):
            tokens.append(exact)
        parts: list[str] = []
        for snake_part in raw.split("_"):
            parts.extend(part.lower() for part in _CAMEL_PART.findall(snake_part))
        if parts != [exact]:
            tokens.extend(part for part in parts if _is_indexable(part))
    return tokens
