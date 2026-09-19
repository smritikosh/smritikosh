"""Models shared by retrieval application services and adapters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "IndexInfo",
    "IndexedChunk",
    "OutlineEntry",
    "SearchOptions",
    "SourceLine",
    "TextMatch",
]


@dataclass(frozen=True)
class IndexInfo:
    """Summarize the contents of a Smritikosh index."""

    files: int
    chunks: int
    vectors: int
    dimensions: int | None


@dataclass(frozen=True)
class IndexedChunk:
    """Represent one source chunk stored in the index."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    chunk_kind: str | None
    #: Definition this chunk came from; None for chunks that cover no single
    #: definition, and for chunks written before symbols were recorded.
    symbol: str | None = None


@dataclass(frozen=True)
class SourceLine:
    """Represent one numbered source line."""

    line_number: int
    text: str


@dataclass(frozen=True)
class OutlineEntry:
    """Summarize one definition stored for a path."""

    symbol: str | None
    chunk_kind: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class TextMatch:
    """Locate one source line containing exact text."""

    path: str
    line_number: int
    text: str
    exact_line: bool


@dataclass(frozen=True)
class SearchOptions:
    """Configure one candidate retrieval operation."""

    top_k: int = 10
    include_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = ()
