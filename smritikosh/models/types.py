"""Shared data types passed between smritikosh pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Capture:
    """Represent a named definition found by a tree-sitter query.

    The node retains the captured source text and its zero-based
    ``start_point`` and ``end_point`` positions. Printing a capture resembles::

        Capture(
            capture_name="definition.function",
            node=<Node type=function_definition, start_point=(0, 0), ...>,
            name="fetch",
            path="src/client.py",
        )
    """

    capture_name: str
    node: Any
    name: str
    path: str


@dataclass
class Symbol:
    """A named symbol discovered in a source file."""

    id: str
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    parent: str | None


@dataclass
class ParsedFile:
    """A parsed source file and its tree-sitter tree."""

    path: str
    language: str
    content: str
    tree: Any


@dataclass
class Chunk:
    """A retrievable chunk of source text."""

    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    chunk_kind: str
    content_hash: str
    #: Name of the definition this chunk came from, when the chunker knew one.
    #: A grouped or whole-file chunk covers no single definition and has None.
    symbol: str | None = None


@dataclass
class SearchResult:
    """A semantic search hit returned to callers."""

    path: str
    start_line: int
    end_line: int
    snippet: str
    score: float
    chunk_kind: str | None = None
    symbol: str | None = None
    chunk_id: str | None = None


class ChunkingStrategy(Protocol):
    """Contract for all chunking strategies.

    mode_name is a stable class-level constant included in @sm.memoized
    fingerprints; changing it or the chunk() implementation will invalidate
    all downstream memos.
    """

    mode_name: str

    def chunk(self, parsed: ParsedFile, captures: list[Capture]) -> list[Chunk]:
        """Split *parsed* into retrievable chunks using tree-sitter *captures*."""
        ...


@dataclass
class SourceFile:
    """A source file selected for indexing."""

    path: str
    language: str
    content: str
    has_tags_scm: bool
    strategy: ChunkingStrategy

    @property
    def chunking_mode(self) -> str:
        """Return the stable chunking mode used for memo fingerprints."""
        return self.strategy.mode_name
