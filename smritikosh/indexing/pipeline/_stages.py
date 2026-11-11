"""CPU-bound pipeline stages — each runs in a thread pool via @sm.threaded.

All three wrappers follow the same naming convention:

    verb_file()   ← the @sm.tracked implementation in its own module
    _verb()       ← the @sm.threaded wrapper used by the pipeline

This ensures the asyncio event loop is never blocked while sm.fan_out
processes many files concurrently.
"""

from __future__ import annotations

from smritikosh.engine import sm
from smritikosh.indexing.chunker import chunk_file
from smritikosh.indexing.extractor import extract_file
from smritikosh.indexing.graph import graph_file
from smritikosh.indexing.parser import parse_file
from smritikosh.models import (
    Call,
    Capture,
    Chunk,
    ChunkingStrategy,
    Import,
    ParsedFile,
    SourceFile,
    Symbol,
)


@sm.threaded
def _parse(source: SourceFile) -> ParsedFile:
    """Wrap parse_file in asyncio.to_thread — tree-sitter releases the GIL."""
    return parse_file(source)


@sm.threaded
def _extract(parsed: ParsedFile, has_tags_scm: bool) -> list[Capture]:
    """Wrap extract_file in asyncio.to_thread — tree-sitter query on a thread."""
    return extract_file(parsed, has_tags_scm)


@sm.threaded
def _chunk(
    parsed: ParsedFile,
    captures: list[Capture],
    strategy: ChunkingStrategy,
) -> list[Chunk]:
    """Wrap chunk_file in asyncio.to_thread — AST traversal on a thread."""
    return chunk_file(parsed, captures, strategy)


@sm.threaded
def _graph(
    parsed: ParsedFile, captures: list[Capture]
) -> tuple[list[Symbol], list[Call], list[Import]]:
    return graph_file(parsed, captures)
