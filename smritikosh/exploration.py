"""Compatibility exports for the read-only exploration API."""

from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.models.retrieval import (
    IndexedChunk,
    IndexInfo,
    OutlineEntry,
    SearchOptions,
    SourceLine,
    TextMatch,
)

ReadOnlyExplorer = DuckDBSourceReader

__all__ = [
    "IndexInfo",
    "IndexedChunk",
    "OutlineEntry",
    "ReadOnlyExplorer",
    "SearchOptions",
    "SourceLine",
    "TextMatch",
]
