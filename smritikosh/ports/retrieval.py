"""Ports used by retrieval application services."""

from __future__ import annotations

from typing import Protocol

from smritikosh.models import Chunk, SearchResult
from smritikosh.models.retrieval import RetrievalChannel, SearchOptions

__all__ = ["CandidateRetriever", "LexicalStore"]


class CandidateRetriever(Protocol):
    """Generate one ranked candidate list for a query."""

    @property
    def channel(self) -> RetrievalChannel:
        """Identify this independent retrieval channel."""

    def retrieve(
        self,
        query: str,
        *,
        options: SearchOptions,
    ) -> list[SearchResult]:
        """Return candidates ordered from most to least relevant."""


class LexicalStore(Protocol):
    """Maintain and query a lexical index alongside dense vectors."""

    def setup(self) -> bool:
        """Create lexical storage when absent.

        Returns True when the stored index was written by an incompatible
        schema, so the caller can clear it together with the caches that
        would otherwise skip refilling it.
        """

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or replace lexical documents for chunks."""

    def delete(self, chunk_id: str) -> None:
        """Remove one lexical document."""

    def delete_path(self, path: str) -> None:
        """Remove every lexical document belonging to a path."""

    def clear(self) -> None:
        """Remove every lexical document and posting."""

    def search(
        self,
        query: str,
        *,
        options: SearchOptions,
    ) -> list[tuple[str, float]]:
        """Return chunk ids and BM25 scores ordered by relevance."""
