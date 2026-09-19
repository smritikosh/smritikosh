"""The read-only retrieval contract -- implementations live in adapters/retrieval/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from smritikosh.models import (
    IndexedChunk,
    IndexInfo,
    OutlineEntry,
    SearchOptions,
    SearchResult,
    SourceLine,
    TextMatch,
)

if TYPE_CHECKING:
    from types import TracebackType

    from smritikosh.ports.embedder import Embedder

__all__ = ["SourceReader"]


class SourceReader(ABC):
    """Reads an existing index without holding a write connection.

    Everything here answers a question about an index that has already been
    built; nothing on this contract may mutate one.  Implementations open
    their own connection and must be closed, either by :meth:`close` or by
    using the reader as a context manager.
    """

    # ----------------------------------------------------------- lifecycle

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection."""

    def __enter__(self) -> SourceReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---------------------------------------------------------------- index

    @abstractmethod
    def index_info(self) -> IndexInfo:
        """Return index counts and embedding dimensions."""

    # --------------------------------------------------------------- search

    @abstractmethod
    def semantic_search(
        self,
        queries: list[str],
        *,
        embedder: Embedder,
        options: SearchOptions | None = None,
    ) -> list[SearchResult]:
        """Rank chunks against one or more natural-language queries."""

    @abstractmethod
    def text_search(
        self,
        text: str,
        *,
        path: str | None = None,
        limit: int = 20,
    ) -> list[IndexedChunk]:
        """Find chunks containing exact text, case-insensitively."""

    @abstractmethod
    def find_text_lines(
        self,
        text: str,
        *,
        path: str | None = None,
        limit: int = 20,
    ) -> list[TextMatch]:
        """Find the individual source lines containing exact text."""

    @abstractmethod
    def find_paths(self, pattern: str, *, limit: int = 50) -> list[str]:
        """Find indexed paths containing a case-insensitive substring."""

    # --------------------------------------------------------------- chunks

    @abstractmethod
    def get_chunks(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[IndexedChunk]:
        """Return chunks from one path, optionally overlapping a line range."""

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, IndexedChunk]:
        """Look up chunks by id, keyed by the id that found them.

        Ids with no stored chunk are absent from the mapping, so a ranked
        caller can tell a miss from a hit rather than pairing its scores
        against a silently shorter result.
        """

    @abstractmethod
    def get_outline(self, path: str) -> list[OutlineEntry]:
        """List what one path defines, without any of its source."""

    @abstractmethod
    def get_source_lines(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[SourceLine]:
        """Rebuild the source of one path from its chunks, line by line."""
