"""The vector storage contract -- implementations live in adapters/vector_store/."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

__all__ = ["VectorStore"]


class VectorStore(ABC):
    """Stores one vector per chunk and ranks them against a query vector."""

    @abstractmethod
    def setup(self, dims: int) -> None:
        """Prepare storage for vectors of width dims, rebuilding if dims changed."""

    @abstractmethod
    def upsert(self, chunk_id: str, vector: list[float]) -> None: ...

    def upsert_many(self, items: Sequence[tuple[str, list[float]]]) -> None:
        """Store several vectors at once.

        Defaults to one :meth:`upsert` per item.  Override wherever writing a
        vector row-by-row is expensive — on DuckDB the per-row parameter bind
        walks every float, so this is the difference between 30 ms and 0.5 ms
        per vector.
        """
        for chunk_id, vector in items:
            self.upsert(chunk_id, vector)

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine_score) pairs ranked by score descending."""

    @abstractmethod
    def delete(self, chunk_id: str) -> None: ...

    def delete_many(self, chunk_ids: Iterable[str]) -> None:
        """Remove several vectors at once.

        Defaults to one :meth:`delete` per id; stores that can do it in a
        single round trip should override.
        """
        for chunk_id in chunk_ids:
            self.delete(chunk_id)

    @abstractmethod
    def clear(self) -> None:
        """Remove every stored vector."""

    @abstractmethod
    def exists(self, chunk_id: str) -> bool: ...

    @abstractmethod
    def get_stored_dims(self) -> int | None:
        """Width the store was last set up with, or None if never set up."""
