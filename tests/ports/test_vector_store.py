"""Tests for the behaviour VectorStore gives implementations for free."""

from __future__ import annotations

from smritikosh.ports.vector_store import VectorStore


class _MinimalStore(VectorStore):
    """Implements only the abstract surface — no delete_many override."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def setup(self, dims: int) -> None: ...

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        self.vectors[chunk_id] = vector

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        return []

    def delete(self, chunk_id: str) -> None:
        self.vectors.pop(chunk_id, None)

    def clear(self) -> None:
        self.vectors.clear()

    def exists(self, chunk_id: str) -> bool:
        return chunk_id in self.vectors

    def get_stored_dims(self) -> int | None:
        return None


def test_delete_many_falls_back_to_one_delete_per_id() -> None:
    """A store that only implements delete still gets a working delete_many."""
    store = _MinimalStore()
    store.upsert("doomed", [1.0])
    store.upsert("also-doomed", [1.0])
    store.upsert("spared", [1.0])

    store.delete_many(["doomed", "also-doomed"])

    assert set(store.vectors) == {"spared"}
