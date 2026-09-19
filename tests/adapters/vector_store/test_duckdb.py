"""Tests for the DuckDB vector store."""

import duckdb
import pytest

from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore

X_AXIS = [1.0, 0.0, 0.0, 0.0]
Y_AXIS = [0.0, 1.0, 0.0, 0.0]


@pytest.fixture
def store() -> DuckDBVectorStore:
    return DuckDBVectorStore(con=duckdb.connect(":memory:"))


def test_should_rank_the_nearest_chunk_first(store: DuckDBVectorStore) -> None:
    store.setup(4)
    store.upsert("far", Y_AXIS)
    store.upsert("near", X_AXIS)

    hits = store.search(X_AXIS, top_k=2)

    assert [chunk_id for chunk_id, _ in hits] == ["near", "far"]


def test_should_reject_dims_when_not_positive(store: DuckDBVectorStore) -> None:
    with pytest.raises(ValueError, match="dims must be positive"):
        store.setup(0)


def test_should_clear_all_vectors(store: DuckDBVectorStore) -> None:
    store.setup(4)
    store.upsert("first", X_AXIS)
    store.upsert("second", Y_AXIS)

    store.clear()

    assert store.exists("first") is False
    assert store.exists("second") is False


def test_should_delete_many_in_one_round_trip(store: DuckDBVectorStore) -> None:
    store.setup(4)
    store.upsert("doomed", X_AXIS)
    store.upsert("also-doomed", Y_AXIS)
    store.upsert("spared", X_AXIS)

    store.delete_many(["doomed", "also-doomed"])

    assert store.exists("doomed") is False
    assert store.exists("also-doomed") is False
    assert store.exists("spared") is True


def test_should_delete_many_without_ids(store: DuckDBVectorStore) -> None:
    store.setup(4)
    store.upsert("spared", X_AXIS)

    store.delete_many([])  # an empty IN () would not parse

    assert store.exists("spared") is True
