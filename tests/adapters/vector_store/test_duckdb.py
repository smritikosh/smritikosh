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


# ── upsert_many ───────────────────────────────────────────────────────────────


def test_should_upsert_many_in_one_round_trip(store: DuckDBVectorStore) -> None:
    store.setup(4)

    store.upsert_many([("near", X_AXIS), ("far", Y_AXIS)])

    assert [chunk_id for chunk_id, _ in store.search(X_AXIS, top_k=2)] == [
        "near",
        "far",
    ]


def test_should_upsert_many_without_items(store: DuckDBVectorStore) -> None:
    """An empty Arrow table has no width to declare, so this must short-circuit."""
    store.setup(4)

    store.upsert_many([])

    assert store.search(X_AXIS, top_k=1) == []


def test_upsert_many_replaces_an_existing_vector(store: DuckDBVectorStore) -> None:
    store.setup(4)
    store.upsert("moving", Y_AXIS)

    store.upsert_many([("moving", X_AXIS)])

    chunk_id, score = store.search(X_AXIS, top_k=1)[0]
    assert chunk_id == "moving"
    assert score == pytest.approx(1.0)


def test_batched_write_stores_what_row_by_row_stored(
    store: DuckDBVectorStore,
) -> None:
    """The Arrow path must be a pure speed change — identical bytes, so search
    results cannot shift. Compares full-width realistic vectors, not axes.
    """
    dims = 768
    vectors = {
        f"chunk{i}": [((i * 7 + j) % 1000) / 1000.0 for j in range(dims)]
        for i in range(5)
    }

    row_by_row = DuckDBVectorStore(con=duckdb.connect(":memory:"))
    row_by_row.setup(dims)
    for chunk_id, vector in vectors.items():
        row_by_row.upsert(chunk_id, vector)

    batched = DuckDBVectorStore(con=duckdb.connect(":memory:"))
    batched.setup(dims)
    batched.upsert_many(list(vectors.items()))

    probe = vectors["chunk3"]
    assert row_by_row.search(probe, top_k=5) == batched.search(probe, top_k=5)
