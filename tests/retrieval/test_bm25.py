"""Tests for the incremental DuckDB BM25 adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from smritikosh.adapters.retrieval.duckdb import (
    DuckDBBm25Store,
    DuckDBLexicalRetriever,
)
from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.models import Chunk, SearchOptions


def _chunk(chunk_id: str, path: str, text: str, symbol: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        path=path,
        start_line=1,
        end_line=1,
        text=text,
        chunk_kind="function",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        symbol=symbol,
    )


def test_should_rank_rare_identifier_above_common_report_terms() -> None:
    connection = duckdb.connect(":memory:")
    store = DuckDBBm25Store(con=connection)
    store.setup()
    store.upsert(
        [
            _chunk("common", "src/report.py", "publish report report", "publish"),
            _chunk(
                "specific",
                "src/idempotency.py",
                "skip duplicate completed event",
                "idempotent_handler",
            ),
        ]
    )

    hits = store.search(
        "duplicate report idempotency",
        options=SearchOptions(top_k=2),
    )

    assert [chunk_id for chunk_id, _ in hits] == ["specific", "common"]


def test_should_replace_and_delete_lexical_documents_incrementally() -> None:
    connection = duckdb.connect(":memory:")
    store = DuckDBBm25Store(con=connection)
    store.setup()
    initial = _chunk("one", "src/a.py", "before", "load")
    store.upsert([initial, initial])
    store.upsert([_chunk("one", "src/a.py", "after", "load")])

    before = store.search("before", options=SearchOptions(top_k=5))
    after = store.search("after", options=SearchOptions(top_k=5))
    store.delete("one")
    deleted = store.search("after", options=SearchOptions(top_k=5))

    assert before == []
    assert [chunk_id for chunk_id, _ in after] == ["one"]
    assert deleted == []


def test_should_apply_path_filters_to_bm25_results() -> None:
    connection = duckdb.connect(":memory:")
    store = DuckDBBm25Store(con=connection)
    store.setup()
    store.upsert(
        [
            _chunk("source", "src/a.py", "retry event", "retry"),
            _chunk("test", "tests/test_a.py", "retry event", "test_retry"),
        ]
    )

    hits = store.search(
        "retry",
        options=SearchOptions(top_k=5, exclude_paths=("tests/%",)),
    )

    assert [chunk_id for chunk_id, _ in hits] == ["source"]


def test_should_explain_how_to_rebuild_when_bm25_index_is_missing() -> None:
    store = DuckDBBm25Store(con=duckdb.connect(":memory:"))

    with pytest.raises(RuntimeError, match="index --full"):
        store.search("anything", options=SearchOptions(top_k=5))


def test_should_rollback_replacement_when_posting_insert_fails() -> None:
    connection = duckdb.connect(":memory:")
    store = DuckDBBm25Store(con=connection)
    store.setup()
    store.upsert([_chunk("one", "src/a.py", "before", "load")])
    failing_connection = MagicMock(wraps=connection)

    def fail_posting_insert(query: str, *args: object) -> object:
        # Postings go in last; failing there leaves the documents already
        # inserted, which is exactly what the transaction has to undo.
        if "lexical_terms" in query and "INSERT" in query:
            raise RuntimeError("posting insert failed")
        return connection.execute(query, *args)

    failing_connection.execute.side_effect = fail_posting_insert
    store._connection = failing_connection

    with pytest.raises(RuntimeError, match="posting insert failed"):
        store.upsert([_chunk("one", "src/a.py", "after", "load")])

    store._connection = connection
    assert [
        chunk_id
        for chunk_id, _ in store.search(
            "before",
            options=SearchOptions(top_k=5),
        )
    ] == ["one"]


def test_should_return_bm25_candidates_carrying_reader_metadata(
    tmp_path: Path,
) -> None:
    """The retriever reads chunks back through the real source-reader port.

    The port keys its lookup by id rather than returning a list, so a
    retriever that iterates the mapping gets ids where it expects chunks.
    """
    database_path: Path = tmp_path / "index.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT,
            path TEXT,
            metadata JSON
        );
        CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, vector FLOAT[3]);
        CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    connection.execute(
        "INSERT INTO nodes"
        " VALUES ('one', 'chunk', 'compute_late_fee', 'billing.py', ?)",
        [
            json.dumps(
                {
                    "start_line": 1,
                    "end_line": 4,
                    "text": "def compute_late_fee(): ...",
                    "chunk_kind": "function",
                }
            )
        ],
    )
    store = DuckDBBm25Store(con=connection)
    store.setup()
    store.upsert(
        [
            _chunk(
                "one", "billing.py", "def compute_late_fee(): ...", "compute_late_fee"
            ),
            _chunk("unindexed", "gone.py", "compute_late_fee was here", "stale"),
        ]
    )
    connection.close()

    reader = DuckDBSourceReader(str(database_path))
    results = DuckDBLexicalRetriever(
        DuckDBBm25Store(str(database_path), read_only=True),
        reader,
    ).retrieve("compute_late_fee", options=SearchOptions(top_k=5))

    assert [result.chunk_id for result in results] == ["one"]
    assert results[0].path == "billing.py"
    assert (results[0].start_line, results[0].end_line) == (1, 4)
    assert results[0].symbol == "compute_late_fee"
    assert results[0].score > 0.0


def test_should_report_an_index_written_by_an_older_schema() -> None:
    """Report the mismatch and leave the rows for the caller to clear.

    Emptying them here would strand the index: the postings are written by a
    memoised stage, so clearing them without the memo cache leaves every file
    a cache hit and nothing to refill them.
    """
    connection = duckdb.connect(":memory:")
    store = DuckDBBm25Store(con=connection)
    store.setup()
    store.upsert([_chunk("one", "src/a.py", "before", "load")])
    connection.execute(
        "UPDATE lexical_metadata SET value = 'old' WHERE key = 'schema_version'"
    )

    assert store.setup() is True
    assert [
        chunk_id
        for chunk_id, _ in store.search("before", options=SearchOptions(top_k=5))
    ] == ["one"]
    assert store.setup() is False
