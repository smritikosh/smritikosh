"""Tests for the DuckDB source-reader adapter's id lookup."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.ports.source_reader import SourceReader


@pytest.fixture()
def reader(tmp_path: Path) -> DuckDBSourceReader:
    """An index holding two chunks in one file, plus the file node itself."""
    db_path: Path = tmp_path / "index.duckdb"
    connection = duckdb.connect(str(db_path))
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
        "INSERT INTO nodes VALUES ('file:billing.py', 'file', NULL, 'billing.py', '{}')"
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, 'chunk', ?, 'billing.py', ?)",
        [
            (
                "c1",
                "charge",
                json.dumps(
                    {
                        "start_line": 1,
                        "end_line": 4,
                        "text": "def charge(): ...",
                        "chunk_kind": "function",
                    }
                ),
            ),
            (
                "c2",
                None,
                json.dumps(
                    {
                        "start_line": 6,
                        "end_line": 9,
                        "text": "TAX_RATE = 0.2",
                        "chunk_kind": None,
                    }
                ),
            ),
        ],
    )
    connection.close()
    return DuckDBSourceReader(str(db_path))


def test_looks_up_every_requested_chunk(reader: DuckDBSourceReader) -> None:
    assert set(reader.get_chunks_by_ids(["c1", "c2"])) == {"c1", "c2"}


def test_returns_the_stored_chunk_contents(reader: DuckDBSourceReader) -> None:
    chunk = reader.get_chunks_by_ids(["c1"])["c1"]

    assert chunk.chunk_id == "c1"
    assert chunk.path == "billing.py"
    assert (chunk.start_line, chunk.end_line) == (1, 4)
    assert chunk.text == "def charge(): ..."
    assert chunk.chunk_kind == "function"
    assert chunk.symbol == "charge"


def test_returns_no_symbol_when_the_chunker_knew_none(
    reader: DuckDBSourceReader,
) -> None:
    chunk = reader.get_chunks_by_ids(["c2"])["c2"]

    assert chunk.symbol is None
    assert chunk.chunk_kind is None


def test_omits_ids_that_are_not_stored(reader: DuckDBSourceReader) -> None:
    """A miss must be visible to the caller, not a silently shorter result.

    A ranked caller zipping scores against a list would mis-pair every entry
    after the gap; keying by id makes the absence something it has to read.
    """
    found = reader.get_chunks_by_ids(["c1", "does-not-exist", "c2"])

    assert set(found) == {"c1", "c2"}
    assert "does-not-exist" not in found


def test_returns_nothing_for_no_ids(reader: DuckDBSourceReader) -> None:
    assert reader.get_chunks_by_ids([]) == {}


def test_ignores_nodes_that_are_not_chunks(reader: DuckDBSourceReader) -> None:
    assert reader.get_chunks_by_ids(["file:billing.py"]) == {}


# ── SourceReader contract ─────────────────────────────────────────────────────


def test_implements_the_source_reader_port(reader: DuckDBSourceReader) -> None:
    assert isinstance(reader, SourceReader)


def test_leaving_the_context_closes_the_connection(reader: DuckDBSourceReader) -> None:
    with reader as opened:
        assert opened is reader

    with pytest.raises(duckdb.ConnectionException):
        reader.index_info()
