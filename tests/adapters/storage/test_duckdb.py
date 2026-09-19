"""Tests for DuckDBAdapter (storage.py — S6)."""

from __future__ import annotations

import duckdb
import pytest

from smritikosh.adapters.storage.duckdb import DuckDBAdapter
from smritikosh.models import Chunk, SourceFile

# ── helpers ─────────────────────────────────────────────────────────────────


class _FakeStrategy:
    mode_name = "ast"


def _source(path: str = "a/b.py", content: str = "x = 1") -> SourceFile:
    return SourceFile(
        path=path,
        language="python",
        content=content,
        has_tags_scm=True,
        strategy=_FakeStrategy(),
    )


def _chunk(
    id: str = "chunk-1",  # noqa: A002
    path: str = "a/b.py",
    text: str = "def foo(): pass",
    start_line: int = 1,
    end_line: int = 1,
    symbol: str | None = None,
) -> Chunk:
    import hashlib

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    return Chunk(
        id=id,
        path=path,
        start_line=start_line,
        end_line=end_line,
        text=text,
        chunk_kind="ast",
        content_hash=content_hash,
        symbol=symbol,
    )


# ── fixture ─────────────────────────────────────────────────────────────────


@pytest.fixture
def adapter() -> DuckDBAdapter:
    return DuckDBAdapter(con=duckdb.connect(":memory:"))


# ── upsert_file_node ────────────────────────────────────────────────────────


def test_upsert_file_node_stores_a_file_row(adapter: DuckDBAdapter) -> None:
    adapter.upsert_file_node(_source())

    rows = adapter.con.execute(
        "SELECT kind, path FROM nodes WHERE kind = 'file'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("file", "a/b.py")


def test_upsert_file_node_is_idempotent(adapter: DuckDBAdapter) -> None:
    src = _source()
    adapter.upsert_file_node(src)
    adapter.upsert_file_node(src)  # second call must not raise

    count = adapter.con.execute(
        "SELECT count(*) FROM nodes WHERE kind = 'file'"
    ).fetchone()[0]
    assert count == 1


def test_files_with_identical_content_get_separate_nodes(
    adapter: DuckDBAdapter,
) -> None:
    """Regression: nodes were keyed by content hash, so twins overwrote.

    A repo full of empty ``__init__.py`` files kept exactly one node between
    them, and the three copies of an agent doc under .agents/.claude/.cursor
    collapsed to one.
    """
    same = "# shared\n"
    for path in ("a/__init__.py", "b/__init__.py", "c/__init__.py"):
        adapter.upsert_file_node(_source(path=path, content=same))

    paths = {
        row[0]
        for row in adapter.con.execute(
            "SELECT path FROM nodes WHERE kind = 'file'"
        ).fetchall()
    }
    assert paths == {"a/__init__.py", "b/__init__.py", "c/__init__.py"}


def test_deleting_one_of_two_identical_files_keeps_the_other(
    adapter: DuckDBAdapter,
) -> None:
    same = "# shared\n"
    adapter.upsert_file_node(_source(path="keep.py", content=same))
    adapter.upsert_file_node(_source(path="drop.py", content=same))

    adapter.delete_file("drop.py")

    paths = {
        row[0]
        for row in adapter.con.execute(
            "SELECT path FROM nodes WHERE kind = 'file'"
        ).fetchall()
    }
    assert paths == {"keep.py"}


# ── upsert_chunk_nodes ──────────────────────────────────────────────────────


def test_upsert_chunk_nodes_stores_all_chunks(adapter: DuckDBAdapter) -> None:
    chunks = [_chunk("c1", end_line=1), _chunk("c2", end_line=2)]
    adapter.upsert_chunk_nodes(chunks)

    ids = {
        r[0]
        for r in adapter.con.execute(
            "SELECT id FROM nodes WHERE kind = 'chunk'"
        ).fetchall()
    }
    assert ids == {"c1", "c2"}


def test_upsert_chunk_nodes_is_idempotent(adapter: DuckDBAdapter) -> None:
    chunk = _chunk()
    adapter.upsert_chunk_nodes([chunk])
    adapter.upsert_chunk_nodes([chunk])

    count = adapter.con.execute(
        "SELECT count(*) FROM nodes WHERE kind = 'chunk'"
    ).fetchone()[0]
    assert count == 1


def test_upsert_chunk_nodes_no_op_on_empty_list(adapter: DuckDBAdapter) -> None:
    adapter.upsert_chunk_nodes([])  # must not raise


# ── delete_chunk_node ───────────────────────────────────────────────────────


def test_delete_chunk_node_removes_only_targeted_chunk(adapter: DuckDBAdapter) -> None:
    adapter.upsert_chunk_nodes([_chunk("keep"), _chunk("gone")])

    adapter.delete_chunk_node("gone")

    ids = {
        r[0]
        for r in adapter.con.execute(
            "SELECT id FROM nodes WHERE kind = 'chunk'"
        ).fetchall()
    }
    assert ids == {"keep"}


# ── get_chunk_ids_for_file ──────────────────────────────────────────────────


def test_get_chunk_ids_for_file_returns_correct_ids(adapter: DuckDBAdapter) -> None:
    adapter.upsert_chunk_nodes(
        [
            _chunk("c1", path="foo.py"),
            _chunk("c2", path="foo.py"),
            _chunk("c3", path="bar.py"),
        ]
    )

    result = adapter.get_chunk_ids_for_file("foo.py")

    assert result == {"c1", "c2"}


def test_get_chunk_ids_for_file_returns_empty_set_when_no_match(
    adapter: DuckDBAdapter,
) -> None:
    assert adapter.get_chunk_ids_for_file("missing.py") == set()


# ── get_chunks_by_ids ───────────────────────────────────────────────────────


def test_get_chunks_by_ids_returns_correct_metadata(adapter: DuckDBAdapter) -> None:
    chunk = _chunk("c1", text="def bar(): ...", start_line=5, end_line=7)
    adapter.upsert_chunk_nodes([chunk])

    result = adapter.get_chunks_by_ids(["c1"])

    assert len(result) == 1
    row = result[0]
    assert row["id"] == "c1"
    assert row["path"] == "a/b.py"
    assert row["start_line"] == 5
    assert row["end_line"] == 7
    assert row["text"] == "def bar(): ..."
    assert row["chunk_kind"] == "ast"


def test_get_chunks_by_ids_returns_the_stored_symbol(adapter: DuckDBAdapter) -> None:
    adapter.upsert_chunk_nodes([_chunk("c1", symbol="PermissionChecker")])

    assert adapter.get_chunks_by_ids(["c1"])[0]["symbol"] == "PermissionChecker"


def test_get_chunks_by_ids_returns_no_symbol_when_the_chunker_knew_none(
    adapter: DuckDBAdapter,
) -> None:
    adapter.upsert_chunk_nodes([_chunk("c1")])

    assert adapter.get_chunks_by_ids(["c1"])[0]["symbol"] is None


def test_get_chunks_by_ids_returns_empty_list_for_empty_input(
    adapter: DuckDBAdapter,
) -> None:
    assert adapter.get_chunks_by_ids([]) == []


def test_get_chunks_by_ids_ignores_unknown_ids(adapter: DuckDBAdapter) -> None:
    assert adapter.get_chunks_by_ids(["does-not-exist"]) == []


# ── file hashes ─────────────────────────────────────────────────────────────


def test_get_file_hash_returns_none_when_unknown(adapter: DuckDBAdapter) -> None:
    assert adapter.get_file_hash("unknown.py") is None


def test_set_and_get_file_hash_round_trips(adapter: DuckDBAdapter) -> None:
    adapter.set_file_hash("foo.py", "abc123")

    assert adapter.get_file_hash("foo.py") == "abc123"


def test_set_file_hash_is_idempotent(adapter: DuckDBAdapter) -> None:
    adapter.set_file_hash("foo.py", "old")
    adapter.set_file_hash("foo.py", "new")

    assert adapter.get_file_hash("foo.py") == "new"


def test_get_all_file_paths_returns_all_tracked_paths(adapter: DuckDBAdapter) -> None:
    adapter.set_file_hash("a.py", "h1")
    adapter.set_file_hash("b.py", "h2")

    assert adapter.get_all_file_paths() == {"a.py", "b.py"}


# ── delete_file ─────────────────────────────────────────────────────────────


def test_delete_file_removes_nodes_and_hash(adapter: DuckDBAdapter) -> None:
    adapter.upsert_file_node(_source("del.py"))
    adapter.upsert_chunk_nodes([_chunk("c1", path="del.py")])
    adapter.set_file_hash("del.py", "h")

    adapter.delete_file("del.py")

    assert adapter.get_chunk_ids_for_file("del.py") == set()
    assert adapter.get_file_hash("del.py") is None
    count = adapter.con.execute(
        "SELECT count(*) FROM nodes WHERE path = 'del.py'"
    ).fetchone()[0]
    assert count == 0


def test_delete_file_does_not_touch_other_files(adapter: DuckDBAdapter) -> None:
    adapter.upsert_chunk_nodes([_chunk("keep-c", path="keep.py")])
    adapter.set_file_hash("keep.py", "h")
    adapter.upsert_chunk_nodes([_chunk("del-c", path="del.py")])
    adapter.set_file_hash("del.py", "h")

    adapter.delete_file("del.py")

    assert adapter.get_chunk_ids_for_file("keep.py") == {"keep-c"}
    assert adapter.get_file_hash("keep.py") == "h"


# ── clear_caches ─────────────────────────────────────────────────────────────


def test_clear_caches_empties_file_hashes(adapter: DuckDBAdapter) -> None:
    """file_hashes rows are deleted; memo_cache absent on first run is skipped."""
    adapter.set_file_hash("a.py", "h1")
    adapter.set_file_hash("b.py", "h2")

    adapter.clear_caches()

    assert adapter.get_all_file_paths() == set()


def test_clear_caches_empties_memo_cache_when_present(adapter: DuckDBAdapter) -> None:
    """Both tables are cleared when memo_cache exists (i.e. after a prior index run)."""
    adapter.set_file_hash("a.py", "h1")
    adapter.con.execute(
        "CREATE TABLE IF NOT EXISTS memo_cache ("
        "  component_path TEXT, cache_key TEXT, result BLOB,"
        "  created_at TIMESTAMP DEFAULT now(),"
        "  PRIMARY KEY (component_path, cache_key)"
        ")"
    )
    adapter.con.execute("INSERT INTO memo_cache VALUES ('p', 'k', NULL, now())")

    adapter.clear_caches()

    assert adapter.get_all_file_paths() == set()
    assert adapter.con.execute("SELECT COUNT(*) FROM memo_cache").fetchone()[0] == 0


def test_clear_caches_is_noop_on_empty_tables(adapter: DuckDBAdapter) -> None:
    """No error when tables exist but are already empty."""
    adapter.clear_caches()  # must not raise


def test_clear_caches_does_not_touch_nodes_table(adapter: DuckDBAdapter) -> None:
    """Chunk/file nodes are preserved; only the incremental caches are wiped."""
    adapter.upsert_chunk_nodes([_chunk("c1")])  # default path: "a/b.py"
    adapter.set_file_hash("a/b.py", "h1")

    adapter.clear_caches()

    assert adapter.get_chunk_ids_for_file("a/b.py") == {"c1"}  # nodes untouched
    assert adapter.get_all_file_paths() == set()  # hashes cleared


def test_clear_nodes_empties_the_whole_nodes_table(adapter: DuckDBAdapter) -> None:
    """Chunk and file nodes both go — it is a DELETE over the table."""
    adapter.upsert_chunk_nodes([_chunk("c1")])
    adapter.upsert_file_node(_source("a/b.py"))

    adapter.clear_nodes()

    assert adapter.get_chunk_ids_for_file("a/b.py") == set()
    assert adapter.con.execute("SELECT count(*) FROM nodes").fetchone()[0] == 0


def test_clear_nodes_leaves_the_incremental_caches_alone(
    adapter: DuckDBAdapter,
) -> None:
    """Clearing the caches is clear_caches' job; build_index calls both."""
    adapter.upsert_chunk_nodes([_chunk("c1")])
    adapter.set_file_hash("a/b.py", "h1")

    adapter.clear_nodes()

    assert adapter.get_all_file_paths() == {"a/b.py"}
