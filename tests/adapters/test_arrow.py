"""Tests for the shared Arrow handoff and the lock that guards it."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
import threading
from unittest.mock import MagicMock

import duckdb
import pytest

from smritikosh.adapters._arrow import INCOMING, WRITE_LOCK, insert_arrow
from smritikosh.adapters.retrieval.duckdb import DuckDBBm25Store
from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore
from smritikosh.models import Chunk

X_AXIS = [1.0, 0.0, 0.0, 0.0]
Y_AXIS = [0.0, 1.0, 0.0, 0.0]


def _chunk(chunk_id: str, path: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        path=path,
        start_line=1,
        end_line=1,
        text=text,
        chunk_kind="function",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        symbol=None,
    )


def _recording_connection(
    connection: duckdb.DuckDBPyConnection, method: str, sink: list[bool]
) -> MagicMock:
    """Wrap *connection*, noting whether WRITE_LOCK is held when *method* runs.

    A wrapper because DuckDB connection attributes are read-only.
    """
    wrapper = MagicMock(wraps=connection)
    original = getattr(connection, method)

    def record(*args: object, **kwargs: object) -> object:
        sink.append(_lock_is_held_elsewhere())
        return original(*args, **kwargs)

    getattr(wrapper, method).side_effect = record
    return wrapper


def _lock_is_held_elsewhere() -> bool:
    """Whether another thread can take WRITE_LOCK right now.

    Asked from a fresh thread because the lock is reentrant: the holding thread
    would always succeed and prove nothing.
    """
    acquired: list[bool] = []

    def probe() -> None:
        if WRITE_LOCK.acquire(blocking=False):
            acquired.append(True)
            WRITE_LOCK.release()
        else:
            acquired.append(False)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(timeout=5)
    assert acquired, "probe thread did not run"
    return not acquired[0]


# ── insert_arrow ──────────────────────────────────────────────────────────────


def test_should_hold_the_lock_while_the_statement_runs() -> None:
    """Registering under a shared name is only safe while nobody else can."""
    pa = pytest.importorskip("pyarrow")
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE sink (value INTEGER)")
    held: list[bool] = []

    insert_arrow(
        _recording_connection(connection, "execute", held),
        pa.table({"value": pa.array([1, 2, 3])}),
        f"INSERT INTO sink SELECT * FROM {INCOMING}",
    )

    assert held == [True]
    assert _lock_is_held_elsewhere() is False  # released afterwards
    assert connection.execute("SELECT count(*) FROM sink").fetchone() == (3,)


def test_should_release_the_lock_when_the_statement_fails() -> None:
    pa = pytest.importorskip("pyarrow")
    connection = duckdb.connect(":memory:")

    with pytest.raises(duckdb.CatalogException):
        insert_arrow(
            connection,
            pa.table({"value": pa.array([1])}),
            f"INSERT INTO no_such_table SELECT * FROM {INCOMING}",
        )

    assert _lock_is_held_elsewhere() is False


def test_should_unregister_the_relation_when_the_statement_fails() -> None:
    """A leaked registration would be read by the next caller's statement."""
    pa = pytest.importorskip("pyarrow")
    connection = duckdb.connect(":memory:")

    with pytest.raises(duckdb.CatalogException):
        insert_arrow(
            connection,
            pa.table({"value": pa.array([1])}),
            f"INSERT INTO no_such_table SELECT * FROM {INCOMING}",
        )

    with pytest.raises(duckdb.CatalogException):
        connection.execute(f"SELECT * FROM {INCOMING}")


# ── Adapter call sites ────────────────────────────────────────────────────────


def test_vector_store_should_hold_the_lock_for_its_write() -> None:
    pytest.importorskip("pyarrow")
    connection = duckdb.connect(":memory:")
    store = DuckDBVectorStore(con=connection)
    store.setup(4)
    held: list[bool] = []

    store._con = _recording_connection(connection, "execute", held)
    store.upsert_many([("a", X_AXIS)])

    assert held == [True]
    assert connection.execute("SELECT count(*) FROM vectors").fetchone() == (1,)


def test_lexical_store_should_hold_the_lock_across_its_transaction() -> None:
    """A batch registered between begin() and commit() loses the transaction."""
    pytest.importorskip("pyarrow")
    connection = duckdb.connect(":memory:")
    store = DuckDBBm25Store(con=connection)
    store.setup()
    held_at_commit: list[bool] = []

    store._connection = _recording_connection(connection, "commit", held_at_commit)
    store.upsert([_chunk("one", "src/a.py", "load and store")])

    assert held_at_commit == [True]
    assert connection.execute("SELECT count(*) FROM lexical_documents").fetchone() == (
        1,
    )


# ── End to end ────────────────────────────────────────────────────────────────

#: Out of process because the failure it guards is a deadlock inside DuckDB that
#: holds the GIL: in-process, it would freeze the session rather than fail, and
#: ``Thread.join(timeout=...)`` would never get a turn.
_CONCURRENT_WRITERS = """
    import threading

    import duckdb

    from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore

    N = 60
    connection = duckdb.connect(":memory:")
    store = DuckDBVectorStore(con=connection)
    store.setup(4)

    def write(tag):
        for i in range(N):
            store.upsert_many([(f"{tag}{i}", [1.0, 0.0, 0.0, 0.0])])

    threads = [threading.Thread(target=write, args=(t,)) for t in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stored = connection.execute("SELECT count(*) FROM vectors").fetchone()[0]
    assert stored == 2 * N, f"stored {stored} of {2 * N}"
"""


def test_two_writers_on_one_connection_should_both_complete() -> None:
    pytest.importorskip("pyarrow")

    try:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_CONCURRENT_WRITERS)],
            capture_output=True,
            text=True,
            timeout=30,  # the happy path runs in under a second
        )
    except subprocess.TimeoutExpired:
        pytest.fail("concurrent writers deadlocked — WRITE_LOCK is not serialising")

    assert completed.returncode == 0, completed.stderr
