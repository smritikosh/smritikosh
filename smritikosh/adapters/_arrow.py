"""Shared Arrow handoff for the DuckDB adapters."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb
    import pyarrow as pa

__all__ = ["INCOMING", "WRITE_LOCK", "insert_arrow"]

#: Relation name the batch is visible under inside *statement*.
INCOMING = "_incoming"

#: Serialises writers on a connection. ``register``/``unregister`` mutate
#: catalog state owned by the connection, and two threads doing it at once
#: deadlock inside DuckDB — with no INSERT involved, and whatever name is used.
#: Doing it inside another thread's open transaction instead invalidates the
#: pending result, so a writer must be able to hold this across a whole
#: transaction; hence reentrant. One lock, because the vector, lexical and memo
#: stores share a connection.
#:
#: Readers do not take it, so this does not make a shared connection
#: thread-safe: use one connection per thread. Separate connections to the same
#: file are unaffected.
WRITE_LOCK = threading.RLock()


def insert_arrow(
    connection: duckdb.DuckDBPyConnection,
    incoming: pa.Table,
    statement: str,
) -> None:
    """Execute *statement*, which reads the batch as the relation ``_incoming``."""
    with WRITE_LOCK:
        connection.register(INCOMING, incoming)
        try:
            connection.execute(statement)
        finally:
            connection.unregister(INCOMING)
