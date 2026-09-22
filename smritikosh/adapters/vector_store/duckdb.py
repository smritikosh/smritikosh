"""DuckDB vector store -- one .duckdb file, native array_cosine_similarity."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Final

import duckdb

from smritikosh.constants import DEFAULT_DB_PATH
from smritikosh.ports.vector_store import VectorStore

__all__ = ["DuckDBVectorStore"]

logger = logging.getLogger(__name__)

DIMS_KEY: Final = "embedder_dims"
INCREMENTAL_CACHES: Final = ("file_hashes", "memo_cache")


class DuckDBVectorStore(VectorStore):
    """Stores vectors in a DuckDB FLOAT[dims] column."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        *,
        con: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self._con = con or duckdb.connect(db_path)
        self._owns_con = con is None
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)"
        )
        # Recover width from the file so a reopened store can search without setup().
        self._dims = self.get_stored_dims()

    def setup(self, dims: int) -> None:
        # dims is interpolated below -- DuckDB cannot bind an array width.
        if dims <= 0:
            raise ValueError(f"dims must be positive, got {dims}")

        stored = self.get_stored_dims()
        if stored != dims:
            self._reset_for_new_model(stored, dims)

        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS vectors ("  # noqa: S608
            f"chunk_id TEXT PRIMARY KEY, "
            f"vector FLOAT[{dims}], "
            f"updated_at TIMESTAMP DEFAULT now())"
        )
        self._con.execute(
            "INSERT OR REPLACE INTO kv_store VALUES (?, ?)", [DIMS_KEY, str(dims)]
        )
        self._dims = dims

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        self.upsert_many([(chunk_id, vector)])

    def upsert_many(self, items: Sequence[tuple[str, list[float]]]) -> None:
        """Write several vectors in one statement, via Arrow.

        Binding a vector as a Python list makes DuckDB convert it one float at
        a time: measured at ~120 us per element, so a 768-wide vector costs
        ~30 ms to store and the write dominates a whole index run.  Cost scales
        linearly with width, confirming it is per-element work rather than
        per-statement — batching the SQL alone changes nothing.

        An Arrow FixedSizeListArray is already a contiguous float32 buffer in
        DuckDB's own layout, so ingesting it skips that conversion entirely.
        Measured on 1063 vectors of 768 dims: 32.4 s row-by-row, 0.57 s here.
        """
        if not items:
            return

        import numpy as np
        import pyarrow as pa

        dims = len(items[0][1])
        flat = np.fromiter(
            (value for _, vector in items for value in vector),
            dtype=np.float32,
            count=len(items) * dims,
        )
        incoming = pa.table(
            {
                "chunk_id": pa.array([chunk_id for chunk_id, _ in items]),
                "vector": pa.FixedSizeListArray.from_arrays(pa.array(flat), dims),
            }
        )
        # Registered under a private name so a concurrent caller cannot see a
        # half-built relation; the INSERT consumes it immediately.
        self._con.register("_incoming_vectors", incoming)
        try:
            self._con.execute(
                "INSERT OR REPLACE INTO vectors (chunk_id, vector) "
                "SELECT chunk_id, vector FROM _incoming_vectors"
            )
        finally:
            self._con.unregister("_incoming_vectors")

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if self._dims is None:
            raise RuntimeError("setup() must run before search()")

        rows = self._con.execute(
            f"""
            SELECT chunk_id, array_cosine_similarity(vector, ?::FLOAT[{self._dims}])
                AS score
            FROM vectors
            ORDER BY score DESC
            LIMIT ?
            """,  # noqa: S608
            [query_vector, top_k],
        ).fetchall()
        return [(chunk_id, score) for chunk_id, score in rows]

    def delete(self, chunk_id: str) -> None:
        self._con.execute("DELETE FROM vectors WHERE chunk_id = ?", [chunk_id])

    def delete_many(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        placeholders = ", ".join("?" * len(ids))
        self._con.execute(
            f"DELETE FROM vectors WHERE chunk_id IN ({placeholders})",  # noqa: S608
            ids,
        )

    def clear(self) -> None:
        """Remove every stored vector without changing model metadata."""
        if self._table_exists("vectors"):
            self._con.execute("DELETE FROM vectors")

    def exists(self, chunk_id: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM vectors WHERE chunk_id = ?", [chunk_id]
        ).fetchone()
        return row is not None

    def get_stored_dims(self) -> int | None:
        row = self._con.execute(
            "SELECT value FROM kv_store WHERE key = ?", [DIMS_KEY]
        ).fetchone()
        return int(row[0]) if row else None

    def close(self) -> None:
        if self._owns_con:
            self._con.close()

    def _reset_for_new_model(self, stored: int | None, dims: int) -> None:
        # A different model makes every stored vector meaningless, and the
        # incremental caches would otherwise skip re-embedding the whole repo.
        if stored is not None:
            logger.warning(
                "Embedding dims changed from %s to %s; dropping stored vectors",
                stored,
                dims,
            )

        self._con.execute("DROP TABLE IF EXISTS vectors")
        for table in INCREMENTAL_CACHES:
            if self._table_exists(table):
                self._con.execute(f"DELETE FROM {table}")  # noqa: S608

    def _table_exists(self, name: str) -> bool:
        # file_hashes and memo_cache belong to S6 and S2; absent during Wave 1.
        row = self._con.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = ?", [name]
        ).fetchone()
        return row is not None
