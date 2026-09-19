"""DuckDB storage adapter — nodes and file_hashes tables in one .duckdb file."""

from __future__ import annotations

import hashlib
import json

import duckdb

from smritikosh.constants import DEFAULT_DB_PATH
from smritikosh.models import Chunk, SourceFile
from smritikosh.ports.storage import StorageAdapter

__all__ = ["DuckDBAdapter"]


class DuckDBAdapter(StorageAdapter):
    """Persists file and chunk nodes in a DuckDB `nodes` table.

    The ``con`` attribute is intentionally public so that
    :class:`~smritikosh.adapters.vector_store.duckdb.DuckDBVectorStore` and
    ``MemoizationStore`` (engine) can share the same database connection.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        *,
        con: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.con = con or duckdb.connect(db_path)
        self._owns_con = con is None
        self._create_tables()

    # ------------------------------------------------------------------ setup

    def _create_tables(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id         TEXT PRIMARY KEY,
                kind       TEXT NOT NULL,
                name       TEXT,
                path       TEXT,
                metadata   JSON,
                updated_at TIMESTAMP DEFAULT now()
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hashes (
                path       TEXT PRIMARY KEY,
                hash       TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT now()
            )
            """
        )

    # ------------------------------------------------------------------ nodes

    def upsert_file_node(self, source: SourceFile) -> None:
        """Insert or update the ``file`` node for *source*.

        Keyed by path, not by content hash.  Two files with identical bytes are
        still two files: hashing the content made them share a primary key, so
        ``INSERT OR REPLACE`` silently collapsed them into one row — a repo with
        24 empty ``__init__.py`` files kept exactly one of them.  The content
        hash is still recorded in the metadata, which is where it is read from.
        """
        content_hash = hashlib.sha256(source.content.encode()).hexdigest()
        metadata = json.dumps(
            {"language": source.language, "content_hash": content_hash}
        )
        self.con.execute(
            """
            INSERT OR REPLACE INTO nodes (id, kind, name, path, metadata)
            VALUES (?, 'file', ?, ?, ?)
            """,
            [f"file:{source.path}", source.path, source.path, metadata],
        )

    def upsert_chunk_nodes(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            (
                chunk.id,
                chunk.symbol,
                chunk.path,
                json.dumps(
                    {
                        "text": chunk.text,
                        "chunk_kind": chunk.chunk_kind,
                        "content_hash": chunk.content_hash,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }
                ),
            )
            for chunk in chunks
        ]
        self.con.executemany(
            """
            INSERT OR REPLACE INTO nodes (id, kind, name, path, metadata)
            VALUES (?, 'chunk', ?, ?, ?)
            """,
            rows,
        )

    def delete_chunk_node(self, chunk_id: str) -> None:
        self.con.execute(
            "DELETE FROM nodes WHERE id = ? AND kind = 'chunk'", [chunk_id]
        )

    def get_chunk_ids_for_file(self, path: str) -> set[str]:
        rows = self.con.execute(
            "SELECT id FROM nodes WHERE path = ? AND kind = 'chunk'", [path]
        ).fetchall()
        return {row[0] for row in rows}

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" * len(chunk_ids))
        rows = self.con.execute(
            f"SELECT id, name, path, metadata FROM nodes"  # noqa: S608
            f" WHERE id IN ({placeholders}) AND kind = 'chunk'",
            chunk_ids,
        ).fetchall()
        result = []
        for chunk_id, symbol, path, raw_meta in rows:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            result.append(
                {
                    "id": chunk_id,
                    "path": path,
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "text": meta["text"],
                    "chunk_kind": meta["chunk_kind"],
                    "symbol": symbol,
                }
            )
        return result

    # -------------------------------------------------------------- file hashes

    def get_all_file_paths(self) -> set[str]:
        rows = self.con.execute("SELECT path FROM file_hashes").fetchall()
        return {row[0] for row in rows}

    def get_file_hash(self, path: str) -> str | None:
        row = self.con.execute(
            "SELECT hash FROM file_hashes WHERE path = ?", [path]
        ).fetchone()
        return row[0] if row else None

    def set_file_hash(self, path: str, hash: str) -> None:  # noqa: A002
        self.con.execute(
            "INSERT OR REPLACE INTO file_hashes (path, hash) VALUES (?, ?)",
            [path, hash],
        )

    def delete_file(self, path: str) -> None:
        """Atomically remove all nodes and the file_hash row for *path*."""
        self.con.begin()
        try:
            self.con.execute("DELETE FROM nodes WHERE path = ?", [path])
            self.con.execute("DELETE FROM file_hashes WHERE path = ?", [path])
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def clear_caches(self) -> None:
        """Atomically empty ``memo_cache`` and ``file_hashes`` — forces a full re-index.

        Both tables are cleared in a single transaction so a crash mid-way
        cannot leave the incremental state partially cleared.
        Tables that do not exist yet (first run) are silently skipped.
        """
        self.con.begin()
        try:
            for table in ("memo_cache", "file_hashes"):
                row = self.con.execute(
                    "SELECT 1 FROM duckdb_tables() WHERE table_name = ?", [table]
                ).fetchone()
                if row:
                    self.con.execute(f"DELETE FROM {table}")  # noqa: S608
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def clear_nodes(self) -> None:
        """Remove all indexed source nodes."""
        self.con.begin()
        try:
            self.con.execute("DELETE FROM nodes")
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    # ------------------------------------------------------------------ misc

    def close(self) -> None:
        if self._owns_con:
            self.con.close()
