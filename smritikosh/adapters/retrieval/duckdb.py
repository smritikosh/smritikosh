"""DuckDB adapters for source reads, dense retrieval, and BM25."""

from __future__ import annotations

from collections import Counter
from typing import Final

import duckdb

from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.constants import DEFAULT_DB_PATH
from smritikosh.models import Chunk, SearchResult
from smritikosh.models.retrieval import IndexedChunk, RetrievalChannel, SearchOptions
from smritikosh.ports.embedder import Embedder
from smritikosh.ports.retrieval import LexicalStore
from smritikosh.retrieval.tokenizer import tokenize_code

__all__ = [
    "DuckDBBm25Store",
    "DuckDBDenseRetriever",
    "DuckDBLexicalRetriever",
    "DuckDBSourceReader",
]

_BM25_K1: Final[float] = 1.2
_BM25_B: Final[float] = 0.75
#: Bumped whenever the stored postings change shape or provenance, so that
#: build_index rebuilds them rather than mixing two tokenisations in one
#: corpus — the document lengths feed a corpus-wide average, so stale rows
#: skew the scores of fresh ones.
_LEXICAL_SCHEMA_VERSION: Final[str] = "2"


class DuckDBBm25Store(LexicalStore):
    """Maintain an incremental Okapi BM25 index in DuckDB tables."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        *,
        con: duckdb.DuckDBPyConnection | None = None,
        read_only: bool = False,
    ) -> None:
        self._connection = con or duckdb.connect(db_path, read_only=read_only)
        self._owns_connection: bool = con is None

    def setup(self) -> bool:
        """Create the lexical tables, reporting an index from an older schema.

        Emptying an incompatible index here would strand it: the postings are
        written by a memoised stage, so clearing them without the memo cache
        leaves every file a cache hit and nothing to refill them.  The caller
        clears both or neither.
        """
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_documents (
                chunk_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                length INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lexical_terms (
                chunk_id TEXT NOT NULL,
                term TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, term)
            );
            CREATE INDEX IF NOT EXISTS lexical_terms_term_idx
                ON lexical_terms(term);
            CREATE TABLE IF NOT EXISTS lexical_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        version_row: tuple[str] | None = self._connection.execute(
            "SELECT value FROM lexical_metadata WHERE key = 'schema_version'"
        ).fetchone()
        self._connection.execute(
            "INSERT OR REPLACE INTO lexical_metadata VALUES ('schema_version', ?)",
            [_LEXICAL_SCHEMA_VERSION],
        )
        return version_row is not None and version_row[0] != _LEXICAL_SCHEMA_VERSION

    def upsert(self, chunks: list[Chunk]) -> None:
        """Replace lexical documents and postings for a chunk batch."""
        if not chunks:
            return
        unique_chunks: dict[str, Chunk] = {chunk.id: chunk for chunk in chunks}
        chunk_ids: list[str] = list(unique_chunks)
        documents: list[tuple[str, str, int]] = []
        postings: list[tuple[str, str, int]] = []
        for chunk in unique_chunks.values():
            searchable: str = " ".join((chunk.path, chunk.symbol or "", chunk.text))
            frequencies: Counter[str] = Counter(tokenize_code(searchable))
            documents.append((chunk.id, chunk.path, sum(frequencies.values())))
            postings.extend(
                (chunk.id, term, frequency) for term, frequency in frequencies.items()
            )
        self._connection.begin()
        try:
            self._delete_many(chunk_ids)
            self._insert_arrow("lexical_documents", documents, ("chunk_id", "path"))
            self._insert_arrow("lexical_terms", postings, ("chunk_id", "term"))
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _insert_arrow(
        self,
        table: str,
        rows: list[tuple[str, str, int]],
        text_columns: tuple[str, str],
    ) -> None:
        """Insert *rows* through Arrow rather than ``executemany``.

        ``executemany`` binds each value individually, which costs ~470 us per
        posting row; a corpus producing 168k postings spent 80 s here.  Handing
        DuckDB three ready-made Arrow columns is the same data in one scan —
        measured at 0.5 us per row, a thousandfold difference.
        """
        if not rows:
            return

        import pyarrow as pa

        first, second = text_columns
        incoming = pa.table(
            {
                first: pa.array([row[0] for row in rows]),
                second: pa.array([row[1] for row in rows]),
                "count": pa.array([row[2] for row in rows], type=pa.int32()),
            }
        )
        self._connection.register("_incoming_lexical", incoming)
        try:
            self._connection.execute(
                f"INSERT INTO {table} SELECT * FROM _incoming_lexical"  # noqa: S608
            )
        finally:
            self._connection.unregister("_incoming_lexical")

    def delete(self, chunk_id: str) -> None:
        """Remove one lexical document and its postings."""
        self._delete_many([chunk_id])

    def delete_path(self, path: str) -> None:
        """Remove every lexical document belonging to a path."""
        rows: list[tuple[str]] = self._connection.execute(
            "SELECT chunk_id FROM lexical_documents WHERE path = ?",
            [path],
        ).fetchall()
        self._delete_many([row[0] for row in rows])

    def clear(self) -> None:
        """Remove every lexical document and posting."""
        self._connection.execute("DELETE FROM lexical_terms")
        self._connection.execute("DELETE FROM lexical_documents")

    def search(
        self,
        query: str,
        *,
        options: SearchOptions,
    ) -> list[tuple[str, float]]:
        """Return BM25-ranked chunk ids for a query."""
        terms: list[str] = list(dict.fromkeys(tokenize_code(query)))
        if not terms or options.top_k <= 0:
            return []
        placeholders: str = ", ".join("?" for _ in terms)
        clauses: list[str] = [f"t.term IN ({placeholders})"]
        filter_parameters: list[object] = list(terms)
        if options.include_paths:
            include: str = " OR ".join("d.path LIKE ?" for _ in options.include_paths)
            clauses.append(f"({include})")
            filter_parameters.extend(options.include_paths)
        for pattern in options.exclude_paths:
            clauses.append("d.path NOT LIKE ?")
            filter_parameters.append(pattern)
        where_clause: str = " AND ".join(clauses)
        try:
            rows: list[tuple[str, float]] = self._connection.execute(
                f"""
            WITH corpus AS (
                SELECT count(*)::DOUBLE AS document_count,
                       coalesce(avg(length), 1.0)::DOUBLE AS average_length
                FROM lexical_documents
            ),
            document_frequency AS (
                SELECT term, count(*)::DOUBLE AS document_frequency
                FROM lexical_terms
                WHERE term IN ({placeholders})
                GROUP BY term
            )
            SELECT t.chunk_id,
                   sum(
                       ln(
                           1.0 + (
                               corpus.document_count
                               - document_frequency.document_frequency
                               + 0.5
                           ) / (
                               document_frequency.document_frequency + 0.5
                           )
                       )
                       * (
                           t.term_frequency * (? + 1.0)
                       ) / (
                           t.term_frequency
                           + ? * (
                               1.0 - ?
                               + ? * d.length / corpus.average_length
                           )
                       )
                   ) AS score
            FROM lexical_terms AS t
            JOIN lexical_documents AS d USING (chunk_id)
            JOIN document_frequency USING (term)
            CROSS JOIN corpus
            WHERE {where_clause}
            GROUP BY t.chunk_id
            ORDER BY score DESC, t.chunk_id
            LIMIT ?
            """,  # noqa: S608
                [
                    *terms,
                    _BM25_K1,
                    _BM25_K1,
                    _BM25_B,
                    _BM25_B,
                    *filter_parameters,
                    options.top_k,
                ],
            ).fetchall()
        except duckdb.CatalogException as exc:
            raise RuntimeError(
                "BM25 index is missing; rebuild with `smritikosh index --full`"
            ) from exc
        return rows

    def close(self) -> None:
        """Close an owned DuckDB connection."""
        if self._owns_connection:
            self._connection.close()

    def _delete_many(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        placeholders: str = ", ".join("?" for _ in chunk_ids)
        self._connection.execute(
            f"DELETE FROM lexical_terms WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._connection.execute(
            f"DELETE FROM lexical_documents WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )


class DuckDBDenseRetriever:
    """Adapt semantic exploration into the candidate-retriever port."""

    channel: RetrievalChannel = RetrievalChannel.DENSE

    def __init__(self, reader: DuckDBSourceReader, embedder: Embedder) -> None:
        self._reader = reader
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        options: SearchOptions,
    ) -> list[SearchResult]:
        """Retrieve dense semantic candidates."""
        return self._reader.semantic_search(
            [query],
            embedder=self._embedder,
            options=options,
        )


class DuckDBLexicalRetriever:
    """Adapt BM25 postings and source metadata into retrieval candidates."""

    channel: RetrievalChannel = RetrievalChannel.LEXICAL

    def __init__(
        self,
        store: DuckDBBm25Store,
        reader: DuckDBSourceReader,
    ) -> None:
        self._store = store
        self._reader = reader

    def retrieve(
        self,
        query: str,
        *,
        options: SearchOptions,
    ) -> list[SearchResult]:
        """Retrieve BM25 candidates with source metadata."""
        hits: list[tuple[str, float]] = self._store.search(
            query,
            options=options,
        )
        # Walking the hits rather than the lookup keeps BM25's order, and an id
        # the reader has no chunk for drops out instead of shifting the scores
        # of everything ranked below it.
        chunks: dict[str, IndexedChunk] = self._reader.get_chunks_by_ids(
            [chunk_id for chunk_id, _ in hits]
        )
        return [
            SearchResult(
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                snippet=chunk.text,
                score=score,
                chunk_kind=chunk.chunk_kind,
                symbol=chunk.symbol,
                chunk_id=chunk.chunk_id,
            )
            for chunk_id, score in hits
            if (chunk := chunks.get(chunk_id)) is not None
        ]
