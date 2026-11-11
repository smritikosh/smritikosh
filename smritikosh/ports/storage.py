"""The storage contract — implementations live in adapters/storage/."""

from __future__ import annotations

from abc import ABC, abstractmethod

from smritikosh.models import Call, Chunk, Edge, Import, SourceFile, Symbol

__all__ = ["StorageAdapter"]


class StorageAdapter(ABC):
    """Persists file-level and chunk-level metadata for the indexing pipeline."""

    # ------------------------------------------------------------------ nodes

    @abstractmethod
    def upsert_file_node(self, source: SourceFile) -> None:
        """INSERT OR REPLACE a file node into the nodes table."""

    @abstractmethod
    def upsert_chunk_nodes(self, chunks: list[Chunk]) -> None:
        """Bulk INSERT OR REPLACE chunk nodes into the nodes table."""

    @abstractmethod
    def delete_chunk_node(self, chunk_id: str) -> None:
        """DELETE the chunk node with the given id."""

    @abstractmethod
    def get_chunk_ids_for_file(self, path: str) -> set[str]:
        """Return all chunk ids currently stored for *path*."""

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Return chunk metadata dicts for the given ids.

        Each dict contains: id, path, start_line, end_line, text, chunk_kind,
        symbol.
        """

    # -------------------------------------------------------------- file hashes

    @abstractmethod
    def get_all_file_paths(self) -> set[str]:
        """Return the paths of all files recorded in file_hashes."""

    @abstractmethod
    def get_file_hash(self, path: str) -> str | None:
        """Return the stored content hash for *path*, or None if unknown."""

    @abstractmethod
    def set_file_hash(self, path: str, hash: str) -> None:  # noqa: A002
        """INSERT OR REPLACE the content hash for *path*."""

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """Atomically remove all nodes, graph rows and the file_hash for *path*."""

    # ------------------------------------------------------------------ graph

    @abstractmethod
    def replace_graph(
        self,
        path: str,
        symbols: list[Symbol],
        calls: list[Call],
        imports: list[Import],
    ) -> None:
        """Replace the symbols, calls and imports recorded for *path*."""

    @abstractmethod
    def load_graph(self) -> tuple[list[Symbol], list[Call], list[Import]]:
        """Return every stored symbol, call and import."""

    @abstractmethod
    def replace_edges(self, edges: list[Edge]) -> None:
        """Replace all resolved call edges."""

    # ------------------------------------------------------------------ reset

    @abstractmethod
    def clear_nodes(self) -> None:
        """Remove all indexed source nodes and the call graph."""

    @abstractmethod
    def clear_caches(self) -> None:
        """Empty the memo cache and the file hashes — forces a full re-index.

        Must run alongside :meth:`clear_nodes` on a full rebuild: dropping the
        nodes while the caches still claim every file is up to date leaves the
        rebuild with nothing to write back.
        """
