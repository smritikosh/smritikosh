"""Vector search index — semantic search over embedded code chunks."""

from __future__ import annotations

from typing import Any

from smritikosh.models import SearchResult
from smritikosh.ports.embedder import Embedder
from smritikosh.ports.storage import StorageAdapter
from smritikosh.ports.vector_store import VectorStore

__all__ = ["VectorIndex"]


class VectorIndex:
    """Semantic search over embedded code chunks."""

    def __init__(
        self,
        vector_store: VectorStore,
        storage: StorageAdapter,
        embedder: Embedder,
    ) -> None:
        self.vector_store = vector_store
        self.storage = storage
        self.embedder = embedder

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return the top-k chunks most similar to *query*, ranked by cosine score desc.

        Raises ValueError if top_k <= 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        # embed_query applies a query-specific task prefix (Voyage/Cohere aware).
        query_vector = await self.embedder.embed_query(query)

        hits = self.vector_store.search(query_vector, top_k)
        if not hits:
            return []

        chunk_ids = [h[0] for h in hits]
        score_map = {h[0]: h[1] for h in hits}

        chunk_metas: list[dict[str, Any]] = self.storage.get_chunks_by_ids(chunk_ids)
        if not chunk_metas:
            return []

        results = [
            SearchResult(
                path=m["path"],
                start_line=m["start_line"],
                end_line=m["end_line"],
                snippet=m["text"],
                score=score_map[m["id"]],
                chunk_kind=m["chunk_kind"],
                symbol=m.get("symbol"),
                chunk_id=m["id"],
            )
            for m in chunk_metas
            if m["id"] in score_map
        ]

        # Re-sort: storage row order is undefined; vector_store.search order
        # is not preserved.
        results.sort(key=lambda r: r.score, reverse=True)
        return results
