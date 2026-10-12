"""Hybrid candidate generation and rank fusion."""

from __future__ import annotations

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    HybridSearchOptions,
    RankedCandidate,
    RetrievalChannel,
    SearchOptions,
)
from smritikosh.ports.retrieval import CandidateRetriever
from smritikosh.retrieval.fusion import fuse_ranked_results

__all__ = ["HybridRetriever"]


class HybridRetriever:
    """Generate and fuse candidates from independent retrieval channels."""

    def __init__(self, retrievers: tuple[CandidateRetriever, ...]) -> None:
        self._retrievers = retrievers

    def retrieve(
        self,
        facets: tuple[str, ...],
        *,
        options: HybridSearchOptions,
        queries: dict[str, str] | None = None,
    ) -> list[RankedCandidate]:
        """Retrieve each facet independently and fuse channel ranks."""
        queries = queries or {facet: facet for facet in facets}
        # Documentation is not filtered out here.  The dense channel already
        # demotes non-production paths by query intent, and apply_metadata_priors
        # weighs them again after fusion, both case-insensitively over the whole
        # path — a LIKE list here can only restate that, less accurately.
        search_options = SearchOptions(
            top_k=options.candidates_per_channel,
            exclude_paths=options.exclude_paths,
        )
        ranked: dict[str, dict[RetrievalChannel, list[SearchResult]]] = {}
        for facet in facets:
            ranked[facet] = {
                retriever.channel: retriever.retrieve(
                    queries[facet],
                    options=search_options,
                )
                for retriever in self._retrievers
            }
        return fuse_ranked_results(ranked, rrf_k=options.rrf_k)[
            : options.max_candidates
        ]
