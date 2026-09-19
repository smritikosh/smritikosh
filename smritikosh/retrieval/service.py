"""Application service for bounded hybrid code search."""

from __future__ import annotations

from smritikosh.models.retrieval import (
    HybridSearchOptions,
    RankedCandidate,
    SearchLocation,
)
from smritikosh.ports.retrieval import CandidateRetriever
from smritikosh.ports.source_reader import SourceReader
from smritikosh.retrieval.definitions import complete_definitions
from smritikosh.retrieval.hybrid import HybridRetriever
from smritikosh.retrieval.priors import apply_metadata_priors, infer_topic
from smritikosh.retrieval.selection import select_seeds

__all__ = ["HybridSearchService"]


def _reported_facets(
    candidate: RankedCandidate,
    facets: tuple[str, ...],
) -> tuple[str, ...]:
    """Name the facets a candidate answers, in the caller's argument order.

    Positional order lets a renderer label facets without re-deriving them.

    Facet credit stops at a channel's leading ranks, so a candidate retrieved
    for exactly one facet can arrive with no credit at all while still holding
    that facet's score.  Falling back to its best-scoring facet keeps it from
    surfacing as though it matched nothing.
    """
    credited: tuple[str, ...] = tuple(
        facet for facet in facets if facet in candidate.facets
    )
    if credited:
        return credited
    best: str | None = max(
        (facet for facet in facets if facet in candidate.facet_scores),
        key=lambda facet: candidate.facet_scores[facet],
        default=None,
    )
    return () if best is None else (best,)


class HybridSearchService:
    """Locate high-recall definitions through independent retrieval ports."""

    def __init__(
        self,
        retrievers: tuple[CandidateRetriever, ...],
        reader: SourceReader,
    ) -> None:
        self._retrievers = retrievers
        self._reader = reader
        self._hybrid = HybridRetriever(retrievers)

    def search(
        self,
        facets: tuple[str, ...],
        *,
        options: HybridSearchOptions | None = None,
    ) -> list[SearchLocation]:
        """Retrieve, fuse, and select locations for caller-supplied facets."""
        options = options or HybridSearchOptions()
        normalized: tuple[str, ...] = tuple(
            facet.strip() for facet in facets if facet.strip()
        )
        if not normalized:
            raise ValueError("At least one retrieval facet is required")
        topic, topic_paths = infer_topic(self._reader, normalized[0])
        queries: dict[str, str] = {
            facet: (
                facet if topic is None or topic in facet.lower() else f"{topic} {facet}"
            )
            for facet in normalized
        }
        candidates: list[RankedCandidate] = self._hybrid.retrieve(
            normalized,
            options=options,
            queries=queries,
        )
        candidates = apply_metadata_priors(
            candidates,
            topic_paths=topic_paths,
            options=options,
        )
        seeds: list[RankedCandidate] = select_seeds(
            candidates,
            normalized,
            options=options,
        )
        located: list[RankedCandidate] = complete_definitions(
            seeds,
            reader=self._reader,
            options=options,
        )
        return [
            SearchLocation(
                path=candidate.result.path,
                start_line=candidate.result.start_line,
                end_line=candidate.result.end_line,
                symbol=candidate.result.symbol,
                facets=_reported_facets(candidate, normalized),
            )
            for candidate in located
        ]
