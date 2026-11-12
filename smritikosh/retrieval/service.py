"""Application service for bounded hybrid code search."""

from __future__ import annotations

from dataclasses import dataclass

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    ExplainedHit,
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
        return [hit.location for hit in self.explained(facets, options=options)]

    def explained(
        self,
        facets: tuple[str, ...],
        *,
        options: HybridSearchOptions | None = None,
    ) -> list[ExplainedHit]:
        """Select the same locations as :meth:`search`, plus ranks and chunk span."""
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
        origins = _chunk_origins(seeds)
        located: list[RankedCandidate] = complete_definitions(
            seeds,
            reader=self._reader,
            options=options,
        )
        hits: list[ExplainedHit] = []
        for candidate in located:
            result = candidate.result
            origin = _origin_for(result, origins)
            hits.append(
                ExplainedHit(
                    location=SearchLocation(
                        path=result.path,
                        start_line=result.start_line,
                        end_line=result.end_line,
                        symbol=result.symbol,
                        facets=_reported_facets(candidate, normalized),
                    ),
                    dense_rank=candidate.score.dense_rank,
                    lexical_rank=candidate.score.lexical_rank,
                    chunk_start_line=origin[0],
                    chunk_end_line=origin[1],
                )
            )
        return hits


@dataclass(frozen=True)
class _Origins:
    by_span: dict[tuple[str, tuple[int, int]], tuple[int, int]]
    by_symbol: dict[tuple[str, str | None, str | None], tuple[int, int]]


def _chunk_origins(seeds: list[RankedCandidate]) -> _Origins:
    """Remember each seed's chunk span, by exact span and by symbol."""
    origins = _Origins(by_span={}, by_symbol={})
    for seed in seeds:
        span = (seed.result.start_line, seed.result.end_line)
        origins.by_span[(seed.result.path, span)] = span
        origins.by_symbol.setdefault(
            (seed.result.path, seed.result.symbol, seed.result.chunk_kind),
            span,
        )
    return origins


def _origin_for(result: SearchResult, origins: _Origins) -> tuple[int, int]:
    """Return the chunk span a widened result came from."""
    span = (result.start_line, result.end_line)
    exact = origins.by_span.get((result.path, span))
    if exact is not None:
        return exact
    return origins.by_symbol.get(
        (result.path, result.symbol, result.chunk_kind),
        span,
    )
