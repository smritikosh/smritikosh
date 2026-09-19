"""Rank-based fusion for independent retrieval channels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    CandidateScore,
    RankedCandidate,
    RetrievalChannel,
)

__all__ = ["fuse_ranked_results"]

_FACET_COVERAGE_RANK: Final[int] = 10


@dataclass
class _Accumulator:
    result: SearchResult
    facets: set[str] = field(default_factory=set)
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused: float = 0.0
    facet_scores: dict[str, float] = field(default_factory=dict)


def _same_definition(left: SearchResult, right: SearchResult) -> bool:
    if (
        left.path != right.path
        or left.chunk_kind != right.chunk_kind
        or left.symbol != right.symbol
    ):
        return False
    if left.symbol is None:
        return (left.chunk_id is not None and left.chunk_id == right.chunk_id) or (
            left.start_line == right.start_line and left.end_line == right.end_line
        )
    return max(left.start_line, right.start_line) <= min(
        left.end_line,
        right.end_line,
    )


def _merge_span(left: SearchResult, right: SearchResult) -> SearchResult:
    return replace(
        left,
        start_line=min(left.start_line, right.start_line),
        end_line=max(left.end_line, right.end_line),
    )


def _canonicalize(results: list[SearchResult]) -> list[SearchResult]:
    canonical: list[SearchResult] = []
    for result in results:
        match_index: int | None = next(
            (
                index
                for index, existing in enumerate(canonical)
                if _same_definition(existing, result)
            ),
            None,
        )
        if match_index is None:
            canonical.append(result)
        else:
            canonical[match_index] = _merge_span(
                canonical[match_index],
                result,
            )
    return canonical


def fuse_ranked_results(
    ranked: dict[str, dict[RetrievalChannel, list[SearchResult]]],
    *,
    rrf_k: int = 60,
) -> list[RankedCandidate]:
    """Fuse dense and lexical ranks with Reciprocal Rank Fusion."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    accumulated: list[_Accumulator] = []
    for facet, channels in ranked.items():
        for channel, results in channels.items():
            for rank, result in enumerate(_canonicalize(results), start=1):
                item: _Accumulator | None = next(
                    (
                        existing
                        for existing in accumulated
                        if _same_definition(existing.result, result)
                    ),
                    None,
                )
                if item is None:
                    item = _Accumulator(result=result)
                    accumulated.append(item)
                else:
                    item.result = _merge_span(item.result, result)
                contribution: float = 1.0 / (rrf_k + rank)
                if rank <= _FACET_COVERAGE_RANK:
                    item.facets.add(facet)
                item.fused += contribution
                item.facet_scores[facet] = (
                    item.facet_scores.get(facet, 0.0) + contribution
                )
                if channel is RetrievalChannel.DENSE:
                    item.dense_rank = min(item.dense_rank or rank, rank)
                elif channel is RetrievalChannel.LEXICAL:
                    item.lexical_rank = min(item.lexical_rank or rank, rank)
    candidates: list[RankedCandidate] = [
        RankedCandidate(
            result=item.result,
            facets=item.facets,
            facet_scores=item.facet_scores,
            score=CandidateScore(
                dense_rank=item.dense_rank,
                lexical_rank=item.lexical_rank,
                fused=item.fused,
            ),
        )
        for item in accumulated
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.score.fused,
            candidate.result.path,
            candidate.result.start_line,
        )
    )
    return candidates
