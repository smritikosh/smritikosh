"""Coverage-aware diverse candidate selection."""

from __future__ import annotations

from collections import Counter

from smritikosh.models.retrieval import HybridSearchOptions, RankedCandidate
from smritikosh.retrieval.tokenizer import tokenize_code

__all__ = ["select_seeds"]


def _tokens(candidate: RankedCandidate) -> set[str]:
    result = candidate.result
    return set(
        tokenize_code(" ".join((result.path, result.symbol or "", result.snippet)))
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union: set[str] = left | right
    return len(left & right) / len(union) if union else 0.0


def _can_select(
    candidate: RankedCandidate,
    *,
    selected_ids: set[int],
    file_counts: Counter[str],
    max_per_file: int,
) -> bool:
    return (
        id(candidate) not in selected_ids
        and file_counts[candidate.result.path] < max_per_file
    )


def select_seeds(
    candidates: list[RankedCandidate],
    facets: tuple[str, ...],
    *,
    options: HybridSearchOptions,
) -> list[RankedCandidate]:
    """Reserve facet coverage, then apply MMR for relevance and diversity."""
    selected: list[RankedCandidate] = []
    selected_ids: set[int] = set()
    file_counts: Counter[str] = Counter()
    for facet in facets:
        eligible: list[RankedCandidate] = [
            candidate for candidate in candidates if facet in candidate.facets
        ]
        eligible.sort(
            key=lambda candidate: (
                -candidate.facet_scores.get(facet, 0.0),
                -candidate.score.fused,
                candidate.result.path,
            )
        )
        match: RankedCandidate | None = next(
            (
                candidate
                for candidate in eligible
                if id(candidate) in selected_ids
                or file_counts[candidate.result.path] < options.max_per_file
            ),
            None,
        )
        if match is not None:
            if id(match) in selected_ids:
                continue
            selected.append(match)
            selected_ids.add(id(match))
            file_counts[match.result.path] += 1
        if len(selected) == options.max_seeds:
            return selected

    maximum_relevance: float = max(
        (candidate.score.fused for candidate in candidates),
        default=1.0,
    )
    token_cache: dict[int, set[str]] = {
        id(candidate): _tokens(candidate) for candidate in candidates
    }
    while len(selected) < options.max_seeds:
        available: list[RankedCandidate] = [
            candidate
            for candidate in candidates
            if _can_select(
                candidate,
                selected_ids=selected_ids,
                file_counts=file_counts,
                max_per_file=options.max_per_file,
            )
        ]
        if not available:
            break

        def utility(candidate: RankedCandidate) -> tuple[float, str, int]:
            relevance: float = candidate.score.fused / maximum_relevance
            redundancy: float = max(
                (
                    _jaccard(
                        token_cache[id(candidate)],
                        token_cache[id(chosen)],
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            score: float = (
                options.mmr_lambda * relevance - (1.0 - options.mmr_lambda) * redundancy
            )
            return (score, candidate.result.path, -candidate.result.start_line)

        chosen: RankedCandidate = max(available, key=utility)
        selected.append(chosen)
        selected_ids.add(id(chosen))
        file_counts[chosen.result.path] += 1
    return selected
