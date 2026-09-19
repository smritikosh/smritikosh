"""Widen selected chunks to the definitions they belong to."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    HybridSearchOptions,
    OutlineEntry,
    RankedCandidate,
)
from smritikosh.ports.source_reader import SourceReader

__all__ = ["complete_definitions"]

#: Chunk kinds that name one definition an outline entry can be widened to.
#: A grouped or whole-file chunk covers no single definition, so the span it
#: already carries is the honest one.
_COMPLETABLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "class",
        "function",
        "method",
    }
)


def complete_definitions(
    seeds: list[RankedCandidate],
    *,
    reader: SourceReader,
    options: HybridSearchOptions,
) -> list[RankedCandidate]:
    """Widen each selected candidate to its whole definition, keeping order.

    Args:
        seeds: Selected candidates, in the order they should be reported.
        reader: Index to read outlines from.
        options: Supplies the cap on how many candidates are returned.

    Returns:
        The candidates, widened and deduplicated, capped at
        ``options.max_results``.
    """
    outlines: dict[str, list[OutlineEntry]] = {}
    completed: list[RankedCandidate] = []
    known: set[tuple[str, str, str | None]] = set()
    for seed in seeds:
        widened: SearchResult = _complete(seed.result, reader, outlines)
        _append_unique(completed, known, replace(seed, result=widened))
    return completed[: options.max_results]


def _complete(
    result: SearchResult,
    reader: SourceReader,
    outlines: dict[str, list[OutlineEntry]],
) -> SearchResult:
    """Return *result* widened to the definition its chunk sits inside.

    A definition too large for one chunk is stored as several windows, so a
    hit on one of them reports a slice of the definition rather than the
    definition.  The outline merges those windows back together, which is
    what makes the whole span available to widen to.
    """
    if result.symbol is None or result.chunk_kind not in _COMPLETABLE_KINDS:
        return result
    if result.path not in outlines:
        outlines[result.path] = reader.get_outline(result.path)
    entry: OutlineEntry | None = next(
        (
            candidate
            for candidate in outlines[result.path]
            if candidate.symbol == result.symbol
            and candidate.chunk_kind == result.chunk_kind
            and candidate.start_line <= result.end_line
            and candidate.end_line >= result.start_line
        ),
        None,
    )
    if entry is None:
        return result
    return replace(result, start_line=entry.start_line, end_line=entry.end_line)


def _append_unique(
    candidates: list[RankedCandidate],
    known: set[tuple[str, str, str | None]],
    candidate: RankedCandidate,
) -> None:
    """Append *candidate* unless the same location is already present.

    Widening is what makes this necessary: two windows of one oversized
    definition are two distinct chunks with distinct spans, so nothing
    upstream treats them as duplicates, but once both are widened they name
    the same lines.
    """
    result: SearchResult = candidate.result
    identity: str = result.symbol or f"@{result.start_line}:{result.end_line}"
    key: tuple[str, str, str | None] = (
        result.path,
        identity,
        result.chunk_kind,
    )
    if key not in known:
        known.add(key)
        candidates.append(candidate)
