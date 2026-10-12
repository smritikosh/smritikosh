"""Configurable metadata priors applied after rank fusion."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Final

from smritikosh.models.retrieval import HybridSearchOptions, RankedCandidate
from smritikosh.ports.source_reader import SourceReader

__all__ = ["apply_metadata_priors", "infer_topic"]

_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+")
_DOCUMENTATION_TERMS: Final[frozenset[str]] = frozenset(
    {"documentation", "docs", "readme"}
)
_TEST_TERMS: Final[frozenset[str]] = frozenset(
    {"test", "tests", "testing", "pytest", "spec", "specs"}
)


def infer_topic(reader: SourceReader, anchor: str) -> tuple[str | None, set[str]]:
    """Infer a codebase-backed topic phrase and its matching paths."""
    words: list[str] = [word.lower() for word in _WORD.findall(anchor)]
    matches: list[tuple[int, int, str, list[str]]] = []
    for width in (3, 2):
        for index in range(len(words) - width + 1):
            phrase: list[str] = words[index : index + width]
            paths: set[str] = set()
            for separator in ("_", "-"):
                paths.update(reader.find_paths(separator.join(phrase), limit=100))
            if len(paths) >= 3:
                matches.append((len(paths), width, " ".join(phrase), sorted(paths)))
    if not matches:
        return (None, set())
    _, _, phrase, paths = max(matches, key=lambda item: (item[0], item[1]))
    return (phrase, set(paths))


def _has_intent(facets: set[str], terms: frozenset[str]) -> bool:
    words: set[str] = {
        word.lower() for facet in facets for word in _WORD.findall(facet)
    }
    return bool(words & terms)


def apply_metadata_priors(
    candidates: list[RankedCandidate],
    *,
    topic_paths: set[str],
    options: HybridSearchOptions,
) -> list[RankedCandidate]:
    """Apply modest source-category and topic-path adjustments after RRF.

    Takes the topic paths rather than a reader: inferring them costs a query
    per candidate phrase, and the caller has already paid for them to build
    its own queries.
    """
    for candidate in candidates:
        path: str = candidate.result.path.lower()
        documentation: bool = (
            path.startswith(("docs/", "documentation/"))
            or "/readme." in path
            or path.endswith((".md", ".mdx", ".rst"))
        )
        test_path: bool = path.startswith("tests/") or "/tests/" in path
        adjusted: dict[str, float] = {}
        for facet, score in candidate.facet_scores.items():
            multiplier: float = (
                options.topic_path_boost
                if candidate.result.path in topic_paths
                else 1.0
            )
            facet_set: set[str] = {facet}
            if documentation and not _has_intent(
                facet_set,
                _DOCUMENTATION_TERMS,
            ):
                multiplier *= options.documentation_penalty
            if "migration" in path and not _has_intent(
                facet_set,
                frozenset({"migration", "schema"}),
            ):
                multiplier *= options.migration_penalty
            if test_path and not _has_intent(facet_set, _TEST_TERMS):
                multiplier *= options.unrelated_test_penalty
            adjusted[facet] = score * multiplier
        candidate.facet_scores = adjusted
        candidate.score = replace(
            candidate.score,
            fused=sum(adjusted.values()),
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate.score.fused,
            candidate.result.path,
            candidate.result.start_line,
        )
    )
    return candidates
