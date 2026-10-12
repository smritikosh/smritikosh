"""Tests for widening selected chunks to whole definitions."""

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    CandidateScore,
    HybridSearchOptions,
    OutlineEntry,
    RankedCandidate,
)
from smritikosh.retrieval.definitions import complete_definitions


class _Reader:
    """Serve one outline per path and record how often it is asked."""

    def __init__(self, outlines: dict[str, list[OutlineEntry]]) -> None:
        self._outlines = outlines
        self.calls: list[str] = []

    def get_outline(self, path: str) -> list[OutlineEntry]:
        self.calls.append(path)
        return self._outlines.get(path, [])


def _candidate(
    path: str,
    start_line: int,
    end_line: int,
    *,
    symbol: str | None = "load",
    chunk_kind: str | None = "function",
) -> RankedCandidate:
    return RankedCandidate(
        result=SearchResult(
            path,
            start_line,
            end_line,
            "code",
            0.5,
            chunk_kind,
            symbol,
        ),
        facets={"order export"},
        facet_scores={"order export": 0.5},
        score=CandidateScore(fused=0.5),
    )


def _spans(candidates: list[RankedCandidate]) -> list[tuple[int, int]]:
    return [(c.result.start_line, c.result.end_line) for c in candidates]


def test_should_widen_a_chunk_window_to_the_whole_definition() -> None:
    """A definition too large for one chunk is stored as several windows.

    A hit on the middle window must report the definition, not the window.
    """
    reader = _Reader({"src/flow.py": [OutlineEntry("load", "function", 1, 300)]})

    located = complete_definitions(
        [_candidate("src/flow.py", 101, 200)],
        reader=reader,
        options=HybridSearchOptions(),
    )

    assert _spans(located) == [(1, 300)]


def test_should_collapse_two_windows_of_one_definition_into_one_location() -> None:
    """Widening is what makes these duplicates; nothing upstream can see it."""
    reader = _Reader({"src/flow.py": [OutlineEntry("load", "function", 1, 300)]})

    located = complete_definitions(
        [
            _candidate("src/flow.py", 101, 200),
            _candidate("src/flow.py", 201, 300),
        ],
        reader=reader,
        options=HybridSearchOptions(),
    )

    assert _spans(located) == [(1, 300)]
    assert reader.calls == ["src/flow.py"], "outline should be read once per path"


def test_should_leave_a_chunk_that_covers_no_single_definition_alone() -> None:
    reader = _Reader({"docs/guide.md": [OutlineEntry(None, "section", 1, 400)]})

    located = complete_definitions(
        [
            _candidate("docs/guide.md", 5, 20, symbol=None, chunk_kind="section"),
            _candidate("src/flow.py", 5, 20, chunk_kind="grouped"),
        ],
        reader=reader,
        options=HybridSearchOptions(),
    )

    assert _spans(located) == [(5, 20), (5, 20)]
    assert reader.calls == [], "a chunk naming no definition needs no outline"


def test_should_keep_distinct_chunks_from_the_same_markdown_section() -> None:
    """Every chunk of a split section carries the same heading as its symbol."""
    reader = _Reader({})

    located = complete_definitions(
        [
            _candidate(
                "Transaction_System.md",
                20,
                42,
                symbol="Transaction Lifecycle Management",
                chunk_kind="section",
            ),
            _candidate(
                "Transaction_System.md",
                43,
                46,
                symbol="Transaction Lifecycle Management",
                chunk_kind="section",
            ),
        ],
        reader=reader,
        options=HybridSearchOptions(),
    )

    assert _spans(located) == [(20, 42), (43, 46)]


def test_should_keep_the_original_span_when_the_outline_has_no_match() -> None:
    """A stale or rebuilt index can hold a chunk the outline no longer names."""
    reader = _Reader({"src/flow.py": [OutlineEntry("save", "function", 1, 300)]})

    located = complete_definitions(
        [_candidate("src/flow.py", 101, 200)],
        reader=reader,
        options=HybridSearchOptions(),
    )

    assert _spans(located) == [(101, 200)]


def test_should_cap_the_reported_locations() -> None:
    reader = _Reader({})

    located = complete_definitions(
        [
            _candidate("src/a.py", 1, 2, symbol="a"),
            _candidate("src/b.py", 1, 2, symbol="b"),
            _candidate("src/c.py", 1, 2, symbol="c"),
        ],
        reader=reader,
        options=HybridSearchOptions(max_results=2),
    )

    assert [candidate.result.path for candidate in located] == [
        "src/a.py",
        "src/b.py",
    ]


def test_should_preserve_the_facets_and_score_of_a_widened_candidate() -> None:
    reader = _Reader({"src/flow.py": [OutlineEntry("load", "function", 1, 300)]})
    seed = _candidate("src/flow.py", 101, 200)

    located = complete_definitions(
        [seed],
        reader=reader,
        options=HybridSearchOptions(),
    )

    assert located[0].facets == seed.facets
    assert located[0].facet_scores == seed.facet_scores
    assert located[0].score.fused == seed.score.fused
