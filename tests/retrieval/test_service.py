"""Tests for the hybrid search application service."""

from __future__ import annotations

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    HybridSearchOptions,
    IndexedChunk,
    OutlineEntry,
    RetrievalChannel,
    SourceLine,
)
from smritikosh.retrieval.service import HybridSearchService


class _Retriever:
    def __init__(
        self,
        channel: RetrievalChannel,
        results: dict[str, list[SearchResult]],
    ) -> None:
        self.channel = channel
        self._results = results

    def retrieve(self, query: str, *, options: object) -> list[SearchResult]:
        return self._results.get(query, [])


class _Reader:
    def find_paths(self, pattern: str, *, limit: int = 50) -> list[str]:
        return []

    def find_text_lines(
        self,
        text: str,
        *,
        path: str | None = None,
        limit: int = 20,
    ) -> list[object]:
        return []

    def get_chunks(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[IndexedChunk]:
        return []

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, IndexedChunk]:
        return {}

    def get_outline(self, path: str) -> list[OutlineEntry]:
        return [OutlineEntry("load", "function", 1, 2)]

    def get_source_lines(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[SourceLine]:
        return [SourceLine(1, f"def load_{path.replace('/', '_')}():")]


def test_should_locate_code_for_every_requested_facet() -> None:
    flow = SearchResult(
        "src/flow.py",
        1,
        2,
        "def load(): pass",
        0.9,
        "function",
        "load",
    )
    retry = SearchResult(
        "src/retry.py",
        1,
        2,
        "def load(): pass",
        0.8,
        "function",
        "load",
    )
    service = HybridSearchService(
        (
            _Retriever(RetrievalChannel.DENSE, {"flow": [flow], "retry": [retry]}),
            _Retriever(RetrievalChannel.LEXICAL, {"flow": [flow], "retry": [retry]}),
        ),
        _Reader(),
    )

    locations = service.search(
        ("flow", "retry"),
        options=HybridSearchOptions(max_seeds=2, max_results=2),
    )

    assert [location.path for location in locations] == [
        "src/flow.py",
        "src/retry.py",
    ]
    assert [location.facets for location in locations] == [("flow",), ("retry",)]


def test_should_name_the_facet_of_a_result_that_missed_the_credit_cutoff() -> None:
    """A location retrieved for a facet must not report matching none of them.

    Credit stops at a channel's leading ranks, so the one real hit for a facet
    can arrive behind enough near-misses to carry no credit while still holding
    that facet's score.
    """
    near_misses = [
        SearchResult(
            f"src/near_{index}.py",
            1,
            2,
            "def load(): pass",
            0.9,
            "function",
            "load",
        )
        for index in range(10)
    ]
    real = SearchResult(
        "src/grace_rules.py",
        1,
        2,
        "def load(): pass",
        0.4,
        "function",
        "load",
    )
    service = HybridSearchService(
        (
            _Retriever(
                RetrievalChannel.DENSE,
                {"grace period": [*near_misses, real]},
            ),
        ),
        _Reader(),
    )

    locations = service.search(
        ("grace period",),
        options=HybridSearchOptions(max_seeds=24, max_results=24),
    )

    reported = {
        location.path: location.facets
        for location in locations
        if location.path == "src/grace_rules.py"
    }
    assert reported == {"src/grace_rules.py": ("grace period",)}


def test_should_return_locations_without_reading_source() -> None:
    reader = _Reader()
    result = SearchResult(
        "src/flow.py",
        1,
        2,
        "def load(): pass",
        0.9,
        "function",
        "load",
    )
    service = HybridSearchService(
        (_Retriever(RetrievalChannel.DENSE, {"flow": [result]}),),
        reader,
    )

    locations = service.search(("flow",), options=HybridSearchOptions(max_results=1))

    assert [(location.start_line, location.end_line) for location in locations] == [
        (1, 2)
    ]
    assert not hasattr(locations[0], "source")
