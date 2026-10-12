"""Tests for rank fusion and diverse candidate selection."""

from __future__ import annotations

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    HybridSearchOptions,
    RetrievalChannel,
)
from smritikosh.retrieval.fusion import fuse_ranked_results
from smritikosh.retrieval.selection import select_seeds


def _result(path: str, symbol: str, snippet: str = "") -> SearchResult:
    return SearchResult(path, 1, 3, snippet, 0.5, "function", symbol)


def test_should_fuse_dense_and_lexical_ranks_without_mixing_raw_scores() -> None:
    shared = _result("src/shared.py", "shared")
    dense_only = _result("src/dense.py", "dense")
    lexical_only = _result("src/lexical.py", "lexical")

    candidates = fuse_ranked_results(
        {
            "facet": {
                RetrievalChannel.DENSE: [shared, dense_only],
                RetrievalChannel.LEXICAL: [shared, lexical_only],
            }
        }
    )

    assert candidates[0].result.symbol == "shared"
    assert candidates[0].score.dense_rank == 1
    assert candidates[0].score.lexical_rank == 1


def test_should_reserve_each_facet_before_filling_for_relevance() -> None:
    first = _result("src/first.py", "first", "alpha")
    second = _result("src/second.py", "second", "beta")
    candidates = fuse_ranked_results(
        {
            "alpha": {
                RetrievalChannel.DENSE: [first],
                RetrievalChannel.LEXICAL: [],
            },
            "beta": {
                RetrievalChannel.DENSE: [second],
                RetrievalChannel.LEXICAL: [],
            },
        }
    )

    selected = select_seeds(
        candidates,
        ("alpha", "beta"),
        options=HybridSearchOptions(max_seeds=2),
    )

    assert {candidate.result.symbol for candidate in selected} == {
        "first",
        "second",
    }


def test_should_count_split_symbol_once_per_channel_and_facet() -> None:
    first_window = SearchResult(
        "src/a.py",
        1,
        10,
        "first",
        0.9,
        "method",
        "publish",
    )
    second_window = SearchResult(
        "src/a.py",
        9,
        20,
        "second",
        0.8,
        "method",
        "publish",
    )

    candidates = fuse_ranked_results(
        {
            "facet": {
                RetrievalChannel.DENSE: [first_window, second_window],
                RetrievalChannel.LEXICAL: [],
            }
        }
    )

    assert len(candidates) == 1
    assert candidates[0].score.fused == 1 / 61


def test_should_keep_same_named_non_overlapping_methods_distinct() -> None:
    first = SearchResult(
        "src/a.py",
        1,
        10,
        "class First",
        0.9,
        "method",
        "save",
    )
    second = SearchResult(
        "src/a.py",
        30,
        40,
        "class Second",
        0.8,
        "method",
        "save",
    )

    candidates = fuse_ranked_results(
        {
            "facet": {
                RetrievalChannel.DENSE: [first, second],
                RetrievalChannel.LEXICAL: [],
            }
        }
    )

    assert [(item.result.start_line, item.result.end_line) for item in candidates] == [
        (1, 10),
        (30, 40),
    ]
