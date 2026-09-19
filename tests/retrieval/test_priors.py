"""Tests for generic post-fusion metadata priors."""

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    CandidateScore,
    HybridSearchOptions,
    RankedCandidate,
)
from smritikosh.retrieval.priors import apply_metadata_priors, infer_topic


class _Reader:
    def find_paths(self, pattern: str, *, limit: int = 50) -> list[str]:
        if pattern == "order_export":
            return [
                "src/order_export.py",
                "src/order_export_handler.py",
                "tests/test_order_export.py",
            ]
        return []


def _candidate(path: str) -> RankedCandidate:
    return RankedCandidate(
        result=SearchResult(path, 1, 2, "code", 0.5, "function", "run"),
        facets={"order export retry"},
        facet_scores={"order export retry": 0.1},
        score=CandidateScore(fused=0.1),
    )


def test_should_infer_the_topic_phrase_the_codebase_backs() -> None:
    phrase, paths = infer_topic(_Reader(), "order export flow")

    assert phrase == "order export"
    assert "src/order_export.py" in paths


def test_should_prefer_topic_source_and_demote_unrequested_documentation() -> None:
    documentation = _candidate("docs/order_export.md")
    sibling = _candidate("src/payment_export.py")
    topic = _candidate("src/order_export.py")
    _, topic_paths = infer_topic(_Reader(), "order export flow")

    ranked = apply_metadata_priors(
        [documentation, sibling, topic],
        topic_paths=topic_paths,
        options=HybridSearchOptions(),
    )

    assert [candidate.result.path for candidate in ranked] == [
        "src/order_export.py",
        "src/payment_export.py",
        "docs/order_export.md",
    ]
