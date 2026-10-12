"""Models shared by retrieval application services and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from smritikosh.models.types import SearchResult

__all__ = [
    "CandidateScore",
    "HybridSearchOptions",
    "IndexInfo",
    "IndexedChunk",
    "OutlineEntry",
    "RankedCandidate",
    "RetrievalChannel",
    "SearchLocation",
    "SearchOptions",
    "SourceLine",
    "TextMatch",
]


class RetrievalChannel(str, Enum):
    """Identify one independent candidate-generation channel.

    ``enum.StrEnum`` would say this more directly but arrived in 3.11, and
    pyproject still supports 3.10.
    """

    DENSE = "dense"
    LEXICAL = "lexical"


@dataclass(frozen=True)
class IndexInfo:
    """Summarize the contents of a Smritikosh index."""

    files: int
    chunks: int
    vectors: int
    dimensions: int | None


@dataclass(frozen=True)
class IndexedChunk:
    """Represent one source chunk stored in the index."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    chunk_kind: str | None
    #: Definition this chunk came from; None for chunks that cover no single
    #: definition, and for chunks written before symbols were recorded.
    symbol: str | None = None


@dataclass(frozen=True)
class SourceLine:
    """Represent one numbered source line."""

    line_number: int
    text: str


@dataclass(frozen=True)
class OutlineEntry:
    """Summarize one definition stored for a path."""

    symbol: str | None
    chunk_kind: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class TextMatch:
    """Locate one source line containing exact text."""

    path: str
    line_number: int
    text: str
    exact_line: bool


@dataclass(frozen=True)
class SearchOptions:
    """Configure one candidate retrieval operation."""

    top_k: int = 10
    include_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    """Retain channel ranks and fused relevance for diagnostics."""

    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused: float = 0.0


@dataclass
class RankedCandidate:
    """Represent a deduplicated candidate that may cover several facets."""

    result: SearchResult
    facets: set[str] = field(default_factory=set)
    facet_scores: dict[str, float] = field(default_factory=dict)
    score: CandidateScore = field(default_factory=CandidateScore)


@dataclass(frozen=True)
class HybridSearchOptions:
    """Bound hybrid candidate generation and result selection."""

    candidates_per_channel: int = 30
    max_candidates: int = 400
    #: How many candidates coverage-and-diversity selection keeps.  It matches
    #: max_results because selection now produces the reported list outright;
    #: the two differed while a dependency round appended to what it chose.
    max_seeds: int = 24
    max_results: int = 24
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    max_per_file: int = 2
    topic_path_boost: float = 1.25
    documentation_penalty: float = 0.75
    migration_penalty: float = 0.85
    unrelated_test_penalty: float = 0.9
    exclude_paths: tuple[str, ...] = (
        ".claude/%",
        ".cursor/%",
        ".git/%",
        ".windsurf/%",
        ".venv/%",
        "node_modules/%",
        "vendor/%",
    )


@dataclass(frozen=True)
class SearchLocation:
    """Locate one selected definition and the facets that matched it.

    A location carries no source: ``explore chunks --range`` reads the exact
    lines when the caller decides which of them are worth reading.
    """

    path: str
    start_line: int
    end_line: int
    symbol: str | None
    facets: tuple[str, ...]
