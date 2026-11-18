"""Tests for the read-only Smritikosh exploration API."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from smritikosh.exploration import ReadOnlyExplorer, SearchOptions


class _StubEmbedder:
    """Return deterministic vectors without loading an embedding model."""

    dims = 3

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        vectors: dict[str, list[float]] = {
            "implementation": [1.0, 0.0, 0.0],
            "tests": [0.0, 1.0, 0.0],
            "ranking": [0.65, 0.76, 0.0],
            "test ranking": [0.65, 0.76, 0.0],
            "reference material ranking": [0.65, 0.0, 0.76],
            "documentation ranking": [0.65, 0.0, 0.76],
        }
        return [vectors[text] for text in texts]


def _make_db(
    tmp_path: Path,
    chunks: list[tuple[str, str, int, int, str, str | None]],
    vectors: dict[str, list[float]] | None = None,
    metadata: dict[str, dict[str, object]] | None = None,
) -> str:
    """Write an index of ``(id, path, start, end, text, symbol)`` chunks.

    Pass *vectors* to make the index searchable; without them only the
    non-semantic reads have anything to work with. *metadata* adds stored
    fields per chunk id, such as ``source_text``.
    """
    db_path: Path = tmp_path / "chunks.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT,
            name TEXT,
            path TEXT,
            metadata JSON
        );
        CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, vector FLOAT[3]);
        CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO kv_store VALUES ('embedder_dims', '3');
        """
    )
    connection.executemany(
        "INSERT INTO nodes VALUES (?, 'chunk', ?, ?, ?)",
        [
            (
                chunk_id,
                symbol,
                path,
                json.dumps(
                    {
                        "start_line": start,
                        "end_line": end,
                        "text": text,
                        "chunk_kind": "function",
                        **(metadata or {}).get(chunk_id, {}),
                    }
                ),
            )
            for chunk_id, path, start, end, text, symbol in chunks
        ],
    )
    if vectors:
        connection.executemany(
            "INSERT INTO vectors VALUES (?, ?)", list(vectors.items())
        )
    connection.close()
    return str(db_path)


@pytest.fixture()
def indexed_db(tmp_path: Path) -> str:
    """Create a minimal on-disk index that can be reopened read-only."""
    db_path: Path = tmp_path / "index.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT,
            path TEXT,
            metadata JSON
        );
        CREATE TABLE vectors (
            chunk_id TEXT PRIMARY KEY,
            vector FLOAT[3]
        );
        CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    rows: list[tuple[str, str, str | None, str, str]] = [
        ("file:src/billing.py", "file", None, "src/billing.py", "{}"),
        ("file:docs/billing.md", "file", None, "docs/billing.md", "{}"),
        (
            "chunk:implementation",
            "chunk",
            "eligible_account",
            "src/billing.py",
            json.dumps(
                {
                    "start_line": 10,
                    "end_line": 20,
                    "text": "def eligible_account():\n    return True",
                    "chunk_kind": "function",
                }
            ),
        ),
        (
            "chunk:test",
            "chunk",
            "test_eligible_account",
            "tests/test_billing.py",
            json.dumps(
                {
                    "start_line": 30,
                    "end_line": 40,
                    "text": "def test_eligible_account():\n    assert True",
                    "chunk_kind": "function",
                }
            ),
        ),
        (
            "chunk:documentation",
            "chunk",
            "Billing eligibility",
            "docs/billing.md",
            json.dumps(
                {
                    "start_line": 1,
                    "end_line": 5,
                    "text": "# Billing eligibility",
                    "chunk_kind": "section",
                }
            ),
        ),
    ]
    connection.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?)", rows)
    connection.executemany(
        "INSERT INTO vectors VALUES (?, ?)",
        [
            ("chunk:implementation", [1.0, 0.0, 0.0]),
            ("chunk:test", [0.0, 1.0, 0.0]),
            ("chunk:documentation", [0.0, 0.0, 1.0]),
        ],
    )
    connection.execute("INSERT INTO kv_store VALUES ('embedder_dims', '3')")
    connection.close()
    return str(db_path)


def test_should_report_index_counts_when_index_exists(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    info = explorer.index_info()
    explorer.close()

    # Assert
    assert (info.files, info.chunks, info.vectors, info.dimensions) == (2, 3, 3, 3)


def test_should_merge_ad_hoc_queries_by_best_score(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    results = explorer.semantic_search(
        ["implementation", "tests"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=2),
    )
    explorer.close()

    # Assert
    assert {result.path for result in results} == {
        "src/billing.py",
        "tests/test_billing.py",
    }
    assert all(result.score == pytest.approx(1.0) for result in results)


def test_should_exclude_paths_when_semantically_searching(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    results = explorer.semantic_search(
        ["tests"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=1, exclude_paths=("tests/%",)),
    )
    explorer.close()

    # Assert
    assert [result.path for result in results] == ["src/billing.py"]


def test_should_demote_test_paths_by_default(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    results = explorer.semantic_search(
        ["ranking"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=1),
    )
    explorer.close()

    # Assert
    assert [result.path for result in results] == ["src/billing.py"]


def test_should_demote_documentation_paths_by_default(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    results = explorer.semantic_search(
        ["reference material ranking"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=1),
    )
    explorer.close()

    # Assert
    assert [result.path for result in results] == ["src/billing.py"]


def test_should_not_demote_tests_when_requested_by_query(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    results = explorer.semantic_search(
        ["test ranking"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=1),
    )
    explorer.close()

    # Assert
    assert [result.path for result in results] == ["tests/test_billing.py"]


def test_should_not_demote_documentation_when_requested_by_query(
    indexed_db: str,
) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    results = explorer.semantic_search(
        ["documentation ranking"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=1),
    )
    explorer.close()

    # Assert
    assert [result.path for result in results] == ["docs/billing.md"]


def test_should_drop_a_hit_that_mostly_repeats_a_better_one(
    tmp_path: Path,
) -> None:
    # Arrange: two windows of one definition, overlapping on lines 8-12.
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:best", "src/a.py", 1, 12, "first window", "load"),
            ("chunk:window", "src/a.py", 8, 14, "second window", "load"),
            ("chunk:other", "src/a.py", 40, 50, "elsewhere", "save"),
        ],
        vectors={
            "chunk:best": [1.0, 0.0, 0.0],
            "chunk:window": [0.99, 0.14, 0.0],
            "chunk:other": [0.98, 0.2, 0.0],
        },
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    results = explorer.semantic_search(
        ["implementation"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=3),
    )
    explorer.close()

    # Assert
    assert [(r.start_line, r.end_line) for r in results] == [(1, 12), (40, 50)]


def test_should_keep_a_hit_that_only_grazes_a_better_one(tmp_path: Path) -> None:
    # Arrange: 2 of the candidate's 11 lines are already covered.
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:best", "src/a.py", 1, 12, "first", "load"),
            ("chunk:next", "src/a.py", 11, 21, "second", "save"),
        ],
        vectors={
            "chunk:best": [1.0, 0.0, 0.0],
            "chunk:next": [0.99, 0.14, 0.0],
        },
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    results = explorer.semantic_search(
        ["implementation"],
        embedder=_StubEmbedder(),
        options=SearchOptions(top_k=3),
    )
    explorer.close()

    # Assert
    assert [(r.start_line, r.end_line) for r in results] == [(1, 12), (11, 21)]


def test_should_find_paths_by_case_insensitive_substring(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    paths = explorer.find_paths("BILLING")
    explorer.close()

    # Assert
    assert paths == [
        "docs/billing.md",
        "src/billing.py",
        "tests/test_billing.py",
    ]


def test_should_find_exact_text_within_one_path(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    chunks = explorer.text_search("eligible_account", path="src/billing.py")
    explorer.close()

    # Assert
    assert [chunk.chunk_id for chunk in chunks] == ["chunk:implementation"]


def test_should_return_chunks_overlapping_requested_lines(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act
    chunks = explorer.get_chunks("src/billing.py", start_line=15, end_line=16)
    explorer.close()

    # Assert
    assert [chunk.chunk_id for chunk in chunks] == ["chunk:implementation"]


def test_should_return_each_matching_line_once_with_its_line_number(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:first", "src/a.py", 1, 3, "import os\nTOTAL = 1\nother", None),
            ("chunk:second", "src/a.py", 2, 4, "TOTAL = 1\nother\nTOTAL = 2", None),
        ],
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    matches = explorer.find_text_lines("total")
    explorer.close()

    # Assert
    assert [(match.line_number, match.text) for match in matches] == [
        (2, "TOTAL = 1"),
        (4, "TOTAL = 2"),
    ]
    assert all(match.exact_line for match in matches)


def test_should_anchor_unmappable_matches_to_their_chunk_start(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [("chunk:grouped", "src/a.py", 5, 40, "RATE = 1\nLIMIT = 2", None)],
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    matches = explorer.find_text_lines("LIMIT")
    explorer.close()

    # Assert
    assert [(match.line_number, match.exact_line) for match in matches] == [(5, False)]


def test_should_collapse_the_windows_of_one_definition_into_one_outline_entry(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:a", "src/a.py", 1, 20, "head", "load"),
            ("chunk:b", "src/a.py", 19, 40, "tail", "load"),
            ("chunk:c", "src/a.py", 42, 50, "next", "save"),
        ],
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    entries = explorer.get_outline("src/a.py")
    explorer.close()

    # Assert
    assert [(e.symbol, e.start_line, e.end_line) for e in entries] == [
        ("load", 1, 40),
        ("save", 42, 50),
    ]


def test_should_keep_unnamed_chunks_apart_in_the_outline(tmp_path: Path) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:a", "src/a.py", 1, 5, "one", None),
            ("chunk:b", "src/a.py", 6, 9, "two", None),
        ],
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    entries = explorer.get_outline("src/a.py")
    explorer.close()

    # Assert
    assert [(e.start_line, e.end_line) for e in entries] == [(1, 5), (6, 9)]


def test_should_rebuild_requested_lines_once_from_overlapping_chunks(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:first", "src/a.py", 1, 3, "one\ntwo\nthree", None),
            ("chunk:second", "src/a.py", 3, 5, "three\nfour\nfive", None),
        ],
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    lines = explorer.get_source_lines("src/a.py", start_line=2, end_line=4)
    explorer.close()

    # Assert
    assert [(line.line_number, line.text) for line in lines] == [
        (2, "two"),
        (3, "three"),
        (4, "four"),
    ]


def test_should_skip_chunks_whose_text_does_not_span_their_lines(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:grouped", "src/a.py", 1, 9, "FIRST = 1\nLAST = 2", None),
            ("chunk:exact", "src/a.py", 4, 4, "middle", None),
        ],
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    lines = explorer.get_source_lines("src/a.py")
    explorer.close()

    # Assert
    assert [(line.line_number, line.text) for line in lines] == [(4, "middle")]


def test_should_rebuild_lines_from_source_text_when_chunk_text_differs(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [("chunk:doc", "docs/a.md", 3, 4, "# Guide\n\nRun the tests.\n", None)],
        metadata={"chunk:doc": {"source_text": "Run the tests.\n"}},
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    lines = explorer.get_source_lines("docs/a.md", start_line=3, end_line=4)
    explorer.close()

    # Assert
    assert [(line.line_number, line.text) for line in lines] == [
        (3, "Run the tests."),
        (4, ""),
    ]


def test_should_report_exact_lines_for_matches_in_source_text(tmp_path: Path) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [("chunk:doc", "docs/a.md", 3, 4, "# Guide\n\nRun the tests.\n", None)],
        metadata={"chunk:doc": {"source_text": "Run the tests.\n"}},
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    matches = explorer.find_text_lines("tests")
    explorer.close()

    # Assert
    assert [(m.line_number, m.text, m.exact_line) for m in matches] == [
        (3, "Run the tests.", True)
    ]


def test_should_not_spend_the_limit_on_chunks_matching_only_in_a_breadcrumb(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [
            ("chunk:a", "docs/a.md", 3, 3, "# Setup\n## Mac\n\nUse brew.", None),
            ("chunk:b", "docs/a.md", 7, 7, "# Setup\n## Linux\n\nUse apt.", None),
            ("chunk:c", "src/z.py", 1, 1, "def setup(): pass", None),
        ],
        metadata={
            "chunk:a": {"source_text": "Use brew."},
            "chunk:b": {"source_text": "Use apt."},
        },
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    matches = explorer.find_text_lines("setup", limit=2)
    explorer.close()

    # Assert
    assert [(m.path, m.line_number) for m in matches] == [("src/z.py", 1)]


def test_should_find_a_source_line_the_retrieval_text_leaves_out(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: str = _make_db(
        tmp_path,
        [("chunk:doc", "docs/a.md", 1, 4, "# Guide\n\nRun the tests.", None)],
        metadata={"chunk:doc": {"source_text": "Guide\n=====\n\nRun the tests."}},
    )
    explorer = ReadOnlyExplorer(db_path)

    # Act
    matches = explorer.find_text_lines("=====")
    explorer.close()

    # Assert
    assert [(m.line_number, m.text, m.exact_line) for m in matches] == [
        (2, "=====", True)
    ]


def test_should_reject_reversed_line_range(indexed_db: str) -> None:
    # Arrange
    explorer = ReadOnlyExplorer(indexed_db)

    # Act / Assert
    with pytest.raises(ValueError, match="cannot be greater"):
        explorer.get_chunks("src/billing.py", start_line=20, end_line=10)
    explorer.close()
