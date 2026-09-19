"""Tests for agent-facing exploration CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import toons
from click.testing import CliRunner

from smritikosh.cli import main
from smritikosh.exploration import (
    IndexedChunk,
    OutlineEntry,
    SourceLine,
)
from smritikosh.exploration_cli import explore
from smritikosh.models import SearchLocation, SearchResult

LEGACY_COMMANDS = ("evidence", "text", "paths", "info", "batch", "discover")


def _mock_explorer() -> MagicMock:
    explorer = MagicMock()
    explorer.__enter__.return_value = explorer
    return explorer


def _run_search(db_path: Path, *extra: str) -> str:
    result = CliRunner().invoke(
        main,
        ["explore", "search", *extra, "--db-path", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_should_expose_tools_search_and_chunks() -> None:
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(main, ["explore", "--help"])

    # Assert
    assert result.exit_code == 0
    assert sorted(explore.commands) == ["chunks", "search", "tools"]
    assert "tools" in result.output
    assert "search" in result.output
    assert "chunks" in result.output


def test_should_list_agent_facing_tool_usage_as_toon() -> None:
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(main, ["explore", "tools", "--toon"])

    # Assert
    assert result.exit_code == 0, result.output
    payload = toons.loads(result.output)
    assert [step["action"] for step in payload["workflow"]] == [
        "search",
        "chunks",
        "answer",
    ]
    assert [tool["name"] for tool in payload["tools"]] == [
        "search",
        "chunks",
        "chunks",
    ]
    assert (
        payload["tools"][0]["usage"] == "smritikosh explore search QUERY... "
        "--max-results 12 [--db-path PATH]"
    )
    assert (
        payload["tools"][1]["usage"]
        == "smritikosh explore chunks --range PATH START END "
        "[--range PATH START END ...] [--db-path PATH]"
    )
    assert payload["search_standards"] == [
        {
            "parameter": "QUERY count",
            "start_with": "4",
            "guidance": "Use one distinct facet per query; never repeat synonyms.",
        },
        {
            "parameter": "QUERY facets",
            "start_with": "flow|state|failure|tests",
            "guidance": "Include exact domain nouns, events, or symbols when known.",
        },
        {
            "parameter": "--max-results",
            "start_with": "12",
            "guidance": (
                "Raise to 18, then 24 only when evidence categories are missing."
            ),
        },
        {
            "parameter": "--db-path",
            "start_with": "the provided index path",
            "guidance": "Set it explicitly when more than one index may exist.",
        },
    ]
    assert payload["agent_guidance"] == [
        "Follow search -> chunks -> answer.",
        "Treat search results as candidates; verify material claims with chunks.",
        "Batch independent reads with repeated --range and avoid overlaps.",
        "Read relevant implementation and test bodies.",
        (
            "Target at most 12 Smritikosh CLI calls after tools; exceed only "
            "to close material evidence gaps."
        ),
        (
            "Trace relevant ownership, flow, conditions, and failure paths; "
            "separate verified findings, assumptions, and unknowns."
        ),
        (
            "Cite every material claim with a chunks-returned "
            "portfolio-relative PATH:START-END."
        ),
        (
            "Answer concisely with findings, flow, failure conditions, tests, "
            "and unverified items."
        ),
    ]
    assert "do not also pass positional PATH" in payload["rules"][0]["rule"]


def test_should_include_agent_guidance_in_tools_prose() -> None:
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(main, ["explore", "tools", "--prose"])

    # Assert
    assert result.exit_code == 0, result.output
    assert "Agent guidance:" in result.output
    assert "Treat search results as candidates" in result.output
    assert "Read relevant implementation and test bodies." in result.output
    assert "portfolio-relative PATH:START-END" in result.output


def test_should_reject_every_removed_exploration_command(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    runner = CliRunner()

    # Act
    results = {
        command: runner.invoke(
            main,
            ["explore", command, "--db-path", str(db_path)],
        )
        for command in LEGACY_COMMANDS
    }

    # Assert
    assert all(result.exit_code != 0 for result in results.values())
    assert all("No such command" in result.output for result in results.values())


def test_should_label_facets_by_query_id_in_two_compact_tables(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = [
        SearchLocation("src/s3.py", 14, 21, "upload", ("upload flow", "upload tests")),
        SearchLocation("src/errors.py", 46, 53, "PathError", ("path validation",)),
    ]

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        output = _run_search(
            db_path,
            "upload flow",
            "path validation",
            "upload tests",
        )

    # Assert
    payload = toons.loads(output)
    assert list(payload) == ["queries", "search_results"]
    assert payload["queries"] == [
        {"id": "Q1", "query": "upload flow"},
        {"id": "Q2", "query": "path validation"},
        {"id": "Q3", "query": "upload tests"},
    ]
    assert payload["search_results"] == [
        {
            "path": "src/s3.py",
            "start_line": 14,
            "end_line": 21,
            "symbol": "upload",
            "facets": "Q1|Q3",
        },
        {
            "path": "src/errors.py",
            "start_line": 46,
            "end_line": 53,
            "symbol": "PathError",
            "facets": "Q2",
        },
    ]


def test_should_omit_source_coverage_and_truncation_metadata(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = [
        SearchLocation("src/s3.py", 14, 21, None, ()),
    ]

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        output = _run_search(db_path, "upload flow")

    # Assert
    payload = toons.loads(output)
    assert list(payload) == ["queries", "search_results"]
    row = payload["search_results"][0]
    assert list(row) == ["path", "start_line", "end_line", "symbol", "facets"]
    assert (row["symbol"], row["facets"]) == ("", "")


def test_should_return_ranges_that_chunks_can_read_directly(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = [
        SearchLocation("src/helper.py", 10, 60, "record_and_publish", ("retry",)),
    ]
    explorer = _mock_explorer()
    explorer.get_source_lines.return_value = [
        SourceLine(10, "def record_and_publish():"),
        SourceLine(60, "    raise"),
    ]

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        search_output = _run_search(db_path, "retry")
    row = toons.loads(search_output)["search_results"][0]
    with patch(
        "smritikosh.exploration_cli.ReadOnlyExplorer",
        return_value=explorer,
    ):
        chunks_result = CliRunner().invoke(
            main,
            [
                "explore",
                "chunks",
                "--range",
                str(row["path"]),
                str(row["start_line"]),
                str(row["end_line"]),
                "--db-path",
                str(db_path),
            ],
        )

    # Assert
    assert chunks_result.exit_code == 0
    explorer.get_source_lines.assert_called_once_with(
        "src/helper.py",
        start_line=10,
        end_line=60,
    )
    assert chunks_result.output == (
        "src/helper.py:10-60\n10  def record_and_publish():\n60      raise\n"
    )


def test_should_clamp_search_limits_instead_of_failing(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = []

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        _run_search(db_path, "query", "--max-results", "48")

    # Assert
    assert service.search.call_args.kwargs["options"].max_results == 24


def test_should_bound_the_number_and_length_of_accepted_queries(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = []

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        _run_search(db_path, *[f"query {index}" for index in range(10)], "x" * 900)

    # Assert
    accepted = service.search.call_args.args[0]
    assert len(accepted) == 8
    assert all(len(query) <= 500 for query in accepted)


def test_should_collapse_repeated_queries_before_retrieving(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = []

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        output = _run_search(db_path, "upload flow", "upload flow", " upload flow ")

    # Assert
    assert service.search.call_args.args[0] == ("upload flow",)
    assert toons.loads(output)["queries"] == [{"id": "Q1", "query": "upload flow"}]


def test_should_reject_a_search_without_any_non_empty_query(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()

    # Act
    result = CliRunner().invoke(
        main,
        ["explore", "search", "   ", "--db-path", str(db_path)],
    )

    # Assert
    assert result.exit_code != 0
    assert "at least one non-empty QUERY" in result.output


def test_should_fuse_dense_and_lexical_channels_for_every_query(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    shared = SearchResult(
        "src/a.py",
        3,
        4,
        "def publish():\n    pass",
        0.91,
        "function",
        "publish",
    )
    explorer.semantic_search.side_effect = [
        [shared],
        [
            shared,
            SearchResult(
                "tests/test_a.py",
                8,
                9,
                "def test_publish():\n    pass",
                0.85,
                "function",
                "test_publish",
            ),
        ],
    ]
    explorer.get_outline.return_value = []
    explorer.get_source_lines.return_value = []
    explorer.find_paths.return_value = []
    explorer.find_text_lines.return_value = []
    lexical_store = MagicMock()
    lexical_store.search.return_value = []
    embedder = MagicMock()

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader",
            return_value=explorer,
        ),
        patch(
            "smritikosh.exploration_cli.DuckDBBm25Store",
            return_value=lexical_store,
        ),
        patch(
            "smritikosh.exploration_cli.make_embedder",
            return_value=embedder,
        ) as embedder_factory,
    ):
        output = _run_search(db_path, "publication flow", "publication tests")

    # Assert
    embedder_factory.assert_called_once_with()
    assert explorer.semantic_search.call_count == 2
    assert explorer.semantic_search.call_args_list[1].args[0] == ["publication tests"]
    assert lexical_store.search.call_count == 2
    payload = toons.loads(output)
    assert [row["path"] for row in payload["search_results"]] == [
        "src/a.py",
        "tests/test_a.py",
    ]


def test_should_expand_search_window_to_complete_definition(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.semantic_search.return_value = [
        SearchResult(
            "src/helper.py",
            50,
            60,
            "except Exception:\n    raise",
            0.8,
            "function",
            "record_and_publish",
        )
    ]
    explorer.get_outline.return_value = [
        OutlineEntry("record_and_publish", "function", 10, 60)
    ]
    explorer.get_source_lines.return_value = []
    explorer.find_paths.return_value = []
    explorer.find_text_lines.return_value = []
    lexical_store = MagicMock()
    lexical_store.search.return_value = []

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader",
            return_value=explorer,
        ),
        patch(
            "smritikosh.exploration_cli.DuckDBBm25Store",
            return_value=lexical_store,
        ),
        patch(
            "smritikosh.exploration_cli.make_embedder",
            return_value=MagicMock(),
        ),
    ):
        output = _run_search(db_path, "publication retry", "--max-results", "1")

    # Assert
    row = toons.loads(output)["search_results"][0]
    assert (row["start_line"], row["end_line"]) == (10, 60)


def test_should_prefer_relevant_source_over_higher_scoring_documentation(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.semantic_search.return_value = [
        SearchResult(
            "docs/retry.md",
            1,
            5,
            "# Generic retry guide",
            0.6,
            "section",
            "Retry guide",
        ),
        SearchResult(
            "clients/redis/client.py",
            40,
            50,
            "def decrement():\n    return redis.decr()",
            0.5,
            "method",
            "decrement",
        ),
    ]
    explorer.get_outline.return_value = [OutlineEntry("decrement", "method", 40, 50)]
    explorer.get_source_lines.return_value = [
        SourceLine(40, "def decrement():"),
        SourceLine(50, "    return redis.decr()"),
    ]
    explorer.find_paths.return_value = []
    explorer.find_text_lines.return_value = []
    explorer.get_chunks_by_ids.return_value = {
        "redis": IndexedChunk(
            "redis",
            "clients/redis/client.py",
            40,
            50,
            "def decrement():\n    return redis.decr()",
            "method",
            "decrement",
        )
    }
    lexical_store = MagicMock()
    lexical_store.search.return_value = [("redis", 1.0)]

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader",
            return_value=explorer,
        ),
        patch(
            "smritikosh.exploration_cli.DuckDBBm25Store",
            return_value=lexical_store,
        ),
        patch(
            "smritikosh.exploration_cli.make_embedder",
            return_value=MagicMock(),
        ),
    ):
        output = _run_search(db_path, "redis decrement retry", "--max-results", "1")

    # Assert
    payload = toons.loads(output)
    assert [row["path"] for row in payload["search_results"]] == [
        "clients/redis/client.py"
    ]


def test_should_print_query_legend_and_locations_as_prose(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.return_value = [
        SearchLocation("src/s3.py", 14, 21, "upload", ("upload flow",)),
    ]

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        output = _run_search(db_path, "upload flow", "--prose")

    # Assert
    assert output == "Q1  upload flow\n\nsrc/s3.py:14-21  upload  Q1\n"


def test_should_close_reader_when_bm25_store_construction_fails(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    reader = MagicMock()

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader",
            return_value=reader,
        ),
        patch(
            "smritikosh.exploration_cli.DuckDBBm25Store",
            side_effect=RuntimeError("cannot open BM25"),
        ),
        patch(
            "smritikosh.exploration_cli.make_embedder",
            return_value=MagicMock(),
        ),
    ):
        result = CliRunner().invoke(
            main,
            ["explore", "search", "query", "--db-path", str(db_path)],
        )

    # Assert
    assert result.exit_code != 0
    reader.close.assert_called_once_with()


def test_should_report_a_missing_bm25_index_as_a_command_error(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    service = MagicMock()
    service.search.side_effect = RuntimeError("BM25 index is missing; rebuild")

    # Act
    with (
        patch(
            "smritikosh.exploration_cli.DuckDBSourceReader", return_value=MagicMock()
        ),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.make_embedder", return_value=MagicMock()),
    ):
        result = CliRunner().invoke(
            main,
            ["explore", "search", "query", "--db-path", str(db_path)],
        )

    # Assert
    assert result.exit_code != 0
    assert "BM25 index is missing" in result.output


def test_should_print_numbered_source_for_a_requested_line_range(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.get_source_lines.return_value = [
        SourceLine(9, "def total():"),
        SourceLine(10, "    return 1"),
    ]

    # Act
    with patch(
        "smritikosh.exploration_cli.ReadOnlyExplorer",
        return_value=explorer,
    ):
        result = CliRunner().invoke(
            main,
            [
                "explore",
                "chunks",
                "src/a.py",
                "--start-line",
                "9",
                "--end-line",
                "10",
                "--db-path",
                str(db_path),
            ],
        )

    # Assert
    assert result.exit_code == 0
    explorer.get_source_lines.assert_called_once_with(
        "src/a.py",
        start_line=9,
        end_line=10,
    )
    explorer.get_chunks.assert_not_called()
    assert result.output == "src/a.py:9-10\n 9  def total():\n10      return 1\n"


def test_should_outline_the_path_when_no_line_range_is_given(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.get_outline.return_value = [
        OutlineEntry("load_orders", "function", 8, 12),
        OutlineEntry(None, "constant", 100, 140),
    ]

    # Act
    with patch(
        "smritikosh.exploration_cli.ReadOnlyExplorer",
        return_value=explorer,
    ):
        result = CliRunner().invoke(
            main,
            ["explore", "chunks", "src/a.py", "--prose", "--db-path", str(db_path)],
        )

    # Assert
    assert result.exit_code == 0
    explorer.get_chunks.assert_not_called()
    assert result.output == (
        "src/a.py  (2 entries)\n8-12     function  load_orders\n100-140  constant  -\n"
    )


def test_should_print_source_lines_as_numbered_text_even_when_toon_is_asked_for(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.get_source_lines.return_value = [SourceLine(9, "def total():")]

    # Act
    with patch(
        "smritikosh.exploration_cli.ReadOnlyExplorer",
        return_value=explorer,
    ):
        result = CliRunner().invoke(
            main,
            [
                "explore",
                "chunks",
                "src/a.py",
                "--start-line",
                "9",
                "--toon",
                "--db-path",
                str(db_path),
            ],
        )

    # Assert
    assert result.exit_code == 0
    assert result.output == "src/a.py:9-9\n9  def total():\n"


def test_should_print_stored_chunks_when_full_is_requested(tmp_path: Path) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.get_chunks.return_value = []

    # Act
    with patch(
        "smritikosh.exploration_cli.ReadOnlyExplorer",
        return_value=explorer,
    ):
        result = CliRunner().invoke(
            main,
            ["explore", "chunks", "src/a.py", "--full", "--db-path", str(db_path)],
        )

    # Assert
    assert result.exit_code == 0
    explorer.get_chunks.assert_called_once_with("src/a.py")
    explorer.get_outline.assert_not_called()


def test_should_read_multiple_source_ranges_with_one_chunks_command(
    tmp_path: Path,
) -> None:
    # Arrange
    db_path: Path = tmp_path / "index.duckdb"
    db_path.touch()
    explorer = _mock_explorer()
    explorer.get_source_lines.side_effect = [
        [SourceLine(1, "first")],
        [SourceLine(7, "second")],
    ]

    # Act
    with patch(
        "smritikosh.exploration_cli.ReadOnlyExplorer",
        return_value=explorer,
    ):
        result = CliRunner().invoke(
            main,
            [
                "explore",
                "chunks",
                "--range",
                "src/a.py",
                "1",
                "2",
                "--range",
                "src/b.py",
                "7",
                "8",
                "--db-path",
                str(db_path),
            ],
        )

    # Assert
    assert result.exit_code == 0
    assert explorer.get_source_lines.call_count == 2
    assert result.output == ("src/a.py:1-1\n1  first\n\nsrc/b.py:7-7\n7  second\n")
