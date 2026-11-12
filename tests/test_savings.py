"""Status, search ledger, and lines-not-opened totals."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import toons
from click.testing import CliRunner

from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.cli import main
from smritikosh.index_status import IndexStatus, load_index_status
from smritikosh.models import ExplainedHit, SearchLocation
from smritikosh.ports.embedder import EmbeddingIdentity
from smritikosh.savings import (
    SearchSavings,
    _as_date,
    format_explain_hit,
    measure_search,
    read_indexed_roots,
    record_search_savings,
    remember_indexed_root,
    savings_total,
)

_ONNX = EmbeddingIdentity("nomic-ai/CodeRankEmbed", "coderank-onnx-int8-v1", 768)


def _index(path: Path, identity: EmbeddingIdentity | None = _ONNX) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE nodes (
            id TEXT, kind TEXT, name TEXT, path TEXT, metadata JSON
        )
        """
    )
    con.execute(
        """
        INSERT INTO nodes VALUES
            ('file:a.py', 'file', 'a.py', 'a.py', '{"language": "python"}'),
            ('c1', 'chunk', 'f', 'a.py', '{}'),
            ('c2', 'chunk', 'g', 'b.ts', '{}'),
            ('c3', 'chunk', 'h', 'README', '{}')
        """
    )
    con.execute("CREATE TABLE vectors (chunk_id TEXT, vector FLOAT[3])")
    con.execute("INSERT INTO vectors VALUES ('c1', [0.1, 0.2, 0.3])")
    con.execute("CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT)")
    if identity is not None:
        con.execute(
            """
            INSERT INTO kv_store VALUES
                ('embedding_identity', ?),
                ('embedder_dims', ?)
            """,
            [json.dumps(identity.__dict__), str(identity.dimensions)],
        )
    con.close()


def test_status_prints_coverage_identity_and_languages(tmp_path: Path) -> None:
    db_path = tmp_path / "smritikosh.duckdb"
    _index(db_path)

    result = CliRunner().invoke(
        main,
        ["status", "--db-path", str(db_path)],
    )

    assert result.exit_code == 0, result.output
    assert f"index {db_path}" in result.output
    assert "files 1" in result.output
    assert "chunks 3" in result.output
    assert "vectors 1" in result.output
    assert "dimensions 768" in result.output
    assert "coverage 0.33" in result.output
    assert "model nomic-ai/CodeRankEmbed" in result.output
    assert "vector space coderank-onnx-int8-v1" in result.output
    assert "languages .py 1, .ts 1, (none) 1" in result.output
    assert "Some chunks were stored and not embedded." in result.output


def test_status_toon_has_the_same_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "smritikosh.duckdb"
    _index(db_path)

    with patch(
        "smritikosh.cli.warm_server_label",
        return_value="in-process ONNX",
    ):
        result = CliRunner().invoke(
            main,
            ["status", "--db-path", str(db_path), "--toon"],
        )

    assert result.exit_code == 0, result.output
    payload = toons.loads(result.output)
    assert payload["files"] == 1
    assert payload["vectors"] == 1
    assert payload["warm_server"] == "in-process ONNX"
    assert payload["languages"][0] == {"suffix": ".py", "chunks": 1}


def test_status_names_a_missing_index(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["status", "--db-path", str(tmp_path / "missing.duckdb")],
    )

    assert result.exit_code != 0
    assert "No index" in result.output


def test_status_rejects_a_file_that_is_not_an_index(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()

    result = CliRunner().invoke(main, ["status", "--db-path", str(db_path)])

    assert result.exit_code != 0
    assert "not a Smritikosh index" in result.output


def test_status_warns_when_this_machine_cannot_read_the_vector_space(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "smritikosh.duckdb"
    _index(
        db_path,
        EmbeddingIdentity("nomic-ai/CodeRankEmbed", "coderank-pytorch-fp32-v1", 768),
    )

    with patch("smritikosh.index_status._mps_available", return_value=False):
        report = load_index_status(str(db_path), warm_server="off")

    assert any("Apple GPU" in warning for warning in report.warnings)


def test_status_warns_on_an_unknown_vector_space(tmp_path: Path) -> None:
    db_path = tmp_path / "smritikosh.duckdb"
    _index(db_path, EmbeddingIdentity("other", "other-v1", 3))

    report = load_index_status(str(db_path), warm_server="off")

    assert any("other-v1" in warning for warning in report.warnings)


def test_status_warns_when_the_index_has_no_embedding_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "smritikosh.duckdb"
    _index(db_path, identity=None)

    report = load_index_status(str(db_path), warm_server="off")

    assert report.model is None
    assert any("No embedding identity" in warning for warning in report.warnings)


def test_status_lines_format_size_and_an_empty_index() -> None:
    report = IndexStatus(
        index_path="smritikosh.duckdb",
        size_bytes=2048,
        files=0,
        chunks=0,
        vectors=0,
        dimensions=None,
        coverage=None,
        model=None,
        vector_space=None,
        warm_server="off",
        languages=(),
        warnings=(),
    )
    text = "\n".join(report.lines())
    assert "size 2.0 KB" in text
    assert "coverage n/a" in text
    assert "dimensions none" in text
    assert "languages none" in text

    large = IndexStatus(
        index_path="smritikosh.duckdb",
        size_bytes=2 * 1024 * 1024,
        files=0,
        chunks=0,
        vectors=0,
        dimensions=3,
        coverage=1.0,
        model="m",
        vector_space="s",
        warm_server="off",
        languages=((".py", 1),),
        warnings=(),
    )
    assert "size 2.0 MB" in "\n".join(large.lines())
    tiny = IndexStatus(
        index_path="smritikosh.duckdb",
        size_bytes=12,
        files=0,
        chunks=0,
        vectors=0,
        dimensions=3,
        coverage=1.0,
        model="m",
        vector_space="s",
        warm_server="off",
        languages=((".py", 1),),
        warnings=(),
    )
    assert "size 12 B" in "\n".join(tiny.lines())


def test_savings_is_zero_until_a_search_is_recorded(tmp_path: Path) -> None:
    missing = CliRunner().invoke(
        main,
        ["savings", "--db-path", str(tmp_path / "missing.duckdb")],
    )
    assert missing.exit_code == 0
    assert "searches 0" in missing.output
    assert "since" not in missing.output

    empty = tmp_path / "empty.duckdb"
    duckdb.connect(str(empty)).close()
    total = savings_total(str(empty))
    assert total.searches == 0
    assert total.since is None


def test_search_records_lines_not_opened_without_explain(tmp_path: Path) -> None:
    db_path = tmp_path / "index.duckdb"
    _index(db_path)
    source = tmp_path / "src" / "s3.py"
    source.parent.mkdir()
    source.write_text("\n".join(f"line {i}" for i in range(30)) + "\n")
    service = MagicMock()
    service.search.return_value = [
        SearchLocation("src/s3.py", 14, 21, "upload", ("upload flow",)),
    ]
    reader = MagicMock()
    reader.embedding_identity.return_value = _ONNX
    reader.indexed_roots.return_value = [str(tmp_path)]

    with (
        patch("smritikosh.exploration_cli.DuckDBSourceReader", return_value=reader),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.resolve_embedder", return_value=MagicMock()),
    ):
        result = CliRunner().invoke(
            main,
            ["explore", "search", "upload flow", "--db-path", str(db_path)],
        )

    assert result.exit_code == 0, result.output
    payload = toons.loads(result.output)
    assert payload["search_results"][0]["path"] == "src/s3.py"
    total = savings_total(str(db_path))
    assert total.searches == 1
    assert total.since == date.today()
    assert total.lines_cited == 8
    assert total.lines_in_files == 30
    assert total.lines_not_opened == 22

    printed = CliRunner().invoke(main, ["savings", "--db-path", str(db_path)])
    assert f"since {date.today().isoformat()}" in printed.output
    assert "searches 1" in printed.output
    assert "lines cited 8" in printed.output
    assert "lines in those files 30" in printed.output
    assert "lines not opened 22" in printed.output


def test_explain_prints_channel_rank_and_the_widened_span(tmp_path: Path) -> None:
    db_path = tmp_path / "index.duckdb"
    _index(db_path)
    service = MagicMock()
    service.explained.return_value = [
        ExplainedHit(
            SearchLocation("src/s3.py", 10, 40, "upload", ("upload flow",)),
            dense_rank=1,
            lexical_rank=4,
            chunk_start_line=14,
            chunk_end_line=16,
        )
    ]
    reader = MagicMock()
    reader.embedding_identity.return_value = _ONNX
    reader.indexed_roots.return_value = []

    with (
        patch("smritikosh.exploration_cli.DuckDBSourceReader", return_value=reader),
        patch("smritikosh.exploration_cli.DuckDBBm25Store", return_value=MagicMock()),
        patch("smritikosh.exploration_cli.HybridSearchService", return_value=service),
        patch("smritikosh.exploration_cli.resolve_embedder", return_value=MagicMock()),
    ):
        result = CliRunner().invoke(
            main,
            [
                "explore",
                "search",
                "upload flow",
                "--explain",
                "--db-path",
                str(db_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "dense rank 1, lexical rank 4" in result.output
    assert "widened 14-16 -> 10-40" in result.output
    assert "1 ranges, 31 lines cited" in result.output
    assert "lines not opened" in result.output
    toons.loads(result.output.split("dense rank", maxsplit=1)[0])


def test_explain_line_names_a_single_channel_or_none() -> None:
    location = SearchLocation("a.py", 1, 2, None, ())
    lexical = format_explain_hit(
        ExplainedHit(location, None, 2, 1, 2),
        {},
    )
    unranked = format_explain_hit(
        ExplainedHit(location, None, None, 1, 2),
        {},
    )
    assert "lexical rank 2" in lexical
    assert "dense rank" not in lexical
    assert "unranked" in unranked


def test_missing_files_do_not_invent_unread_lines(tmp_path: Path) -> None:
    savings = measure_search(
        [
            SearchLocation("missing.py", 1, 4, None, ()),
            SearchLocation("missing.py", 2, 3, None, ()),
        ],
        [tmp_path],
    )
    assert savings.ranges == 2
    assert savings.lines_cited == 6
    assert savings.lines_in_files == 4
    assert savings.lines_not_opened == 0


def test_index_remembers_each_repo_root_once(tmp_path: Path) -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT)")

    remember_indexed_root(con, str(tmp_path))
    remember_indexed_root(con, str(tmp_path))

    assert read_indexed_roots(con) == [str(tmp_path.resolve())]


def test_reader_reports_stored_roots_and_ignores_a_bad_value(tmp_path: Path) -> None:
    db_path = tmp_path / "index.duckdb"
    _index(db_path)
    con = duckdb.connect(str(db_path))
    remember_indexed_root(con, str(tmp_path))
    con.close()

    reader = DuckDBSourceReader(str(db_path))
    try:
        assert reader.indexed_roots() == [str(tmp_path.resolve())]
    finally:
        reader.close()

    con = duckdb.connect(str(db_path))
    con.execute(
        "UPDATE kv_store SET value = ? WHERE key = 'indexed_roots'",
        ['"not-a-list"'],
    )
    con.close()
    reader = DuckDBSourceReader(str(db_path))
    try:
        assert reader.indexed_roots() == []
    finally:
        reader.close()
    con = duckdb.connect(str(db_path))
    try:
        assert read_indexed_roots(con) == []
    finally:
        con.close()


def test_record_skips_a_path_that_is_not_a_database(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    record_search_savings(str(blocked), SearchSavings(1, 1, 1))


def test_read_roots_is_empty_when_the_table_is_missing() -> None:
    con = duckdb.connect(":memory:")
    assert read_indexed_roots(con) == []
    assert _as_date(None) is None
    assert _as_date(date(2026, 11, 11)) == date(2026, 11, 11)
    assert _as_date(datetime(2026, 11, 11, 1, 2, 3)) == date(2026, 11, 11)
    assert _as_date("2026-11-11T00:00:00") == date(2026, 11, 11)
    assert _as_date("") is None


def test_savings_total_survives_an_unreadable_database(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    assert savings_total(str(blocked)).searches == 0
