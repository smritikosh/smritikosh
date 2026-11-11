"""Tests for the call-graph queries on the DuckDB source reader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.adapters.storage.duckdb import DuckDBAdapter
from smritikosh.engine import _memo
from smritikosh.indexing.pipeline import build_index
from smritikosh.models import ImpactRow
from smritikosh.ports.embedder import Embedder

FILES = {
    "app/__init__.py": "",
    "app/util.py": "def helper():\n    pass\n",
    "app/other.py": "class Box:\n    def helper(self):\n        pass\n",
    "app/service.py": (
        "from app.util import helper\n"
        "\n"
        "\n"
        "def run():\n"
        "    helper()\n"
        "    helper()\n"
        "\n"
        "\n"
        "def unknown(client):\n"
        "    client.helper()\n"
    ),
    "app/api.py": "from app.service import run\n\n\ndef handle():\n    run()\n",
    "app/loop.py": "def ping():\n    pong()\n\n\ndef pong():\n    ping()\n",
}


class _Embedder(Embedder):
    dims = 4
    model_id = "stub:4"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture()
def reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DuckDBSourceReader]:
    # build_index leaves the global memo store on the connection closed below.
    monkeypatch.setattr(_memo, "_memo_store", _memo._memo_store)
    repo: Path = tmp_path / "repo"
    for name, source in FILES.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(source)
    db_path: str = str(tmp_path / "index.duckdb")
    storage = DuckDBAdapter(db_path)
    build_index(str(repo), embedder=_Embedder(), storage=storage)
    storage.close()
    with DuckDBSourceReader(db_path) as reader:
        yield reader


def test_finds_every_definition_with_a_bare_name(reader: DuckDBSourceReader) -> None:
    assert [s.id for s in reader.find_symbols("helper")] == [
        "app/other.py:Box.helper",
        "app/util.py:helper",
    ]


def test_finds_a_definition_by_qualified_name(reader: DuckDBSourceReader) -> None:
    assert [s.id for s in reader.find_symbols("Box.helper")] == [
        "app/other.py:Box.helper"
    ]


def test_finds_a_definition_by_id(reader: DuckDBSourceReader) -> None:
    [symbol] = reader.find_symbols("app/util.py:helper")

    assert (symbol.path, symbol.start_line, symbol.end_line) == ("app/util.py", 1, 2)


def test_finds_nothing_for_an_unknown_name(reader: DuckDBSourceReader) -> None:
    assert reader.find_symbols("missing") == []


def test_lists_direct_callers_with_every_call_line(
    reader: DuckDBSourceReader,
) -> None:
    assert reader.impact("app/util.py:helper") == [
        ImpactRow(1, "app/service.py:run", "app/service.py", (5, 6), 4, 6)
    ]


def test_follows_callers_to_the_requested_depth(reader: DuckDBSourceReader) -> None:
    rows = reader.impact("app/util.py:helper", depth=2)

    assert [(r.depth, r.symbol) for r in rows] == [
        (1, "app/service.py:run"),
        (2, "app/api.py:handle"),
    ]


def test_lists_callees(reader: DuckDBSourceReader) -> None:
    rows = reader.impact("app/api.py:handle", direction="out", depth=2)

    assert rows == [
        ImpactRow(1, "app/service.py:run", "app/api.py", (5,), 4, 6),
        ImpactRow(2, "app/util.py:helper", "app/service.py", (5, 6), 1, 2),
    ]


def test_stops_walking_a_cycle(reader: DuckDBSourceReader) -> None:
    rows = reader.impact("app/loop.py:ping", depth=5)

    assert [(r.depth, r.symbol) for r in rows] == [
        (1, "app/loop.py:pong"),
        (2, "app/loop.py:ping"),
    ]


def test_caps_the_rows(reader: DuckDBSourceReader) -> None:
    assert len(reader.impact("app/util.py:helper", depth=2, limit=1)) == 1


def test_counts_same_named_calls_that_resolved_to_nothing(
    reader: DuckDBSourceReader,
) -> None:
    assert reader.unresolved_calls("app/util.py:helper") == 1


def test_asks_for_a_reindex_when_the_graph_is_missing(tmp_path: Path) -> None:
    db_path: str = str(tmp_path / "old.duckdb")
    duckdb.connect(db_path).close()

    with (
        DuckDBSourceReader(db_path) as reader,
        pytest.raises(RuntimeError, match="no call graph"),
    ):
        reader.find_symbols("helper")
