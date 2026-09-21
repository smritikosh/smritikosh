"""Tests for smritikosh.cli — index and search commands.

Coverage map
────────────
_make_embedder          – constructs FastEmbedEmbedder with CodeRankEmbed
_open_stores            – correct types, shared DuckDB connection
index command           – happy path, --full ordering, --db-path passthrough,
                          --watch output, watch re-index progress callback,
                          storage.close() in success + error paths
search command          – result formatting, empty result, not-indexed guard,
                          --top-k passthrough, multiline snippet indent,
                          no [kind] brackets when chunk_kind is None,
                          storage.close() in success + not-indexed paths
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from smritikosh.cli import _make_embedder, _open_stores, _watch_loop, main
from smritikosh.models import SearchResult

# ── Stubs ─────────────────────────────────────────────────────────────────────


class _StubEmbedder:
    """Minimal Embedder that never loads a model or hits the network."""

    dims = 3
    model_id = "stub:3"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]] * len(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]] * len(texts)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def repo(tmp_path: Path) -> str:
    """An empty directory that satisfies click.Path(exists=True, file_okay=False)."""
    return str(tmp_path)


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.duckdb")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_stores(dims: int | None = 3) -> tuple[MagicMock, MagicMock]:
    """Return a *(storage, vector_store)* MagicMock pair.

    *dims* is what ``vector_store.get_stored_dims()`` returns — ``None``
    signals "nothing indexed yet", which the search command guards against.
    """
    storage = MagicMock()
    storage.con = MagicMock()
    vector_store = MagicMock()
    vector_store.get_stored_dims.return_value = dims
    return storage, vector_store


def _result(
    *,
    path: str = "src/billing.py",
    start_line: int = 10,
    end_line: int = 15,
    snippet: str = "def pay(): pass",
    score: float = 0.85,
    chunk_kind: str | None = "function",
) -> SearchResult:
    return SearchResult(
        path=path,
        start_line=start_line,
        end_line=end_line,
        snippet=snippet,
        score=score,
        chunk_kind=chunk_kind,
    )


# ── _make_embedder ────────────────────────────────────────────────────────────


def test_make_embedder_uses_coderankembed() -> None:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder
    from smritikosh.constants import DEFAULT_MODEL

    emb = _make_embedder()

    assert isinstance(emb, FastEmbedEmbedder)
    assert emb._model_name == DEFAULT_MODEL
    assert emb._model_name == "nomic-ai/CodeRankEmbed"


# ── _open_stores ──────────────────────────────────────────────────────────────


def test_open_stores_returns_correct_adapter_types(tmp_path: Path) -> None:
    from smritikosh.adapters.storage.duckdb import DuckDBAdapter
    from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore

    storage, vector_store = _open_stores(str(tmp_path / "t.duckdb"))
    try:
        assert isinstance(storage, DuckDBAdapter)
        assert isinstance(vector_store, DuckDBVectorStore)
    finally:
        storage.close()


def test_open_stores_vector_store_shares_connection(tmp_path: Path) -> None:
    """storage and vector_store must share exactly one DuckDB connection."""
    storage, vector_store = _open_stores(str(tmp_path / "t.duckdb"))
    try:
        assert vector_store._con is storage.con
    finally:
        storage.close()


# ── smritikosh index ──────────────────────────────────────────────────────────


def test_index_exits_zero_and_prints_done(runner, repo, db_path) -> None:
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.indexing.pipeline.build_index"),
    ):
        result = runner.invoke(main, ["index", repo, "--db-path", db_path])

    assert result.exit_code == 0
    assert "Done in" in result.output


def test_index_full_announces_the_rebuild(runner, repo, db_path) -> None:
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.indexing.pipeline.build_index"),
    ):
        result = runner.invoke(main, ["index", repo, "--db-path", db_path, "--full"])

    assert result.exit_code == 0
    assert "Clearing indexes and incremental caches" in result.output


def test_index_full_delegates_clearing_to_build_index(runner, repo, db_path) -> None:
    """The CLI owns no clearing logic — build_index clears what it rebuilds.

    Splitting the two let a direct build_index(full=True) clear the indexes
    without the caches, which emptied the index instead of rebuilding it.
    """
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.indexing.pipeline.build_index") as mock_build,
    ):
        runner.invoke(main, ["index", repo, "--db-path", db_path, "--full"])

    storage.clear_caches.assert_not_called()
    storage.clear_nodes.assert_not_called()
    assert mock_build.call_args.kwargs["full"] is True


def test_index_without_full_does_not_request_a_rebuild(runner, repo, db_path) -> None:
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.indexing.pipeline.build_index") as mock_build,
    ):
        runner.invoke(main, ["index", repo, "--db-path", db_path])

    assert mock_build.call_args.kwargs["full"] is False


def test_index_db_path_forwarded_to_open_stores(runner, repo, tmp_path: Path) -> None:
    storage, vector_store = _mock_stores()
    custom_db = str(tmp_path / "custom.duckdb")

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch(
            "smritikosh.cli._open_stores", return_value=(storage, vector_store)
        ) as mock_open,
        patch("smritikosh.indexing.pipeline.build_index"),
    ):
        runner.invoke(main, ["index", repo, "--db-path", custom_db])

    mock_open.assert_called_once_with(custom_db)


def test_index_storage_closed_on_success(runner, repo, db_path) -> None:
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.indexing.pipeline.build_index"),
    ):
        runner.invoke(main, ["index", repo, "--db-path", db_path])

    storage.close.assert_called_once()


def test_index_storage_closed_when_build_raises(runner, repo, db_path) -> None:
    """try/finally must close storage even if build_index raises."""
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch(
            "smritikosh.indexing.pipeline.build_index",
            side_effect=RuntimeError("unexpected failure"),
        ),
    ):
        runner.invoke(main, ["index", repo, "--db-path", db_path])

    storage.close.assert_called_once()


def test_index_watch_flag_prints_watching_message(runner, repo, db_path) -> None:
    """--watch/-L shows the 'Watching …' banner (watch loop itself stubbed out)."""
    storage, vector_store = _mock_stores()

    with (
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.indexing.pipeline.build_index"),
        # Replace the infinite async loop with a no-op coroutine.
        patch("smritikosh.cli._watch_loop", new_callable=AsyncMock),
    ):
        result = runner.invoke(main, ["index", repo, "--db-path", db_path, "--watch"])

    assert result.exit_code == 0
    assert "Watching" in result.output


def test_watch_loop_passes_on_file_indexed(repo) -> None:
    """Re-index must drive the same progress-bar callback as the first index."""
    storage, vector_store = _mock_stores()
    captured: dict[str, object] = {}

    async def fake_awatch(*_args, **_kwargs):
        yield {("modified", "foo.py")}

    def fake_build(*_args, **kwargs) -> None:
        captured["on_file_indexed"] = kwargs.get("on_file_indexed")
        callback = kwargs.get("on_file_indexed")
        if callback is not None:
            callback("foo.py")

    with (
        patch("watchfiles.awatch", fake_awatch),
        patch(
            "smritikosh.indexing.pipeline.build_index",
            side_effect=fake_build,
        ),
        patch(
            "smritikosh.indexing.pipeline.count_source_files",
            return_value=1,
        ),
    ):
        asyncio.run(_watch_loop(repo, _StubEmbedder(), storage, vector_store))

    assert callable(captured.get("on_file_indexed"))


# ── smritikosh search ─────────────────────────────────────────────────────────


def test_search_prints_formatted_result(runner, db_path) -> None:
    storage, vector_store = _mock_stores(dims=3)
    mock_idx = MagicMock()
    mock_idx.search = AsyncMock(return_value=[_result()])

    with (
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.indexing.vector_index.VectorIndex", return_value=mock_idx),
    ):
        result = runner.invoke(
            main, ["search", "token validation", "--db-path", db_path]
        )

    assert result.exit_code == 0
    assert "src/billing.py:10-15" in result.output
    assert "score=0.850" in result.output
    assert "[function]" in result.output
    assert "def pay(): pass" in result.output


def test_search_prints_no_results_message_when_index_is_empty(runner, db_path) -> None:
    storage, vector_store = _mock_stores(dims=3)
    mock_idx = MagicMock()
    mock_idx.search = AsyncMock(return_value=[])

    with (
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.indexing.vector_index.VectorIndex", return_value=mock_idx),
    ):
        result = runner.invoke(main, ["search", "q", "--db-path", db_path])

    assert result.exit_code == 0
    assert "No results found" in result.output


def test_search_errors_when_nothing_indexed(runner, db_path) -> None:
    """get_stored_dims() returns None → friendly ClickException, not a raw crash."""
    storage, vector_store = _mock_stores(dims=None)

    with patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)):
        result = runner.invoke(main, ["search", "q", "--db-path", db_path])

    assert result.exit_code != 0
    assert "No index found" in result.output
    assert "smritikosh index" in result.output  # points user at the fix


def test_search_storage_closed_on_success(runner, db_path) -> None:
    storage, vector_store = _mock_stores(dims=3)
    mock_idx = MagicMock()
    mock_idx.search = AsyncMock(return_value=[])

    with (
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.indexing.vector_index.VectorIndex", return_value=mock_idx),
    ):
        runner.invoke(main, ["search", "q", "--db-path", db_path])

    storage.close.assert_called_once()


def test_search_storage_closed_when_not_indexed(runner, db_path) -> None:
    """try/finally must close storage even when the 'not indexed' guard fires."""
    storage, vector_store = _mock_stores(dims=None)

    with patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)):
        runner.invoke(main, ["search", "q", "--db-path", db_path])

    storage.close.assert_called_once()


def test_search_top_k_forwarded_to_vector_index(runner, db_path) -> None:
    storage, vector_store = _mock_stores(dims=3)
    mock_idx = MagicMock()
    mock_idx.search = AsyncMock(return_value=[])

    with (
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.indexing.vector_index.VectorIndex", return_value=mock_idx),
    ):
        runner.invoke(main, ["search", "find me", "--top-k", "3", "--db-path", db_path])

    mock_idx.search.assert_awaited_once_with("find me", 3)


def test_search_indents_every_line_of_multiline_snippet(runner, db_path) -> None:
    storage, vector_store = _mock_stores(dims=3)
    mock_idx = MagicMock()
    mock_idx.search = AsyncMock(
        return_value=[_result(snippet="def foo():\n    return 42")]
    )

    with (
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.indexing.vector_index.VectorIndex", return_value=mock_idx),
    ):
        result = runner.invoke(main, ["search", "q", "--db-path", db_path])

    indented = [line for line in result.output.splitlines() if line.startswith("    ")]
    assert len(indented) == 2
    assert indented[0].strip() == "def foo():"
    assert indented[1].strip() == "return 42"


def test_search_omits_chunk_kind_brackets_when_kind_is_none(runner, db_path) -> None:
    storage, vector_store = _mock_stores(dims=3)
    mock_idx = MagicMock()
    mock_idx.search = AsyncMock(return_value=[_result(chunk_kind=None)])

    with (
        patch("smritikosh.cli._open_stores", return_value=(storage, vector_store)),
        patch("smritikosh.cli._make_embedder", return_value=_StubEmbedder()),
        patch("smritikosh.indexing.vector_index.VectorIndex", return_value=mock_idx),
    ):
        result = runner.invoke(main, ["search", "q", "--db-path", db_path])

    score_line = next(line for line in result.output.splitlines() if "score=" in line)
    assert "[" not in score_line
