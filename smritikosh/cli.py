"""smritikosh CLI: index a repo, then search via vector similarity."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import click

from smritikosh.constants import DEFAULT_DB_PATH
from smritikosh.exploration_cli import explore

if TYPE_CHECKING:
    from smritikosh.adapters.storage.duckdb import DuckDBAdapter
    from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore
    from smritikosh.ports.embedder import Embedder
    from smritikosh.ports.storage import StorageAdapter
    from smritikosh.ports.vector_store import VectorStore


# ── Embedder factory ──────────────────────────────────────────────────────────


def _make_embedder() -> Embedder:
    """Thin Click wrapper around :func:`smritikosh.adapters.embedder.make_embedder`."""
    from smritikosh.adapters.embedder import make_embedder

    return make_embedder()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _open_stores(db_path: str) -> tuple[DuckDBAdapter, DuckDBVectorStore]:
    """Open and return a *(storage, vector_store)* pair sharing one DuckDB connection.

    The caller is responsible for calling ``storage.close()`` when done;
    ``vector_store`` shares the connection so only ``storage`` needs closing.
    """
    from smritikosh.adapters.storage.duckdb import DuckDBAdapter
    from smritikosh.adapters.vector_store.duckdb import DuckDBVectorStore

    storage = DuckDBAdapter(db_path)
    vector_store = DuckDBVectorStore(db_path, con=storage.con)
    return storage, vector_store


def _build_index_with_progress(
    repo_path: str,
    embedder: Embedder,
    storage: StorageAdapter,
    vector_store: VectorStore,
    *,
    n_files: int | None = None,
    full: bool = False,
) -> None:
    """Run :func:`build_index` with a Click progress bar driven by ``on_file_indexed``.

    *n_files* is the bar length.  When omitted it is counted from *repo_path*
    with the same filters as the pipeline.  Safe to call from a worker thread
    (the bar lives on the same thread as the pipeline).
    """
    from smritikosh.indexing.pipeline import build_index, count_source_files

    if n_files is None:
        n_files = count_source_files(repo_path)

    with click.progressbar(
        length=n_files,
        show_eta=True,
        show_percent=True,
        bar_template="  %(bar)s  %(info)s",
        fill_char="█",
        empty_char="░",
    ) as bar:

        def _on_file(path: str) -> None:  # noqa: E306
            bar.update(1)

        build_index(
            repo_path,
            embedder,
            storage,
            vector_store,
            on_file_indexed=_on_file,
            full=full,
        )


async def _watch_loop(
    repo_path: str,
    embedder: Embedder,
    storage: StorageAdapter,
    vector_store: VectorStore,
) -> None:
    """Re-index whenever a file inside *repo_path* changes.

    Requires the ``smritikosh[watch]`` optional extra (``watchfiles`` package).
    ``build_index`` uses ``asyncio.run()`` internally, so it is dispatched via
    ``asyncio.to_thread`` to run in a worker thread with its own event loop.
    """
    try:
        from watchfiles import awatch  # type: ignore[import]
    except ImportError as exc:
        raise click.ClickException(
            "watchfiles is required for --watch. "
            "Install it with: pip install watchfiles"
        ) from exc

    async for changes in awatch(repo_path):
        click.echo(f"  {len(changes)} change(s) detected — re-indexing …")
        # build_index calls asyncio.run() internally; dispatch to a worker
        # thread so it can create its own loop without conflicting with ours.
        await asyncio.to_thread(
            _build_index_with_progress,
            repo_path,
            embedder,
            storage,
            vector_store,
        )
        click.echo("  Done.")


# ── CLI group ─────────────────────────────────────────────────────────────────


@click.group()
def main() -> None:
    """smritikosh -- semantic vector search over a codebase."""


main.add_command(explore)


# ── smritikosh index ──────────────────────────────────────────────────────────


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--db-path",
    default=DEFAULT_DB_PATH,
    show_default=True,
    type=click.Path(),
    help="DuckDB database file.",
)
@click.option(
    "--watch",
    "-L",
    is_flag=True,
    help="Stay alive and re-index on file changes (requires watchfiles).",
)
@click.option(
    "--full",
    is_flag=True,
    help="Force a full rebuild (clears the indexes and incremental caches).",
)
def index(
    repo_path: str,
    db_path: str,
    watch: bool,
    full: bool,
) -> None:
    """Build or incrementally update the vector index for REPO_PATH."""
    from smritikosh.indexing.pipeline import count_source_files

    emb = _make_embedder()
    storage, vector_store = _open_stores(db_path)
    try:
        if full:
            click.echo("Clearing indexes and incremental caches — full rebuild forced.")

        n_files = count_source_files(repo_path)
        click.echo(f"Indexing {repo_path!r} — {n_files} source file(s) …")

        t0 = time.perf_counter()
        _build_index_with_progress(
            repo_path,
            emb,
            storage,
            vector_store,
            n_files=n_files,
            full=full,
        )
        elapsed = time.perf_counter() - t0
        click.echo(f"Done in {elapsed:.1f}s.")

        if watch:
            click.echo(f"Watching {repo_path!r} for changes (Ctrl-C to stop) …")
            try:
                asyncio.run(_watch_loop(repo_path, emb, storage, vector_store))
            except KeyboardInterrupt:
                click.echo("\nWatch stopped.")
    finally:
        storage.close()


# ── smritikosh search ─────────────────────────────────────────────────────────


@main.command()
@click.argument("query")
@click.option(
    "--top-k",
    default=10,
    show_default=True,
    help="Number of results to return.",
)
@click.option(
    "--db-path",
    default=DEFAULT_DB_PATH,
    show_default=True,
    type=click.Path(),
    help="DuckDB database file.",
)
def search(
    query: str,
    top_k: int,
    db_path: str,
) -> None:
    """Run a semantic search QUERY against the built vector index."""
    from smritikosh.indexing.vector_index import VectorIndex

    storage, vector_store = _open_stores(db_path)
    results = []  # populated inside try; display happens after storage is closed
    t0 = time.perf_counter()
    try:
        if vector_store.get_stored_dims() is None:
            raise click.ClickException(
                f"No index found at {db_path!r}. "
                "Run `smritikosh index <repo_path>` first."
            )
        emb = _make_embedder()
        idx = VectorIndex(vector_store, storage, emb)
        results = asyncio.run(idx.search(query, top_k))
    finally:
        storage.close()

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not results:
        click.echo(f"No results found.  ({elapsed_ms:.0f} ms)")
        return

    click.echo(f"Found {len(results)} result(s) in {elapsed_ms:.0f} ms:\n")
    for r in results:
        line_range = f"{r.start_line}-{r.end_line}"
        kind_str = f"  [{r.chunk_kind}]" if r.chunk_kind else ""
        click.echo(f"{r.path}:{line_range}  score={r.score:.3f}{kind_str}")
        for line in (r.snippet or "").splitlines():
            click.echo(f"    {line}")
        click.echo()


if __name__ == "__main__":
    main()
