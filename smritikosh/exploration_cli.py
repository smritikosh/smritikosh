"""CLI commands for read-only, agent-driven index exploration."""

from __future__ import annotations

from dataclasses import asdict

import click
import toons

from smritikosh.adapters.embedder import make_embedder
from smritikosh.adapters.retrieval.duckdb import (
    DuckDBBm25Store,
    DuckDBDenseRetriever,
    DuckDBLexicalRetriever,
)
from smritikosh.adapters.retrieval.source import DuckDBSourceReader
from smritikosh.constants import DEFAULT_DB_PATH
from smritikosh.exploration import (
    IndexedChunk,
    OutlineEntry,
    ReadOnlyExplorer,
    SourceLine,
)
from smritikosh.models import HybridSearchOptions, SearchLocation
from smritikosh.ports.embedder import Embedder
from smritikosh.retrieval.service import HybridSearchService

__all__ = ["explore"]

MAX_QUERIES = 8
MAX_QUERY_CHARS = 500


def _echo_toon(value: object) -> None:
    """Emit TOON — a declared schema for roughly half of JSON's tokens."""
    click.echo(toons.dumps(value))


def _echo_chunks(chunks: list[IndexedChunk]) -> None:
    if not chunks:
        click.echo("No chunks found.")
        return
    for chunk in chunks:
        kind: str = f"  [{chunk.chunk_kind}]" if chunk.chunk_kind else ""
        click.echo(f"{chunk.path}:{chunk.start_line}-{chunk.end_line}{kind}")
        for line in chunk.text.splitlines():
            click.echo(f"    {line}")
        click.echo()


def _echo_outline(entries: list[OutlineEntry], *, path: str) -> None:
    if not entries:
        click.echo("Nothing indexed for that path.")
        return
    click.echo(f"{path}  ({len(entries)} entries)")
    width: int = max(len(f"{e.start_line}-{e.end_line}") for e in entries)
    for entry in entries:
        span: str = f"{entry.start_line}-{entry.end_line}"
        kind: str = entry.chunk_kind or "chunk"
        name: str = entry.symbol or "-"
        click.echo(f"{span:<{width}}  {kind}  {name}")


def _echo_source_lines(lines: list[SourceLine], *, path: str) -> None:
    if not lines:
        click.echo("No source lines found.")
        return
    click.echo(f"{path}:{lines[0].line_number}-{lines[-1].line_number}")
    width: int = len(str(lines[-1].line_number))
    for line in lines:
        click.echo(f"{line.line_number:>{width}}  {line.text}")


def _search_payload(
    locations: list[SearchLocation],
    *,
    queries: tuple[str, ...],
) -> dict[str, object]:
    """Render located definitions as two compact TOON tables.

    Facets are labelled by query id rather than repeated in full: every row
    would otherwise carry the whole query text it matched.
    """
    labels: dict[str, str] = {
        query: f"Q{index}" for index, query in enumerate(queries, start=1)
    }
    return {
        "queries": [{"id": labels[query], "query": query} for query in labels],
        "search_results": [
            {
                "path": location.path,
                "start_line": location.start_line,
                "end_line": location.end_line,
                "symbol": location.symbol or "",
                "facets": "|".join(labels[facet] for facet in location.facets),
            }
            for location in locations
        ],
    }


def _echo_search_prose(
    locations: list[SearchLocation],
    *,
    queries: tuple[str, ...],
) -> None:
    labels: dict[str, str] = {
        query: f"Q{index}" for index, query in enumerate(queries, start=1)
    }
    for query, label in labels.items():
        click.echo(f"{label}  {query}")
    click.echo()
    if not locations:
        click.echo("No results found.")
        return
    for location in locations:
        symbol: str = f"  {location.symbol}" if location.symbol else ""
        facets: str = "|".join(labels[facet] for facet in location.facets)
        click.echo(
            f"{location.path}:{location.start_line}-{location.end_line}"
            f"{symbol}  {facets}"
        )


def _tools_payload() -> dict[str, object]:
    """Describe the exploration workflow in compact, machine-readable rows."""
    return {
        "workflow": [
            {
                "step": 1,
                "action": "search",
                "instruction": "Pass 4-8 focused facets of one question.",
            },
            {
                "step": 2,
                "action": "chunks",
                "instruction": "Read selected search_result paths and line ranges.",
            },
            {
                "step": 3,
                "action": "answer",
                "instruction": "Cite the returned PATH:START-END evidence.",
            },
        ],
        "tools": [
            {
                "name": "search",
                "purpose": "Find relevant definitions and direct references.",
                "usage": (
                    "smritikosh explore search QUERY... "
                    "--max-results 12 [--db-path PATH]"
                ),
                "returns": "TOON rows: path,start_line,end_line,symbol,facets",
            },
            {
                "name": "chunks",
                "purpose": "Read several exact source ranges in one process.",
                "usage": (
                    "smritikosh explore chunks "
                    "--range PATH START END [--range PATH START END ...] "
                    "[--db-path PATH]"
                ),
                "returns": "Numbered source lines grouped by PATH:START-END",
            },
            {
                "name": "chunks",
                "purpose": "List indexed definitions in one file.",
                "usage": ("smritikosh explore chunks PATH [--db-path PATH]"),
                "returns": "TOON outline rows with symbols and line ranges",
            },
        ],
        "search_standards": [
            {
                "parameter": "QUERY count",
                "start_with": "4",
                "guidance": "Use one distinct facet per query; never repeat synonyms.",
            },
            {
                "parameter": "QUERY facets",
                "start_with": "flow|state|failure|tests",
                "guidance": (
                    "Include exact domain nouns, events, or symbols when known."
                ),
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
        ],
        "agent_guidance": [
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
        ],
        "rules": [
            {
                "rule": (
                    "Use --range PATH START END; do not also pass positional PATH, "
                    "--start-line, --end-line, or --full."
                )
            },
            {"rule": "Search returns locations only; use chunks to read source."},
            {"rule": "TOON is the default; --toon is accepted for compatibility."},
        ],
    }


def _echo_tools_prose() -> None:
    """Print the exploration workflow for a human reader."""
    click.echo("1. Search with 4-8 focused facets of one question:")
    click.echo(
        "   smritikosh explore search QUERY... --max-results 12 [--db-path PATH]"
    )
    click.echo("2. Read selected ranges (repeat --range to batch reads):")
    click.echo(
        "   smritikosh explore chunks --range PATH START END "
        "[--range PATH START END ...] [--db-path PATH]"
    )
    click.echo("3. Cite the returned PATH:START-END evidence.")
    click.echo()
    click.echo("Agent guidance:")
    for instruction in _tools_payload()["agent_guidance"]:
        click.echo(f"- {instruction}")
    click.echo()
    click.echo("Do not combine --range with positional PATH, line options, or --full.")
    click.echo("TOON is the default output format.")


@click.group()
def explore() -> None:
    """Explore an existing index through read-only agent-friendly commands."""


@explore.command("tools")
@click.option(
    "--prose",
    is_flag=True,
    help="Print prose for a human instead of the default TOON rows.",
)
@click.option(
    "--toon",
    is_flag=True,
    hidden=True,
)
def list_tools(prose: bool, toon: bool) -> None:  # noqa: ARG001
    """List commands, exact syntax, workflow, and composition rules."""
    if prose:
        _echo_tools_prose()
        return
    _echo_toon(_tools_payload())


@explore.command("search")
@click.argument("queries", nargs=-1, required=True)
@click.option(
    "--max-results",
    "--max_results",
    default=24,
    show_default=True,
    type=click.IntRange(min=1, max=24, clamp=True),
)
@click.option(
    "--db-path",
    "--db_path",
    default=DEFAULT_DB_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--prose",
    is_flag=True,
    help="Print prose for a human instead of the default TOON rows.",
)
@click.option(
    # Accepted and ignored: TOON is the default now, but these names are
    # written into agent prompts that predate it, and failing those with
    # "no such option" costs a whole turn to rediscover an unchanged command.
    "--toon",
    "--json-output",
    "--json_output",
    is_flag=True,
    hidden=True,
)
def hybrid_search(
    queries: tuple[str, ...],
    max_results: int,
    db_path: str,
    prose: bool,
    toon: bool,  # noqa: ARG001
) -> None:
    """Search one question's facets; pass 4-8 focused QUERY values.

    Dense and BM25 candidates are fused with Reciprocal Rank Fusion, reserved
    for facet coverage, selected for diversity, and expanded to complete
    definitions plus their direct references. Only locations are returned:
    read the ones worth reading with `explore chunks --range PATH START END`.
    """
    # Repeated facets would retrieve the same candidates twice and label one
    # row with the same id twice, so they collapse before any work is done.
    queries = tuple(
        dict.fromkeys(
            query.strip()[:MAX_QUERY_CHARS] for query in queries if query.strip()
        )
    )[:MAX_QUERIES]
    if not queries:
        raise click.UsageError("Provide at least one non-empty QUERY")
    embedder: Embedder = make_embedder()
    reader = DuckDBSourceReader(db_path)
    try:
        lexical_store = DuckDBBm25Store(db_path, read_only=True)
        try:
            service = HybridSearchService(
                (
                    DuckDBDenseRetriever(reader, embedder),
                    DuckDBLexicalRetriever(lexical_store, reader),
                ),
                reader,
            )
            try:
                locations: list[SearchLocation] = service.search(
                    queries,
                    options=HybridSearchOptions(max_results=max_results),
                )
            except RuntimeError as exc:
                raise click.ClickException(str(exc)) from exc
        finally:
            lexical_store.close()
    finally:
        reader.close()
    if prose:
        _echo_search_prose(locations, queries=queries)
        return
    _echo_toon(_search_payload(locations, queries=queries))


@explore.command("chunks")
@click.argument("path", required=False)
@click.option("--start-line", "--start_line", type=click.IntRange(min=1))
@click.option("--end-line", "--end_line", type=click.IntRange(min=1))
@click.option(
    "--range",
    "ranges",
    nargs=3,
    multiple=True,
    type=(str, click.IntRange(min=1), click.IntRange(min=1)),
    help="Read PATH START END; repeat to retrieve several ranges in one process.",
)
@click.option(
    "--full",
    is_flag=True,
    help="Print every stored chunk in full instead of an outline.",
)
@click.option(
    "--db-path",
    "--db_path",
    default=DEFAULT_DB_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--prose",
    is_flag=True,
    help="Print prose for a human instead of the default TOON rows.",
)
@click.option(
    # Accepted and ignored: TOON is the default now, but these names are
    # written into agent prompts that predate it, and failing those with
    # "no such option" costs a whole turn to rediscover an unchanged command.
    "--toon",
    "--json-output",
    "--json_output",
    is_flag=True,
    hidden=True,
)
def get_chunks(
    path: str | None,
    start_line: int | None,
    end_line: int | None,
    ranges: tuple[tuple[str, int, int], ...],
    full: bool,
    db_path: str,
    prose: bool,
    toon: bool,  # noqa: ARG001
) -> None:
    """Retrieve indexed code from an exact PATH.

    With a line range, the requested source is rebuilt once and printed with
    line numbers. Without one, the path's definitions are listed as an
    outline; pass --full to print every stored chunk instead.

    Source lines always print as numbered text. TOON must quote any value
    holding a colon, and code is full of them, so a table of lines costs
    more than the numbered text it would replace and reads worse.
    """
    if ranges:
        if path is not None or start_line is not None or end_line is not None or full:
            raise click.UsageError(
                "--range cannot be combined with PATH, line options, or --full"
            )
        with ReadOnlyExplorer(db_path) as explorer:
            for index, (range_path, range_start, range_end) in enumerate(ranges):
                if range_start > range_end:
                    raise click.BadParameter(
                        "START cannot be greater than END",
                        param_hint="--range",
                    )
                lines: list[SourceLine] = explorer.get_source_lines(
                    range_path,
                    start_line=range_start,
                    end_line=range_end,
                )
                if index:
                    click.echo()
                _echo_source_lines(lines, path=range_path)
        return
    if path is None:
        raise click.UsageError("Provide PATH or at least one --range PATH START END")
    ranged: bool = start_line is not None or end_line is not None
    try:
        with ReadOnlyExplorer(db_path) as explorer:
            if ranged:
                lines: list[SourceLine] = explorer.get_source_lines(
                    path,
                    start_line=start_line,
                    end_line=end_line,
                )
            elif full:
                chunks: list[IndexedChunk] = explorer.get_chunks(path)
            else:
                entries: list[OutlineEntry] = explorer.get_outline(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if ranged:
        _echo_source_lines(lines, path=path)
        return
    if full:
        if not prose:
            _echo_toon([asdict(chunk) for chunk in chunks])
            return
        _echo_chunks(chunks)
        return
    if not prose:
        _echo_toon([asdict(entry) for entry in entries])
        return
    _echo_outline(entries, path=path)
