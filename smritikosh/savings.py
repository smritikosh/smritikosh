"""Count source lines a search cited against the files those lines came from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb

from smritikosh.constants import INDEXED_ROOTS_KEY
from smritikosh.models import ExplainedHit, SearchLocation

__all__ = [
    "SavingsTotal",
    "SearchSavings",
    "format_explain_hit",
    "format_savings_line",
    "format_savings_total",
    "measure_search",
    "read_indexed_roots",
    "record_search_savings",
    "remember_indexed_root",
    "savings_total",
]


@dataclass(frozen=True)
class SearchSavings:
    """One search: ranges returned, lines inside them, and lines in those files."""

    ranges: int
    lines_cited: int
    lines_in_files: int

    @property
    def lines_not_opened(self) -> int:
        return max(self.lines_in_files - self.lines_cited, 0)


@dataclass(frozen=True)
class SavingsTotal:
    """Running total since the first recorded search."""

    since: date | None
    searches: int
    lines_cited: int
    lines_in_files: int

    @property
    def lines_not_opened(self) -> int:
        return max(self.lines_in_files - self.lines_cited, 0)


def remember_indexed_root(con: duckdb.DuckDBPyConnection, repo_path: str) -> None:
    """Remember *repo_path* so later searches can count lines in its files."""
    root = str(Path(repo_path).resolve())
    roots = read_indexed_roots(con)
    if root in roots:
        return
    roots.append(root)
    con.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        [INDEXED_ROOTS_KEY, json.dumps(roots)],
    )


def read_indexed_roots(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return repo roots stored in ``kv_store``, or an empty list."""
    try:
        row = con.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            [INDEXED_ROOTS_KEY],
        ).fetchone()
    except duckdb.Error:
        return []
    if row is None:
        return []
    loaded = json.loads(row[0])
    return [str(item) for item in loaded] if isinstance(loaded, list) else []


def measure_search(
    locations: list[SearchLocation],
    roots: list[Path],
) -> SearchSavings:
    """Count cited lines and the full length of each file those lines came from.

    Each returned range contributes ``end_line - start_line + 1``. Each file
    is counted once. When the file is not on disk, its length is the last
    cited line, so unread lines are not invented.
    """
    cited = 0
    per_file: dict[str, int] = {}
    for location in locations:
        cited += location.end_line - location.start_line + 1
        per_file[location.path] = max(
            per_file.get(location.path, 0),
            location.end_line,
        )
    file_lines = 0
    for path, lower_bound in per_file.items():
        counted = _lines_in_file(path, roots)
        file_lines += counted if counted is not None else lower_bound
    return SearchSavings(
        ranges=len(locations),
        lines_cited=cited,
        lines_in_files=file_lines,
    )


def format_savings_line(savings: SearchSavings) -> str:
    """One summary line for a single search."""
    return (
        f"{savings.ranges} ranges, {savings.lines_cited} lines cited, "
        f"{savings.lines_in_files} lines in those files, "
        f"{savings.lines_not_opened} lines not opened"
    )


def format_explain_hit(hit: ExplainedHit, labels: dict[str, str]) -> str:
    """One plain-text line: channel ranks, facet, and any widened symbol span."""
    location = hit.location
    symbol = f"  {location.symbol}" if location.symbol else ""
    channels: list[str] = []
    if hit.dense_rank is not None:
        channels.append(f"dense rank {hit.dense_rank}")
    if hit.lexical_rank is not None:
        channels.append(f"lexical rank {hit.lexical_rank}")
    channel = ", ".join(channels) if channels else "unranked"
    widened = ""
    cited = (location.start_line, location.end_line)
    chunk = (hit.chunk_start_line, hit.chunk_end_line)
    if chunk != cited:
        widened = (
            f", widened {hit.chunk_start_line}-{hit.chunk_end_line}"
            f" -> {location.start_line}-{location.end_line}"
        )
    facets = "|".join(labels[facet] for facet in location.facets if facet in labels)
    return (
        f"{location.path}:{location.start_line}-{location.end_line}"
        f"{symbol}  {channel}{widened}  {facets}"
    )


def record_search_savings(db_path: str, savings: SearchSavings) -> None:
    """Append one row to ``search_savings``. A locked or unreadable file is skipped."""
    try:
        con = duckdb.connect(db_path)
    except duckdb.Error:
        return
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS search_savings (
                searched_at TIMESTAMP DEFAULT now(),
                ranges INTEGER NOT NULL,
                lines_cited INTEGER NOT NULL,
                lines_in_files INTEGER NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO search_savings (ranges, lines_cited, lines_in_files)
            VALUES (?, ?, ?)
            """,
            [savings.ranges, savings.lines_cited, savings.lines_in_files],
        )
    except duckdb.Error:
        return
    finally:
        con.close()


def savings_total(db_path: str) -> SavingsTotal:
    """Sum every recorded search. Missing index or table means nothing counted yet."""
    path = Path(db_path)
    if not path.is_file():
        return SavingsTotal(None, 0, 0, 0)
    try:
        con = duckdb.connect(db_path, read_only=True)
    except duckdb.Error:
        return SavingsTotal(None, 0, 0, 0)
    try:
        tables = con.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'search_savings'
            """
        ).fetchone()
        if tables is None:
            return SavingsTotal(None, 0, 0, 0)
        row = con.execute(
            """
            SELECT min(searched_at), count(*),
                   coalesce(sum(lines_cited), 0),
                   coalesce(sum(lines_in_files), 0)
            FROM search_savings
            """
        ).fetchone()
    except duckdb.Error:
        return SavingsTotal(None, 0, 0, 0)
    finally:
        con.close()
    if row is None or row[1] == 0:
        return SavingsTotal(None, 0, 0, 0)
    return SavingsTotal(
        since=_as_date(row[0]),
        searches=int(row[1]),
        lines_cited=int(row[2]),
        lines_in_files=int(row[3]),
    )


def format_savings_total(total: SavingsTotal) -> str:
    """Plain-text running total. No ``since`` line until a search has been recorded."""
    lines = []
    if total.since is not None:
        lines.append(f"since {total.since.isoformat()}")
    lines.extend(
        [
            f"searches {total.searches}",
            f"lines cited {total.lines_cited}",
            f"lines in those files {total.lines_in_files}",
            f"lines not opened {total.lines_not_opened}",
        ]
    )
    return "\n".join(lines)


def _lines_in_file(path: str, roots: list[Path]) -> int | None:
    candidates = [Path(path)]
    candidates.extend(root / path for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return len(candidate.read_text(errors="replace").splitlines())
    return None


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value[:10])
    return None
