"""Generic bounded definition and dependency expansion."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Final

from smritikosh.models import SearchResult
from smritikosh.models.retrieval import (
    CandidateScore,
    HybridSearchOptions,
    IndexedChunk,
    OutlineEntry,
    RankedCandidate,
    SourceLine,
    TextMatch,
)
from smritikosh.ports.source_reader import SourceReader
from smritikosh.retrieval.tokenizer import tokenize_code

__all__ = ["CandidateExpander"]

_DIRECT_CALL: Final[re.Pattern[str]] = re.compile(
    r"(?:@|(?<![\w.]))([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_QUALIFIED_CALL: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_IGNORED_CALLS: Final[frozenset[str]] = frozenset(
    {
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "set",
        "str",
        "super",
        "tuple",
    }
)


class CandidateExpander:
    """Expand selected seeds without repository- or domain-specific rules."""

    def __init__(self, reader: SourceReader) -> None:
        self._reader = reader
        self._definition_cache: dict[str, list[SearchResult]] = {}
        self._outline_cache: dict[str, list[OutlineEntry]] = {}

    def expand(
        self,
        seeds: list[RankedCandidate],
        *,
        options: HybridSearchOptions,
    ) -> list[RankedCandidate]:
        """Expand definitions and one dependency round, preserving seed priority."""
        parents: list[RankedCandidate] = [
            replace(seed, result=self._complete_definition(seed.result))
            for seed in seeds
        ]
        expanded: list[RankedCandidate] = []
        known: set[tuple[str, str, str | None]] = set()
        for parent in parents:
            self._append_unique(expanded, known, parent)
        groups: list[list[RankedCandidate]] = [
            self._dependencies(
                parent,
                limit=options.max_dependencies_per_seed,
            )
            for parent in parents
        ]
        for index in range(options.max_dependencies_per_seed):
            for dependencies in groups:
                if index < len(dependencies):
                    self._append_unique(expanded, known, dependencies[index])
        return expanded[: options.max_results]

    def _dependencies(
        self,
        parent: RankedCandidate,
        *,
        limit: int,
    ) -> list[RankedCandidate]:
        result: SearchResult = parent.result
        lines: list[SourceLine] = self._reader.get_source_lines(
            result.path,
            start_line=result.start_line,
            end_line=min(result.end_line, result.start_line + 119),
        )
        dependencies: list[tuple[int, int, RankedCandidate]] = []
        facet_roots: set[str] = {
            token[:5]
            for facet in parent.facets
            for token in tokenize_code(facet)
            if len(token) >= 3
        }
        for position, name in enumerate(
            self._call_names(lines, current_symbol=result.symbol)[:20]
        ):
            for definition in self._find_definitions(name, parent_path=result.path):
                definition = self._complete_definition(definition)
                result_roots: set[str] = {
                    token[:5]
                    for token in tokenize_code(
                        " ".join(
                            (
                                definition.path,
                                definition.symbol or "",
                            )
                        )
                    )
                    if len(token) >= 3
                }
                dependencies.append(
                    (
                        len(facet_roots & result_roots),
                        -position,
                        RankedCandidate(
                            result=definition,
                            facets=set(parent.facets),
                            facet_scores=dict(parent.facet_scores),
                            score=CandidateScore(fused=parent.score.fused * 0.9),
                        ),
                    )
                )
        dependencies.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2].result.path,
            )
        )
        return [candidate for _, _, candidate in dependencies[:limit]]

    def _find_definitions(
        self,
        name: str,
        *,
        parent_path: str,
    ) -> list[SearchResult]:
        if name not in self._definition_cache:
            matches: list[TextMatch] = self._reader.find_text_lines(name, limit=100)
            found: list[SearchResult] = []
            for match in matches:
                for chunk in self._reader.get_chunks(
                    match.path,
                    start_line=match.line_number,
                    end_line=match.line_number,
                ):
                    if chunk.symbol == name and chunk.chunk_kind in {
                        "function",
                        "method",
                    }:
                        found.append(self._result_from_chunk(chunk))
            self._definition_cache[name] = found
        definitions: list[SearchResult] = list(self._definition_cache[name])
        definitions.sort(
            key=lambda result: (
                self._path_distance(parent_path, result.path),
                result.path.startswith(("tests/", ".")),
                result.path,
                result.start_line,
            )
        )
        same_file: list[SearchResult] = [
            result for result in definitions if result.path == parent_path
        ]
        if same_file:
            return same_file[:2]
        return definitions[:2]

    def _complete_definition(self, result: SearchResult) -> SearchResult:
        if result.symbol is None or result.chunk_kind not in {
            "class",
            "function",
            "method",
        }:
            return result
        if result.path not in self._outline_cache:
            self._outline_cache[result.path] = self._reader.get_outline(result.path)
        entries: list[OutlineEntry] = [
            entry
            for entry in self._outline_cache[result.path]
            if entry.symbol == result.symbol
            and entry.chunk_kind == result.chunk_kind
            and entry.start_line <= result.end_line
            and entry.end_line >= result.start_line
        ]
        if not entries:
            return result
        return replace(
            result,
            start_line=entries[0].start_line,
            end_line=entries[0].end_line,
        )

    @staticmethod
    def _append_unique(
        candidates: list[RankedCandidate],
        known: set[tuple[str, str, str | None]],
        candidate: RankedCandidate,
    ) -> None:
        result: SearchResult = candidate.result
        identity: str = result.symbol or f"@{result.start_line}:{result.end_line}"
        key: tuple[str, str, str | None] = (
            result.path,
            identity,
            result.chunk_kind,
        )
        if key not in known:
            known.add(key)
            candidates.append(candidate)

    @staticmethod
    def _call_names(
        lines: list[SourceLine],
        *,
        current_symbol: str | None,
    ) -> list[str]:
        source: str = "\n".join(
            line.text
            for line in lines
            if not line.text.lstrip().startswith(("def ", "async def ", "class "))
        )
        found: list[str] = _DIRECT_CALL.findall(source)
        found.extend(_QUALIFIED_CALL.findall(source))
        names: list[str] = []
        for name in found:
            if (
                name in _IGNORED_CALLS
                or name == current_symbol
                or name[:1].isupper()
                or name.startswith("test_")
                or name in names
            ):
                continue
            names.append(name)
        return names

    @staticmethod
    def _path_distance(left: str, right: str) -> int:
        left_parts: list[str] = left.split("/")[:-1]
        right_parts: list[str] = right.split("/")[:-1]
        shared: int = 0
        for left_part, right_part in zip(left_parts, right_parts, strict=False):
            if left_part != right_part:
                break
            shared += 1
        return len(left_parts) + len(right_parts) - (2 * shared)

    @staticmethod
    def _result_from_chunk(chunk: IndexedChunk) -> SearchResult:
        return SearchResult(
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            snippet=chunk.text,
            score=0.0,
            chunk_kind=chunk.chunk_kind,
            symbol=chunk.symbol,
            chunk_id=chunk.chunk_id,
        )
