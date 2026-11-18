"""Regex-based chunking for config files (TOML). Tree-sitter captures are ignored."""

from __future__ import annotations

import re

from smritikosh.engine import sm
from smritikosh.indexing.strategies._helpers import make_text_chunks
from smritikosh.models import Capture, Chunk, ParsedFile

__all__ = ["RegexChunkingStrategy"]


class RegexChunkingStrategy:
    """Split content at regex pattern boundaries; whole-file fallback.

    *max_chars* additionally caps each section at what the embedder reads, so
    a config with one enormous table does not lose its tail.
    """

    mode_name = "regex"

    def __init__(self, split_pattern: str, max_chars: int | None = None) -> None:
        self._pattern = re.compile(split_pattern, re.MULTILINE)
        self.max_chars = max_chars

    @sm.tracked
    def chunk(self, parsed: ParsedFile, captures: list[Capture]) -> list[Chunk]:  # noqa: ARG002
        content = parsed.content
        matches = list(self._pattern.finditer(content))

        if not matches:
            text = content.rstrip("\n")
            return make_text_chunks(
                parsed, text, "regex", 1, text.count("\n") + 1, self.max_chars
            )

        chunks: list[Chunk] = []
        boundaries = [m.start() for m in matches] + [len(content)]

        for i, start_pos in enumerate(boundaries[:-1]):
            end_pos = boundaries[i + 1]
            section = content[start_pos:end_pos].rstrip("\n")
            if not section.strip():
                continue
            start_line = content[:start_pos].count("\n") + 1
            # Counted from the section itself, whose trailing blank lines were
            # stripped, so the range ends on the section's last line.
            end_line = start_line + section.count("\n")
            # The boundary the split matched is the section's own header —
            # "[tool.ruff]" in a TOML file — which is the nearest thing a
            # regex-chunked file has to a symbol name.
            header = section.lstrip().splitlines()[0].strip()
            chunks.extend(
                make_text_chunks(
                    parsed,
                    section,
                    "regex",
                    start_line,
                    end_line,
                    self.max_chars,
                    header or None,
                )
            )

        return chunks
