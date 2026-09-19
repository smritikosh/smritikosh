"""Shared test doubles for the indexing tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from smritikosh.indexing.strategies._helpers import chunk_id
from smritikosh.models import Capture, Chunk, ParsedFile

# ── FakeStrategy ───────────────────────────────────────────────────────────


class FakeStrategy:
    """Stand-in for a chunking strategy. Routing and discovery never chunk."""

    mode_name = "fake"

    def chunk(self, parsed: ParsedFile, captures: list[Capture]) -> list[Chunk]:  # noqa: ARG002
        return []


# ── FakeNode ──────────────────────────────────────────────────────────────

_id_counter = 0


def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


@dataclass
class FakeNode:
    """Minimal tree-sitter node double for unit tests."""

    start_byte: int
    end_byte: int
    start_point: tuple[int, int]  # (row, col) 0-indexed
    end_point: tuple[int, int]
    id: int = field(default_factory=_next_id)


# ── Builder helpers ────────────────────────────────────────────────────────


def node_for(content: str, text: str, node_id: int | None = None) -> FakeNode:
    """Return a FakeNode whose byte range exactly covers *text* in *content*."""
    enc = content.encode()
    text_enc = text.encode()
    pos = enc.find(text_enc)
    assert pos != -1, f"{text!r} not found in content"
    start_row = content[:pos].count("\n")
    end_row = content[: pos + len(text)].count("\n")
    n = FakeNode(
        start_byte=pos,
        end_byte=pos + len(text_enc),
        start_point=(start_row, 0),
        end_point=(end_row, 0),
    )
    if node_id is not None:
        n.id = node_id
    return n


def parsed(content: str, path: str = "test.py", language: str = "python") -> ParsedFile:
    """Return a ParsedFile wrapping *content*."""
    return ParsedFile(path=path, language=language, content=content, tree=None)


def cap(name: str, node: FakeNode, path: str = "test.py", key: str = "") -> Capture:
    """Return a Capture for *node* with the given capture *name*.

    *key* populates ``Capture.name`` — the captured symbol/key text, which the
    JSON path builder reads.
    """
    return Capture(capture_name=name, node=node, name=key, path=path)


def cid(
    text: str,
    path: str = "test.py",
    start_line: int = 1,
    end_line: int = 1,
) -> str:
    """Return the expected chunk id for *text* at the given location.

    Defers to the production formula rather than restating it, so the two
    cannot drift apart.  The formula itself is pinned literally in
    ``test_should_build_stable_ids_from_path_lines_and_text``.
    """
    return chunk_id(path, text, start_line, end_line)
