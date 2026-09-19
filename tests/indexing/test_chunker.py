"""Tests for chunk orchestration (chunk_file) and content-addressed chunk ids."""

from __future__ import annotations

import hashlib

from smritikosh.engine._fingerprint import _tracked_logic_fps
from smritikosh.indexing.chunker import chunk_file
from smritikosh.indexing.strategies._helpers import (
    build_chunk,
    chunk_id,
    content_hash,
    make_group_chunks,
    make_node_chunks,
    make_text_chunks,
)
from smritikosh.indexing.strategies.ast import (
    CHUNK_GROUPING,
    AstChunkingStrategy,
    _apply_grouping,
)
from smritikosh.indexing.strategies.regex import RegexChunkingStrategy
from smritikosh.indexing.strategies.section import SectionChunkingStrategy
from smritikosh.models import Capture, Chunk, ParsedFile
from tests.indexing.conftest import cap, cid, node_for, parsed


class RecordingStrategy:
    mode_name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[ParsedFile, list[Capture]]] = []

    def chunk(self, parsed_file: ParsedFile, captures: list[Capture]) -> list[Chunk]:
        self.calls.append((parsed_file, captures))
        return [build_chunk(parsed_file.path, "delegated", "own", 1, 1)]


def test_should_delegate_chunking_entirely_to_strategy() -> None:
    p = parsed("def foo(): pass\n")
    captures = [cap("definition.function", node_for(p.content, "def foo(): pass"))]
    strategy = RecordingStrategy()

    chunks = chunk_file(p, captures, strategy)

    assert strategy.calls == [(p, captures)]
    assert chunks == [build_chunk(p.path, "delegated", "own", 1, 1)]


def test_should_register_chunk_and_strategy_logic_for_memo_invalidation() -> None:
    assert hasattr(chunk_file, "_logic_fingerprint")
    assert hasattr(AstChunkingStrategy.chunk, "_logic_fingerprint")
    assert hasattr(SectionChunkingStrategy.chunk, "_logic_fingerprint")
    assert hasattr(RegexChunkingStrategy.chunk, "_logic_fingerprint")
    assert hasattr(build_chunk, "_logic_fingerprint")
    assert {
        chunk_file._logic_fingerprint,  # type: ignore[attr-defined]
        AstChunkingStrategy.chunk._logic_fingerprint,  # type: ignore[attr-defined]
        SectionChunkingStrategy.chunk._logic_fingerprint,  # type: ignore[attr-defined]
        RegexChunkingStrategy.chunk._logic_fingerprint,  # type: ignore[attr-defined]
        build_chunk._logic_fingerprint,  # type: ignore[attr-defined]
    } <= _tracked_logic_fps


def test_should_build_stable_ids_from_path_lines_and_text() -> None:
    """Pins the id formula itself — every other test reaches it through ``cid``."""
    text = "def foo(): pass"
    identity = "\0".join(["a.py", "1:1", text])
    expected_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    expected_hash = hashlib.sha256(text.encode()).hexdigest()

    first = build_chunk("a.py", text, "function", 1, 1)
    second = build_chunk("a.py", text, "function", 1, 1)

    assert first.id == second.id == expected_id == chunk_id("a.py", text, 1, 1)
    assert first.content_hash == expected_hash == content_hash(text)


def test_should_keep_identical_text_in_different_files_distinct() -> None:
    text = "def foo(): pass"

    first = build_chunk("a.py", text, "function", 1, 1)
    second = build_chunk("b.py", text, "function", 1, 1)

    assert first.id != second.id
    assert first.content_hash == second.content_hash


def test_should_not_duplicate_init_text_when_extending_class_span() -> None:
    content = "class Foo:\n    def __init__(self): pass\n"
    p = parsed(content)
    class_node = node_for(content, content.rstrip("\n"))
    init_node = node_for(content, "def __init__(self): pass")

    chunks = _apply_grouping(
        p,
        [
            cap("definition.class", class_node),
            cap("definition.class_init", init_node),
        ],
        CHUNK_GROUPING,
    )

    assert len(chunks) == 1
    assert chunks[0].text.count("def __init__") == 1
    assert chunks[0].chunk_kind == "class"


def test_should_make_node_group_and_text_chunks_via_build_chunk() -> None:
    content = "X = 1\nY = 2\n"
    p = parsed(content)
    n1 = node_for(content, "X = 1")
    n2 = node_for(content, "Y = 2")

    (node_chunk,) = make_node_chunks(p, n1, "constant")
    (group_chunk,) = make_group_chunks(p, [n1, n2], "constant")
    (text_chunk,) = make_text_chunks(p, "hello", "regex", 1, 1)

    assert node_chunk.id == cid("X = 1")
    assert group_chunk.id == cid("X = 1\nY = 2", start_line=1, end_line=2)
    assert text_chunk.id == cid("hello")
    assert node_chunk.start_line == 1
    assert group_chunk.end_line >= group_chunk.start_line
