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
        return [
            build_chunk(parsed_file.path, parsed_file.content.rstrip("\n"), "own", 1, 1)
        ]


def test_should_delegate_chunking_entirely_to_strategy() -> None:
    p = parsed("def foo(): pass\n")
    captures = [cap("definition.function", node_for(p.content, "def foo(): pass"))]
    strategy = RecordingStrategy()

    chunks = chunk_file(p, captures, strategy)

    assert strategy.calls == [(p, captures)]
    assert chunks == [build_chunk(p.path, "def foo(): pass", "own", 1, 1)]


def test_should_register_chunk_and_strategy_logic_for_memo_invalidation() -> None:
    assert hasattr(chunk_file, "_logic_fingerprint")
    assert hasattr(AstChunkingStrategy.chunk, "_logic_fingerprint")
    assert hasattr(SectionChunkingStrategy.chunk, "_logic_fingerprint")
    assert hasattr(RegexChunkingStrategy.chunk, "_logic_fingerprint")
    assert hasattr(build_chunk, "_logic_fingerprint")
    assert hasattr(make_group_chunks, "_logic_fingerprint")
    assert {
        chunk_file._logic_fingerprint,  # type: ignore[attr-defined]
        AstChunkingStrategy.chunk._logic_fingerprint,  # type: ignore[attr-defined]
        SectionChunkingStrategy.chunk._logic_fingerprint,  # type: ignore[attr-defined]
        RegexChunkingStrategy.chunk._logic_fingerprint,  # type: ignore[attr-defined]
        build_chunk._logic_fingerprint,  # type: ignore[attr-defined]
        make_group_chunks._logic_fingerprint,  # type: ignore[attr-defined]
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


def test_should_keep_blank_lines_between_grouped_definitions() -> None:
    content = "LOW = 1\n\nHIGH = 9\n"
    p = parsed(content)

    (chunk,) = make_group_chunks(
        p, [node_for(content, "LOW = 1"), node_for(content, "HIGH = 9")], "constant"
    )

    assert chunk.text == "LOW = 1\n\nHIGH = 9"
    assert (chunk.start_line, chunk.end_line) == (1, 3)


def test_should_start_a_new_group_chunk_where_other_code_sits_between() -> None:
    content = (
        "class Limits:\n    LOW = 1\n\n    def check(self): pass\n\n    HIGH = 9\n"
    )
    p = parsed(content)
    low = node_for(content, "LOW = 1")
    high = node_for(content, "HIGH = 9")

    chunks = make_group_chunks(p, [low, high], "class_constant")

    assert [(c.text, c.start_line, c.end_line) for c in chunks] == [
        ("    LOW = 1", 2, 2),
        ("    HIGH = 9", 6, 6),
    ]


def test_should_emit_a_shared_line_once_when_two_definitions_sit_on_it() -> None:
    content = "A = 1; B = 2\n"
    p = parsed(content)

    (chunk,) = make_group_chunks(
        p, [node_for(content, "A = 1"), node_for(content, "B = 2")], "constant"
    )

    assert chunk.text == "A = 1; B = 2"
    assert (chunk.start_line, chunk.end_line) == (1, 1)


def test_should_make_every_group_chunk_an_exact_slice_of_the_file() -> None:
    content = "A = 1\n\nB = 2\n\ndef f(): pass\n\nC = 3\n\nD = 4\n"
    p = parsed(content)
    nodes = [node_for(content, text) for text in ("A = 1", "B = 2", "C = 3", "D = 4")]
    source_lines: list[str] = content.split("\n")

    chunks = make_group_chunks(p, nodes, "constant", max_chars=8)

    assert len(chunks) > 2
    for chunk in chunks:
        assert (
            chunk.text.split("\n")
            == (source_lines[chunk.start_line - 1 : chunk.end_line])
        )


def test_should_record_exact_source_lines_when_chunk_text_differs() -> None:
    content = "# Guide\n\nRun the tests.\n"
    p = parsed(content, "doc.md", "markdown")
    heading = cap(
        "definition.section", node_for(content, "# Guide"), path="doc.md", key="Guide"
    )

    (chunk,) = chunk_file(p, [heading], SectionChunkingStrategy())

    assert chunk.text.startswith("# Guide")
    assert chunk.source_text == "\n".join(
        content.split("\n")[chunk.start_line - 1 : chunk.end_line]
    )


def test_should_read_markdown_headings_back_from_their_chunks() -> None:
    content = "# Guide\n\nIntro.\n\n## Setup\n\nInstall it.\n\n## Usage\n\nRun it.\n"
    p = parsed(content, "doc.md", "markdown")
    headings = [
        cap("definition.section", node_for(content, f"{marks} {name}"), key=name)
        for marks, name in (("#", "Guide"), ("##", "Setup"), ("##", "Usage"))
    ]

    chunks = chunk_file(p, headings, SectionChunkingStrategy())

    rebuilt: dict[int, str] = {}
    for chunk in chunks:
        assert chunk.source_text is not None
        for offset, line in enumerate(chunk.source_text.split("\n")):
            rebuilt[chunk.start_line + offset] = line
    assert [rebuilt[number] for number in sorted(rebuilt)] == content.splitlines()


def test_should_record_no_source_lines_when_chunk_text_is_the_source() -> None:
    p = parsed("def foo(): pass\n")
    captures = [cap("definition.function", node_for(p.content, "def foo(): pass"))]

    (chunk,) = chunk_file(p, captures, AstChunkingStrategy())

    assert chunk.source_text is None
