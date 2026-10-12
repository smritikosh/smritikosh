"""Tests for the embedder-driven chunk size budget.

An encoder silently drops tokens past its context window, so a chunk larger
than that window is stored whole but embedded from its prefix alone — the tail
becomes unsearchable with no error anywhere.  These tests pin the split that
prevents it.
"""

from __future__ import annotations

import pytest

from smritikosh.indexing.strategies._helpers import (
    _CHARS_PER_TOKEN,
    budget_chars,
    chunk_id,
    content_hash,
    split_oversized,
)
from smritikosh.indexing.strategies.ast import AstChunkingStrategy
from smritikosh.indexing.strategies.regex import RegexChunkingStrategy
from smritikosh.indexing.strategies.section import SectionChunkingStrategy
from smritikosh.ports.embedder import Embedder
from tests.indexing.conftest import cap, node_for, parsed

# ── budget_chars ──────────────────────────────────────────────────────────────


def test_should_convert_token_window_to_a_char_budget() -> None:
    assert budget_chars(512) == 1147


def test_should_stay_under_the_naive_ratio() -> None:
    """The budget must reserve headroom, not spend the whole window.

    Special tokens and a safety factor both come off the top; a budget equal
    to ``max_tokens * ratio`` would land over the limit on dense content.
    """
    assert budget_chars(512) < int(512 * _CHARS_PER_TOKEN)


def test_should_scale_with_a_larger_context_window() -> None:
    assert budget_chars(8192) > budget_chars(512) * 10


def test_should_stay_positive_for_a_tiny_window() -> None:
    assert budget_chars(1) >= 1


def test_should_stay_conservative_against_dense_code() -> None:
    """The ratio must under-fill the window, not over-fill it.

    Measured minimum on real chunks was 1.39 chars/token; a ratio above that
    would let punctuation-dense code overflow and truncate again.
    """
    assert _CHARS_PER_TOKEN <= 2.5


# ── split_oversized ───────────────────────────────────────────────────────────


def test_should_return_a_single_window_when_under_budget() -> None:
    text = "a = 1\nb = 2"
    assert split_oversized(text, max_chars=1000) == [(text, 0, 1)]


def test_should_keep_every_window_within_budget() -> None:
    text = "\n".join(f"CONST_{i} = {i}" for i in range(200))

    windows = split_oversized(text, max_chars=100)

    assert len(windows) > 1
    assert all(len(piece) <= 100 for piece, _, _ in windows)


def test_should_never_break_a_line_across_windows() -> None:
    lines = [f"CONST_{i} = {i}" for i in range(120)]

    windows = split_oversized("\n".join(lines), max_chars=90)

    emitted = {line for piece, _, _ in windows for line in piece.split("\n")}
    assert emitted == set(lines)


def test_should_cover_every_line_with_no_gaps() -> None:
    lines = [f"line_{i}" for i in range(60)]

    windows = split_oversized("\n".join(lines), max_chars=70)

    covered = {i for _, first, last in windows for i in range(first, last + 1)}
    assert covered == set(range(len(lines)))


def test_should_overlap_consecutive_windows() -> None:
    """A definition on a seam must be reachable from either side."""
    lines = [f"CONST_{i} = {i}" for i in range(80)]

    windows = split_oversized("\n".join(lines), max_chars=100, overlap_lines=2)

    # Each window after the first restarts before the previous one ended.
    for (_, _, prev_last), (_, next_first, _) in zip(
        windows, windows[1:], strict=False
    ):
        assert next_first <= prev_last


def test_should_emit_an_overlong_line_rather_than_cutting_it() -> None:
    """One line over budget is a minified blob; cutting it corrupts the snippet."""
    text = "x" * 500

    windows = split_oversized(text, max_chars=100)

    assert [piece for piece, _, _ in windows] == [text]


def test_should_terminate_when_budget_is_smaller_than_the_overlap() -> None:
    lines = [f"line_{i}" for i in range(20)]

    windows = split_oversized("\n".join(lines), max_chars=10, overlap_lines=5)

    assert len(windows) >= len(lines) - 1  # progressed rather than looping


# ── strategy plumbing ─────────────────────────────────────────────────────────

_MANY_CONSTANTS = "\n".join(f"CONST_{i}: int = {i}" for i in range(300))


def test_ast_strategy_should_split_an_oversized_constant_group() -> None:
    content = _MANY_CONSTANTS
    p = parsed(content)
    captures = [
        cap("definition.constant", node_for(content, f"CONST_{i}: int = {i}"))
        for i in range(300)
    ]

    uncapped = AstChunkingStrategy().chunk(p, captures)
    capped = AstChunkingStrategy(max_chars=600).chunk(p, captures)

    assert len(uncapped) == 1  # the bug: one blob, embedded from its prefix
    assert len(capped) > 1
    assert all(len(c.text) <= 600 for c in capped)


def test_ast_strategy_should_leave_small_captures_as_one_chunk() -> None:
    content = "A = 1\nB = 2"
    p = parsed(content)
    captures = [
        cap("definition.constant", node_for(content, "A = 1")),
        cap("definition.constant", node_for(content, "B = 2")),
    ]

    chunks = AstChunkingStrategy(max_chars=600).chunk(p, captures)

    assert len(chunks) == 1


def test_section_strategy_should_split_an_oversized_whole_file() -> None:
    p = parsed(_MANY_CONSTANTS)

    chunks = SectionChunkingStrategy(max_chars=500).chunk(p, [])

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_regex_strategy_should_split_an_oversized_section() -> None:
    content = "[table]\n" + _MANY_CONSTANTS
    p = parsed(content)

    chunks = RegexChunkingStrategy(r"^\[+[^\]]+\]", max_chars=500).chunk(p, [])

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_split_chunks_should_keep_distinct_path_aware_ids() -> None:
    """Ids stay distinct while content hashes remain reusable."""
    p = parsed(_MANY_CONSTANTS)

    chunks = SectionChunkingStrategy(max_chars=500).chunk(p, [])

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(
        c.id == chunk_id(c.path, c.text, c.start_line, c.end_line) for c in chunks
    )
    assert all(c.content_hash == content_hash(c.text) for c in chunks)


def test_split_chunks_should_report_line_numbers_within_the_original() -> None:
    p = parsed(_MANY_CONSTANTS)

    chunks = SectionChunkingStrategy(max_chars=500).chunk(p, [])

    assert chunks[0].start_line == 1
    assert all(c.end_line >= c.start_line for c in chunks)
    assert max(c.end_line for c in chunks) <= _MANY_CONSTANTS.count("\n") + 1


# ── Embedder contract ─────────────────────────────────────────────────────────


class _BareEmbedder(Embedder):
    """Adapter that declares no context window."""

    @property
    def dims(self) -> int:
        return 3

    @property
    def model_id(self) -> str:
        return "bare:3"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts)


def test_embedder_should_default_to_the_smallest_common_window() -> None:
    assert _BareEmbedder().max_tokens == 512


# ── The real thing: no chunk may truncate under the real tokenizer ────────────


@pytest.mark.slow
def test_no_chunk_should_truncate_under_the_real_tokenizer() -> None:
    """End-to-end guard against the silent-truncation bug.

    Uses the actual fastembed tokenizer rather than the char-ratio estimate,
    so a bad ratio fails here instead of shipping.
    """
    fastembed = pytest.importorskip("fastembed")

    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    embedder = FastEmbedEmbedder()
    max_tokens = embedder.max_tokens
    # Read off the tokenizer, not hardcoded — the point of the property.
    assert max_tokens > 0

    p = parsed(_MANY_CONSTANTS)
    chunks = SectionChunkingStrategy(budget_chars(max_tokens)).chunk(p, [])
    assert len(chunks) > 1

    # Truncation must be disabled to measure this at all: with it on, encode()
    # caps its own output at max_length, so an over-long chunk reports exactly
    # 512 and the overflow it is meant to reveal stays invisible.
    tokenizer = fastembed.TextEmbedding(embedder._model_name).model.tokenizer
    tokenizer.no_truncation()
    for chunk in chunks:
        n = len(tokenizer.encode(chunk.text).ids)
        assert n <= max_tokens, (
            f"{chunk.path}:{chunk.start_line} needs {n} tokens, "
            f"window is {max_tokens} — its tail would be dropped"
        )

    inventory = ", ".join(f"`P13_ALL{i:04}`" for i in range(300))
    markdown = f"# Attribute inventory\n\n{inventory}\n"
    markdown_parsed = parsed(markdown, "inventory.md", language="markdown")
    markdown_chunks = SectionChunkingStrategy(budget_chars(max_tokens)).chunk(
        markdown_parsed,
        [
            cap(
                "definition.section",
                node_for(markdown, "# Attribute inventory"),
                key="Attribute inventory",
            )
        ],
    )
    for chunk in markdown_chunks:
        n = len(tokenizer.encode(chunk.text).ids)
        assert n <= max_tokens, (
            f"{chunk.path}:{chunk.start_line} needs {n} tokens, "
            f"window is {max_tokens} — its tail would be dropped"
        )
