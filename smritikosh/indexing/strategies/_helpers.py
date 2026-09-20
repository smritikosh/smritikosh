"""Shared chunk-building helpers for the strategies package."""

from __future__ import annotations

import hashlib
from typing import Any

from smritikosh.engine import sm
from smritikosh.models import Chunk, ParsedFile

TreeSitterNode = Any

#: Conservative chars-per-token ratio for sizing a chunk without tokenising it.
#: Measured over 340 real chunks from a 366-file Python repo: median 3.27, mean
#: 3.30, but the *minimum* was 1.39 — punctuation-dense code tokenises far
#: worse than prose.  2.5 is the 5th percentile, so it under-fills the window
#: for typical code rather than overflowing it for the dense tail.  Splitting
#: slightly more than necessary is cheap; truncation is silent data loss.
_CHARS_PER_TOKEN = 2.5

#: Lines of context repeated between consecutive windows of a split chunk, so
#: a definition landing on a seam stays retrievable from either side.
_OVERLAP_LINES = 2

#: Tokens the tokenizer adds itself — [CLS] … [SEP] on BERT-family encoders,
#: which every fastembed default is.  They count against the same window, so
#: a budget that ignores them lands exactly one or two tokens over.
_SPECIAL_TOKENS = 2

#: Extra slack on top of the measured ratio.  The ratio is a 5th-percentile
#: estimate, not a guarantee: a chunk of unusually dense punctuation can still
#: tokenise worse than 2.5 chars/token.  Trading ~10% of the window for
#: headroom is far cheaper than silently truncating a chunk's tail.
_SAFETY_FACTOR = 0.9

#: Chunks below this are too small to carry meaning, and they actively hurt
#: retrieval: cosine similarity rises as content shrinks, so a fragment like
#: ``"descriptor": "chapter"`` (23 chars) outranks whole functions.  Measured on
#: the reference corpus, merging such a fragment with its siblings *raised* its
#: own score (0.652 -> 0.745) — a floor makes chunks more informative, it does
#: not merely suppress noise.
_MIN_CHUNK_CHARS = 80


def budget_chars(max_tokens: int) -> int:
    """Chars a chunk may hold before the model would truncate it.

    Deliberately pessimistic: subtracts the tokenizer's own special tokens,
    then applies a safety factor so a denser-than-average chunk still fits.
    """
    usable = max(max_tokens - _SPECIAL_TOKENS, 1)
    return max(int(usable * _CHARS_PER_TOKEN * _SAFETY_FACTOR), 1)


def split_oversized(
    text: str,
    max_chars: int,
    *,
    overlap_lines: int = _OVERLAP_LINES,
) -> list[tuple[str, int, int]]:
    """Pack *text* into ``(piece, start_offset, end_offset)`` line windows.

    Offsets are zero-based line numbers relative to *text*, so a caller can
    map them onto the chunk's own ``start_line``.  Text at or under budget
    comes back as a single window, which keeps the common path allocation-free.

    Whole lines are never broken, so every window stays independently
    readable.  A single line longer than the budget is emitted on its own
    rather than hard-cut mid-token — one over-long line is a minified bundle
    or a giant literal, and cutting it would corrupt the snippet for no
    retrieval gain.
    """
    if len(text) <= max_chars:
        return [(text, 0, text.count("\n"))]

    lines = text.split("\n")
    windows: list[tuple[str, int, int]] = []
    start = 0
    while start < len(lines):
        size = 0
        end = start
        while end < len(lines):
            # The first line of a window always goes in, even if it alone
            # busts the budget — otherwise an over-long line stalls the walk.
            if size and size + len(lines[end]) + 1 > max_chars:
                break
            size += len(lines[end]) + 1
            end += 1
        windows.append(("\n".join(lines[start:end]), start, end - 1))
        if end >= len(lines):
            break
        # Step back for overlap, but always advance so a budget smaller than
        # the overlap cannot loop forever.
        start = max(end - overlap_lines, start + 1)

    # A trailing remainder below the floor is a fragment, not a chunk — fold it
    # back into its predecessor.  Only when the merged result still fits: the
    # budget is what keeps the embedder from truncating, so overshooting it
    # here would reintroduce the very data loss the split exists to prevent.
    # A fragment that cannot be merged safely is left as its own window.
    if len(windows) > 1 and len(windows[-1][0]) < _MIN_CHUNK_CHARS:
        tail_last = windows[-1][2]
        head_first = windows[-2][1]
        # Re-slice from *lines* rather than joining the two window strings: the
        # windows overlap, so concatenating them would repeat the shared lines.
        merged = "\n".join(lines[head_first : tail_last + 1])
        if len(merged) <= max_chars:
            windows[-2:] = [(merged, head_first, tail_last)]
    return windows


def chunk_id(path: str, text: str, start_line: int, end_line: int) -> str:
    """Return a stable location-aware storage id."""
    identity: str = f"{path}\0{start_line}:{end_line}\0{text}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    """Full SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode()).hexdigest()


def extract_node_text(
    parsed: ParsedFile,
    node: TreeSitterNode,
    content_bytes: bytes | None = None,
) -> str:
    """Extract node text using byte offsets. Pass *content_bytes* to encode once."""
    raw = content_bytes if content_bytes is not None else parsed.content.encode()
    return raw[node.start_byte : node.end_byte].decode()


@sm.tracked
def build_chunk(
    path: str,
    text: str,
    chunk_kind: str,
    start_line: int,
    end_line: int,
    symbol: str | None = None,
) -> Chunk:
    """Build a path-addressed chunk carrying a separate content hash."""
    digest = content_hash(text)
    return Chunk(
        id=chunk_id(path, text, start_line, end_line),
        path=path,
        start_line=start_line,
        end_line=end_line,
        text=text,
        chunk_kind=chunk_kind,
        content_hash=digest,
        symbol=symbol,
    )


@sm.tracked
def build_chunks(
    path: str,
    text: str,
    chunk_kind: str,
    start_line: int,
    end_line: int,
    max_chars: int | None = None,
    symbol: str | None = None,
) -> list[Chunk]:
    """Build one Chunk per line-window of *text*, splitting if over budget.

    ``max_chars=None`` disables splitting and yields exactly one chunk, which
    is what the tests and any caller without an embedder in scope want.
    Because ids include path and text, a split piece is memoised independently
    without colliding with identical source in another file.

    Every window of a split definition keeps that definition's *symbol*: the
    windows are pieces of one function, and naming them all lets a reader see
    which definition a fragment belongs to.
    """
    if max_chars is None or len(text) <= max_chars:
        return [build_chunk(path, text, chunk_kind, start_line, end_line, symbol)]
    return [
        build_chunk(
            path,
            piece,
            chunk_kind,
            start_line + first,
            start_line + last,
            symbol,
        )
        for piece, first, last in split_oversized(text, max_chars)
    ]


def make_node_chunks(
    parsed: ParsedFile,
    node: TreeSitterNode,
    chunk_kind: str,
    content_bytes: bytes | None = None,
    max_chars: int | None = None,
    symbol: str | None = None,
) -> list[Chunk]:
    """Build Chunks from a single tree-sitter node, splitting if oversized."""
    text = extract_node_text(parsed, node, content_bytes)
    return build_chunks(
        parsed.path,
        text,
        chunk_kind,
        node.start_point[0] + 1,
        node.end_point[0] + 1,
        max_chars,
        symbol,
    )


def make_group_chunks(
    parsed: ParsedFile,
    nodes: list[TreeSitterNode],
    chunk_kind: str,
    content_bytes: bytes | None = None,
    max_chars: int | None = None,
) -> list[Chunk]:
    """Build Chunks by concatenating *nodes*, splitting if oversized.

    This is the path that produced the 190-line constants blob: every
    module-level constant in a file lands in one bucket, so the group is
    exactly the kind of chunk that outgrows the model's window.
    """
    raw = content_bytes if content_bytes is not None else parsed.content.encode()
    texts = [extract_node_text(parsed, n, raw) for n in nodes]
    return build_chunks(
        parsed.path,
        "\n".join(texts),
        chunk_kind,
        min(n.start_point[0] + 1 for n in nodes),
        max(n.end_point[0] + 1 for n in nodes),
        max_chars,
    )


def make_text_chunks(
    parsed: ParsedFile,
    text: str,
    chunk_kind: str,
    start_line: int,
    end_line: int,
    max_chars: int | None = None,
    symbol: str | None = None,
) -> list[Chunk]:
    """Build Chunks from a pre-extracted text string, splitting if oversized."""
    return build_chunks(
        parsed.path, text, chunk_kind, start_line, end_line, max_chars, symbol
    )
