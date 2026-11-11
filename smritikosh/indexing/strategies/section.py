"""Section-based chunking for document files (Markdown, JSON, YAML)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from smritikosh.engine import sm
from smritikosh.indexing.strategies._helpers import (
    _MIN_CHUNK_CHARS,
    TreeSitterNode,
    build_chunk,
    make_text_chunks,
    split_oversized,
)
from smritikosh.models import Capture, Chunk, ParsedFile

__all__ = ["SectionChunkingStrategy"]

#: Capture names the JSON query emits, shallowest first.  Selection walks this
#: order and stops at the first depth whose capture fits the budget.
_JSON_DEPTHS: Final[tuple[str, ...]] = (
    "definition.section",
    "definition.subsection",
    "definition.subsubsection",
)

#: Comment marker for the key-path line prepended to a structured chunk.
#: JSON has no comment syntax, so ``//`` is a label rather than a comment.
#: YAML uses ``#``, which is a real comment and stays out of the mapping.
_PATH_MARKERS: Final[dict[str, str]] = {"json": "// ", "yaml": "# "}

#: Languages whose tags query emits the three depths in :data:`_JSON_DEPTHS`.
_STRUCTURED_LANGUAGES: Final[frozenset[str]] = frozenset({"json", "yaml"})

_MARKDOWN_SECTION_CAPTURE: Final[str] = "definition.section"
_MARKDOWN_BREAK_FLOOR: Final[float] = 0.55
# Markdown inventories can be punctuation-dense enough to tokenize at roughly
# 1.55 chars/token. The shared 2.5 chars/token budget is tuned for mixed source,
# so reserve additional headroom here to keep documentation under the window.
_MARKDOWN_BUDGET_FACTOR: Final[float] = 0.65
_MARKDOWN_TOKEN_RESERVE: Final[float] = 0.0625
_MARKDOWN_SOFT_TARGET: Final[float] = 0.8
#: Share of a chunk's budget the heading breadcrumb may spend. The breadcrumb
#: repeats every ancestor, so a deep hierarchy of long headings would otherwise
#: crowd out the content it is supposed to introduce.
_MARKDOWN_PREFIX_SHARE: Final[int] = 3


@dataclass(frozen=True)
class _MarkdownBlock:
    """A semantic Markdown block with source-relative line offsets."""

    text: str
    first_line: int
    last_line: int


def _heading_level(capture: Capture, raw: bytes) -> int:
    """Return the Markdown level of a captured ATX or Setext heading."""
    heading = raw[capture.node.start_byte : capture.node.end_byte].decode()
    first_line = heading.lstrip().splitlines()[0]
    if first_line.startswith("#"):
        return min(len(first_line) - len(first_line.lstrip("#")), 6)
    lines = heading.rstrip().splitlines()
    if lines and lines[-1].lstrip().startswith("="):
        return 1
    return 2


def _render_breadcrumb(hierarchy: list[tuple[int, str]]) -> str:
    """Render a heading stack as Markdown, one heading per line."""
    return "\n".join(f"{'#' * level} {name}" for level, name in hierarchy)


def _heading_prefix(
    hierarchy: list[tuple[int, str]],
    budget: int | None,
    measure: Callable[[str], int] = len,
) -> str:
    """Build a compact Markdown breadcrumb, retaining nearest ancestors first.

    *budget* is the chunk budget and *measure* counts in whatever unit that
    budget is expressed in, so a caller sizing chunks by model tokens caps the
    breadcrumb in tokens too.  Ancestors are dropped outermost-first, and a
    nearest heading that still overruns on its own is trimmed: a chunk made
    almost entirely of repeated ancestor headings embeds as its hierarchy
    rather than as its content.
    """
    if not hierarchy:
        return ""
    if budget is None:
        return _render_breadcrumb(hierarchy) + "\n\n"
    limit = max(min(budget // _MARKDOWN_PREFIX_SHARE, budget - 1), 0)
    if limit == 0:
        return ""
    selected = hierarchy[:]
    while len(selected) > 1 and measure(_render_breadcrumb(selected) + "\n\n") > limit:
        selected.pop(0)
    prefix = _render_breadcrumb(selected) + "\n\n"
    if measure(prefix) <= limit:
        return prefix
    # One heading over the share on its own: keep as much of its text as fits
    # rather than emit a breadcrumb that leaves no room for the body.
    level, name = selected[-1]
    low, high, best = 0, len(name), 0
    while low <= high:
        middle = (low + high) // 2
        if measure(_render_breadcrumb([(level, name[:middle])]) + "\n\n") <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == 0:
        return ""
    return _render_breadcrumb([(level, name[:best].rstrip())]) + "\n\n"


def _preferred_markdown_cut(window: str, minimum: int) -> int:
    """Find the strongest semantic boundary near the end of a window."""
    paragraph_cut = window.rfind("\n\n") + 2
    if paragraph_cut > 1:
        return paragraph_cut
    line_cut = window.rfind("\n") + 1
    if line_cut > 0:
        return line_cut
    sentence_ends = list(re.finditer(r"""[.!?][)"'\]]*\s+""", window))
    if sentence_ends:
        sentence_cut = sentence_ends[-1].end()
        if sentence_cut >= minimum:
            return sentence_cut
    spaces = list(re.finditer(r"\s+", window))
    if spaces:
        word_cut = spaces[-1].end()
        if word_cut >= minimum:
            return word_cut
    return len(window)


def _split_markdown(
    text: str,
    max_chars: int,
) -> list[tuple[str, int, int]]:
    """Split prose at paragraph, line, sentence, then word boundaries."""
    if not text:
        return []
    windows: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        stop = min(start + max_chars, len(text))
        if stop < len(text):
            window = text[start:stop]
            minimum = max(int(max_chars * _MARKDOWN_BREAK_FLOOR), 1)
            stop = start + _preferred_markdown_cut(window, minimum)
        piece = text[start:stop]
        first_line = text.count("\n", 0, start)
        last_line = first_line + piece.count("\n")
        windows.append((piece, first_line, last_line))
        start = stop
    return windows


def _markdown_blocks(text: str) -> list[_MarkdownBlock]:
    """Split Markdown into paragraphs, tables, lists, and fenced-code blocks."""
    blocks: list[_MarkdownBlock] = []
    current: list[str] = []
    first_line = 0
    fence: str | None = None
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        marker = stripped[:3] if stripped.startswith(("```", "~~~")) else None
        if marker and (fence is None or marker == fence):
            fence = marker if fence is None else None
        if not line.strip() and fence is None:
            current.append(line)
            block_text = "".join(current)
            if block_text.strip() and not re.fullmatch(
                r"(?:-{3,}|\*{3,}|_{3,})", block_text.strip()
            ):
                blocks.append(_MarkdownBlock(block_text, first_line, index))
            current = []
            first_line = index + 1
            continue
        if not current:
            first_line = index
        current.append(line)
    block_text = "".join(current)
    if block_text.strip() and not re.fullmatch(
        r"(?:-{3,}|\*{3,}|_{3,})", block_text.strip()
    ):
        blocks.append(_MarkdownBlock(block_text, first_line, len(lines) - 1))
    return blocks


def _token_split_block(
    block: _MarkdownBlock,
    prefix: str,
    token_counter: Callable[[str], int],
    hard_limit: int,
) -> list[_MarkdownBlock]:
    """Split one oversized block using exact model-token measurements."""
    windows: list[_MarkdownBlock] = []
    start = 0
    while start < len(block.text):
        low, high, best = 1, len(block.text) - start, 0
        while low <= high:
            middle = (low + high) // 2
            candidate = prefix + block.text[start : start + middle]
            if token_counter(candidate) <= hard_limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        best = max(best, 1)
        if start + best < len(block.text):
            window = block.text[start : start + best]
            minimum = max(int(best * _MARKDOWN_BREAK_FLOOR), 1)
            best = _preferred_markdown_cut(window, minimum)
        piece = block.text[start : start + best]
        first = block.first_line + block.text.count("\n", 0, start)
        last = first + piece.count("\n")
        windows.append(_MarkdownBlock(piece, first, last))
        start += best
    return windows


def _build_markdown_pack(
    parsed: ParsedFile,
    blocks: list[_MarkdownBlock],
    start_line: int,
    prefix: str,
    symbol: str | None,
) -> Chunk:
    """Build one chunk from a run of complete semantic blocks."""
    return build_chunk(
        parsed.path,
        prefix + "".join(block.text for block in blocks),
        "section",
        start_line + blocks[0].first_line,
        start_line + blocks[-1].last_line,
        symbol,
    )


def _pack_markdown_blocks(
    parsed: ParsedFile,
    text: str,
    start_line: int,
    prefix: str,
    symbol: str | None,
    token_counter: Callable[[str], int],
    max_tokens: int,
) -> list[Chunk]:
    """Pack complete Markdown blocks toward a soft target under a hard limit."""
    hard_limit = max(
        max_tokens - max(int(max_tokens * _MARKDOWN_TOKEN_RESERVE), 1),
        1,
    )
    soft_target = max(int(hard_limit * _MARKDOWN_SOFT_TARGET), 1)
    chunks: list[Chunk] = []
    pending: list[_MarkdownBlock] = []
    # Carried across iterations rather than re-derived: measuring the pending
    # run again on every block re-tokenizes the whole buffer each time, and
    # the count is already known from the step that admitted the last block.
    empty_tokens: int = token_counter(prefix)
    pending_text: str = ""
    pending_tokens: int = empty_tokens
    for block in _markdown_blocks(text):
        candidate_tokens: int = token_counter(prefix + pending_text + block.text)
        crosses_target_farther = candidate_tokens >= soft_target and abs(
            pending_tokens - soft_target
        ) <= abs(candidate_tokens - soft_target)
        if pending and (
            pending_tokens >= soft_target
            or candidate_tokens > hard_limit
            or crosses_target_farther
        ):
            chunks.append(
                _build_markdown_pack(parsed, pending, start_line, prefix, symbol)
            )
            pending = []
            pending_text = ""
            pending_tokens = empty_tokens
            # The run this block was measured against is gone, so what matters
            # now is whether the block fits a chunk of its own.
            candidate_tokens = token_counter(prefix + block.text)
        if candidate_tokens > hard_limit:
            pieces = _token_split_block(block, prefix, token_counter, hard_limit)
            chunks.extend(
                _build_markdown_pack(parsed, [piece], start_line, prefix, symbol)
                for piece in pieces
            )
        else:
            pending.append(block)
            pending_text += block.text
            pending_tokens = candidate_tokens
    if pending:
        chunks.append(_build_markdown_pack(parsed, pending, start_line, prefix, symbol))
    return chunks


def _merge_markdown_fragments(
    chunks: list[Chunk],
    max_chars: int | None,
    token_counter: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Merge adjacent tiny Markdown chunks from one section.

    Only chunks sharing a heading merge.  Chunks arrive here for the whole
    file, so an undersized section would otherwise absorb the one after it and
    report that one's heading as its symbol -- the outline and every cited
    ``PATH:START-END`` would then attribute the first section's prose to the
    second.  A section too small to reach the floor on its own stays small;
    it still carries its own breadcrumb, which is the context the merge was
    trying to buy.
    """
    if max_chars is None and (token_counter is None or max_tokens is None):
        return chunks

    def fits(text: str) -> bool:
        if token_counter is not None and max_tokens is not None:
            reserve = max(int(max_tokens * _MARKDOWN_TOKEN_RESERVE), 1)
            return token_counter(text) <= max(max_tokens - reserve, 1)
        return max_chars is not None and len(text) <= max_chars

    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and len(merged[-1].text) < _MIN_CHUNK_CHARS
            and merged[-1].symbol == chunk.symbol
        ):
            previous = merged[-1]
            text = f"{previous.text.rstrip()}\n\n{chunk.text.lstrip()}"
            if fits(text):
                merged[-1] = build_chunk(
                    chunk.path,
                    text,
                    "section",
                    previous.start_line,
                    chunk.end_line,
                    chunk.symbol,
                )
                continue
        merged.append(chunk)
    if (
        len(merged) > 1
        and len(merged[-1].text) < _MIN_CHUNK_CHARS
        and merged[-2].symbol == merged[-1].symbol
    ):
        previous, tail = merged[-2:]
        text = f"{previous.text.rstrip()}\n\n{tail.text.lstrip()}"
        if fits(text):
            merged[-2:] = [
                build_chunk(
                    tail.path,
                    text,
                    "section",
                    previous.start_line,
                    tail.end_line,
                    tail.symbol,
                )
            ]
    return merged


def _markdown_text_chunks(
    parsed: ParsedFile,
    text: str,
    start_line: int,
    prefix: str = "",
    symbol: str | None = None,
    max_chars: int | None = None,
    token_counter: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Build strictly budgeted Markdown chunks with context on every window."""
    if not text.strip():
        return []
    if token_counter is not None and max_tokens is not None:
        return _pack_markdown_blocks(
            parsed,
            text,
            start_line,
            prefix,
            symbol,
            token_counter,
            max_tokens,
        )
    budget = None if max_chars is None else max(max_chars - len(prefix), 1)
    windows = (
        [(text, 0, text.count("\n"))]
        if budget is None
        else _split_markdown(text, budget)
    )
    if not windows:
        windows = [("", 0, 0)]
    return [
        build_chunk(
            parsed.path,
            prefix + piece,
            "section",
            start_line + first,
            start_line + last,
            symbol,
        )
        for piece, first, last in windows
    ]


def _chunk_markdown(
    parsed: ParsedFile,
    captures: list[Capture],
    max_chars: int | None,
    max_tokens: int | None,
    token_counter: Callable[[str], int] | None,
) -> list[Chunk]:
    """Partition Markdown by headings without nested or code-block duplication."""
    # The breadcrumb is budgeted in whatever unit sizes the chunk.  Token mode
    # clears max_chars, so it must hand the breadcrumb a token budget instead
    # of inheriting the cleared one and ending up with no cap at all.
    prefix_budget: int | None = max_chars
    prefix_measure: Callable[[str], int] = len
    if token_counter is not None and max_tokens is not None:
        max_chars = None
        prefix_budget = max_tokens
        prefix_measure = token_counter
    elif max_chars is not None:
        max_chars = max(int(max_chars * _MARKDOWN_BUDGET_FACTOR), 1)
        prefix_budget = max_chars
    raw = parsed.content.encode()
    headings = sorted(
        (c for c in captures if c.capture_name == _MARKDOWN_SECTION_CAPTURE),
        key=lambda capture: capture.node.start_byte,
    )
    if not headings:
        chunks = _markdown_text_chunks(
            parsed,
            parsed.content,
            1,
            max_chars=max_chars,
            token_counter=token_counter,
            max_tokens=max_tokens,
        )
        return _merge_markdown_fragments(
            chunks,
            max_chars,
            token_counter,
            max_tokens,
        )

    chunks: list[Chunk] = []
    first_start = headings[0].node.start_byte
    preamble = raw[:first_start].decode()
    chunks.extend(
        _markdown_text_chunks(
            parsed,
            preamble,
            1,
            max_chars=max_chars,
            token_counter=token_counter,
            max_tokens=max_tokens,
        )
    )

    hierarchy: list[tuple[int, str]] = []
    for index, capture in enumerate(headings):
        level = _heading_level(capture, raw)
        hierarchy = [(depth, name) for depth, name in hierarchy if depth < level]
        hierarchy.append((level, capture.name.strip()))
        prefix = _heading_prefix(hierarchy, prefix_budget, prefix_measure)
        next_start = (
            headings[index + 1].node.start_byte
            if index + 1 < len(headings)
            else len(raw)
        )
        body = raw[capture.node.end_byte : next_start].decode()
        leading_newlines = len(body) - len(body.lstrip("\n"))
        body = body[leading_newlines:]
        body_line = capture.node.end_point[0] + 1 + leading_newlines
        symbol = " > ".join(name for _, name in hierarchy)
        chunks.extend(
            _markdown_text_chunks(
                parsed,
                body,
                body_line,
                prefix,
                symbol,
                max_chars,
                token_counter,
                max_tokens,
            )
        )
    if not chunks and parsed.content.strip():
        chunks = _markdown_text_chunks(
            parsed,
            parsed.content,
            1,
            max_chars=max_chars,
            token_counter=token_counter,
            max_tokens=max_tokens,
        )
    return _merge_markdown_fragments(
        chunks,
        max_chars,
        token_counter,
        max_tokens,
    )


def _contains(outer: TreeSitterNode, inner: TreeSitterNode) -> bool:
    """True when *inner*'s byte range sits inside *outer*'s.

    Containment is decided on byte offsets rather than by walking
    ``node.parent`` — the same idiom ``ast.py`` uses for class/init pairing.
    Offsets are all a capture is guaranteed to carry.
    """
    return (
        inner.start_byte >= outer.start_byte
        and inner.end_byte <= outer.end_byte
        # Strictly smaller, so a node never counts as containing itself.
        and (inner.end_byte - inner.start_byte) < (outer.end_byte - outer.start_byte)
    )


def _select_by_size(
    captures: list[Capture],
    max_chars: int | None,
) -> list[tuple[Capture, list[Capture]]]:
    """Pick the shallowest captures that fit *max_chars*, with their ancestors.

    Returns ``(capture, ancestors)`` pairs, outermost ancestor first, so a
    caller can build a key path.  A capture within budget is emitted and
    everything nested inside it is skipped; an oversized one is discarded in
    favour of the captures it contains.  Depth is chosen by size, so a small
    config still yields one chunk per top-level key while a large one splits
    along its own key boundaries.

    Descent stops in two cases, both falling back to line-window splitting: a
    capture with no deeper capture inside it, and one whose children are all
    below :data:`_MIN_CHUNK_CHARS` — splitting into fragments that small would
    hurt retrieval more than a coarse chunk does.
    """
    by_depth: list[list[Capture]] = [
        [c for c in captures if c.capture_name == name] for name in _JSON_DEPTHS
    ]
    # Anything the query emitted under an unexpected name still gets chunked.
    extra = [c for c in captures if c.capture_name not in _JSON_DEPTHS]

    selected: list[tuple[Capture, list[Capture]]] = []

    def walk(cap: Capture, depth: int, ancestors: list[Capture]) -> None:
        size = cap.node.end_byte - cap.node.start_byte
        children = (
            [c for c in by_depth[depth + 1] if _contains(cap.node, c.node)]
            if depth + 1 < len(by_depth)
            else []
        )
        if max_chars is None or size <= max_chars or not children:
            selected.append((cap, ancestors))
            return
        # Descending would fragment this section into keys too small to carry
        # meaning; keep the parent whole and let line-windowing split it.
        if all(
            (c.node.end_byte - c.node.start_byte) < _MIN_CHUNK_CHARS for c in children
        ):
            selected.append((cap, ancestors))
            return
        for child in children:
            walk(child, depth + 1, [*ancestors, cap])

    for top in by_depth[0]:
        walk(top, 0, [])
    selected.extend((c, []) for c in extra)
    selected.sort(key=lambda pair: pair[0].node.start_byte)
    return selected


def _key_path(cap: Capture, ancestors: list[Capture]) -> str:
    """Dotted key path for *cap*, e.g. ``chapters.overview.error_codes``."""
    return ".".join(c.name for c in [*ancestors, cap] if c.name)


def _run_path(run: list[tuple[Capture, list[Capture]]]) -> str:
    """Key path for a run: the single key, or ``parent.a+b`` for merged ones."""
    cap, ancestors = run[0]
    if len(run) == 1:
        return _key_path(cap, ancestors)
    parent = ".".join(a.name for a in ancestors if a.name)
    keys = "+".join(c.name for c, _ in run if c.name)
    return f"{parent}.{keys}" if parent else keys


def _merge_runs(
    selected: list[tuple[Capture, list[Capture]]],
    max_chars: int | None,
) -> list[list[tuple[Capture, list[Capture]]]]:
    """Group adjacent undersized siblings into runs that fit *max_chars*.

    A small config is mostly one-line keys (``"projectKey": "ICCSVC"``), and one
    chunk each retrieves badly — similarity climbs as content shrinks, so those
    fragments crowd out real matches.  Grouping siblings under their shared
    parent path gives each chunk enough context to be worth matching.

    Only same-parent neighbours merge, so a run never spans two unrelated
    subtrees.  A capture already at or above the floor stands alone.
    """
    runs: list[list[tuple[Capture, list[Capture]]]] = []
    for pair in selected:
        cap, ancestors = pair
        size = cap.node.end_byte - cap.node.start_byte
        if runs and size < _MIN_CHUNK_CHARS:
            run = runs[-1]
            last_cap, last_ancestors = run[-1]
            same_parent = [a.name for a in last_ancestors] == [
                a.name for a in ancestors
            ]
            run_size = cap.node.end_byte - run[0][0].node.start_byte
            last_size = last_cap.node.end_byte - last_cap.node.start_byte
            fits = max_chars is None or run_size <= max_chars
            if same_parent and last_size < _MIN_CHUNK_CHARS and fits:
                run.append(pair)
                continue
        runs.append([pair])
    return runs


def _document_span(node: TreeSitterNode) -> tuple[int, int] | None:
    """Return the byte range of the YAML document that contains *node*."""
    current = node
    while current is not None:
        if getattr(current, "type", None) == "document":
            return current.start_byte, current.end_byte
        current = getattr(current, "parent", None)
    return None


def _pair_scalar(content: str, node: TreeSitterNode) -> str:
    """Return the scalar after the first colon of a ``key: value`` pair."""
    text = content.encode()[node.start_byte : node.end_byte].decode()
    value = text.split(":", 1)[1].strip() if ":" in text else ""
    value = value.splitlines()[0].strip() if value else ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.split()[0] if value else ""


def _yaml_resource(content: str, captures: list[Capture], node: TreeSitterNode) -> str:
    """Return ``Kind/name`` for the document that contains *node*.

    ``kind`` is a top-level key. ``name`` is the ``name`` key inside that
    document's ``metadata`` block, so a nested ``spec`` name is not picked up.
    An empty string means the document has no such keys, or the node is a test
    double with no parent chain.
    """
    span = _document_span(node)
    if span is None:
        return ""
    start, end = span

    def inside(cap: Capture) -> bool:
        return start <= cap.node.start_byte and cap.node.end_byte <= end

    kind = ""
    for cap in captures:
        if (
            cap.capture_name == "definition.section"
            and cap.name == "kind"
            and inside(cap)
        ):
            kind = _pair_scalar(content, cap.node)
            break

    resource_name = ""
    for cap in captures:
        if (
            cap.capture_name != "definition.section"
            or cap.name != "metadata"
            or not inside(cap)
        ):
            continue
        for child in captures:
            if (
                child.capture_name == "definition.subsection"
                and child.name == "name"
                and cap.node.start_byte <= child.node.start_byte
                and child.node.end_byte <= cap.node.end_byte
            ):
                resource_name = _pair_scalar(content, child.node)
                break
        break

    if kind and resource_name:
        return f"{kind}/{resource_name}"
    return kind


def _chunk_heading(path: str, resource: str) -> str:
    """Label stored on a structured chunk, without its comment marker."""
    if resource:
        return f"{resource} {path}".strip()
    return path


class SectionChunkingStrategy:
    """One chunk per section capture, sized to the embedder's window.

    Markdown headings define non-overlapping source spans. Each chunk receives
    its active heading hierarchy as a breadcrumb, and oversized prose splits at
    paragraph, line, sentence, or word boundaries in that order.

    JSON and YAML share three behaviours:

    * **Depth by size.** The query captures three nesting levels and selection
      takes the shallowest that *fits* (:func:`_select_by_size`), so chunks
      align with key boundaries rather than arbitrary lines.
    * **Runs.** Adjacent undersized siblings merge into one chunk
      (:func:`_merge_runs`), labelled ``parent.a+b`` — a config of one-line
      keys would otherwise become chunks too small to retrieve.
    * **Key paths.** Each chunk is prefixed with its dotted path, because a
      mid-file fragment is otherwise unidentifiable to a reader and to the
      embedder alike.  The prefix is part of ``text``, so it is both embedded
      and shown as the search snippet; ``start_line`` still points at the
      captured value, so the printed range covers one line fewer than the
      printed snippet.  YAML leads that prefix with ``Kind/name`` when the
      document has a top-level ``kind`` and a ``metadata.name``.
    """

    mode_name = "section"

    def __init__(
        self,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.token_counter = token_counter

    @sm.tracked
    def chunk(self, parsed: ParsedFile, captures: list[Capture]) -> list[Chunk]:
        if parsed.language == "markdown":
            return _chunk_markdown(
                parsed,
                captures,
                self.max_chars,
                self.max_tokens,
                self.token_counter,
            )

        if not captures:
            lines = parsed.content.splitlines()
            return make_text_chunks(
                parsed,
                parsed.content,
                "section",
                1,
                max(len(lines), 1),
                self.max_chars,
            )

        structured = parsed.language in _STRUCTURED_LANGUAGES
        raw = parsed.content.encode()
        selected = _select_by_size(captures, self.max_chars if structured else None)
        groups = (
            _merge_runs(selected, self.max_chars)
            if structured
            else [[pair] for pair in selected]
        )
        chunks: list[Chunk] = []
        for run in groups:
            cap, ancestors = run[0]
            last_node = run[-1][0].node
            # A run spans from its first capture to its last, so the text
            # between merged siblings (commas, newlines) travels with them.
            body = raw[cap.node.start_byte : last_node.end_byte].decode()
            first_line = cap.node.start_point[0] + 1
            path = _run_path(run) if structured else ""
            if not path:
                chunks.extend(
                    make_text_chunks(
                        parsed,
                        body,
                        "section",
                        first_line,
                        last_node.end_point[0] + 1,
                        self.max_chars,
                        cap.name or None,
                    )
                )
                continue

            resource = (
                _yaml_resource(parsed.content, captures, cap.node)
                if parsed.language == "yaml"
                else ""
            )
            heading = _chunk_heading(path, resource)
            # Split the body first, then label each window.  Prefixing before
            # the split would make the prefix its own line, so the splitter
            # could peel it off into a 4-char chunk and leave the body
            # unlabelled — the exact degenerate fragment this work removes.
            prefix = f"{_PATH_MARKERS[parsed.language]}{heading}\n"
            budget = (
                None if self.max_chars is None else max(self.max_chars - len(prefix), 1)
            )
            windows = (
                split_oversized(body, budget) if budget is not None else [(body, 0, 0)]
            )
            for piece, first, last in windows:
                chunks.append(
                    build_chunk(
                        parsed.path,
                        prefix + piece,
                        "section",
                        first_line + first,
                        first_line + last,
                        heading,
                    )
                )
        return chunks
