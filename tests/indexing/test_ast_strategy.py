"""Tests for AstChunkingStrategy, _deduplicate_by_priority, and _find_class_init."""

from __future__ import annotations

from smritikosh.indexing.strategies.ast import (
    CAPTURE_PRIORITY,
    AstChunkingStrategy,
    _deduplicate_by_priority,
    _find_class_init,
)
from tests.indexing.conftest import cap, cid, node_for, parsed

# ── _deduplicate_by_priority ──────────────────────────────────────────────


class TestDeduplicateByPriority:
    def test_keeps_single_capture_unchanged(self) -> None:
        content = "x = 1"
        node = node_for(content, "x = 1", node_id=1)

        result = _deduplicate_by_priority(
            [cap("definition.constant", node)], CAPTURE_PRIORITY
        )

        assert len(result) == 1

    def test_enum_beats_class_for_same_node(self) -> None:
        content = "class Tenant(str, Enum): pass"
        node = node_for(content, content, node_id=42)
        class_cap = cap("definition.class", node)
        enum_cap = cap("definition.enum", node)

        result = _deduplicate_by_priority([class_cap, enum_cap], CAPTURE_PRIORITY)

        assert len(result) == 1
        assert result[0].capture_name == "definition.enum"

    def test_order_does_not_affect_winner(self) -> None:
        content = "class Foo(str, Enum): pass"
        node = node_for(content, content, node_id=7)
        enum_first = [cap("definition.enum", node), cap("definition.class", node)]
        class_first = [cap("definition.class", node), cap("definition.enum", node)]

        r1 = _deduplicate_by_priority(enum_first, CAPTURE_PRIORITY)
        r2 = _deduplicate_by_priority(class_first, CAPTURE_PRIORITY)

        assert r1[0].capture_name == r2[0].capture_name == "definition.enum"

    def test_different_nodes_both_kept(self) -> None:
        content = "def foo(): pass\ndef bar(): pass\n"
        n1 = node_for(content, "def foo(): pass", node_id=10)
        n2 = node_for(content, "def bar(): pass", node_id=11)

        result = _deduplicate_by_priority(
            [cap("definition.function", n1), cap("definition.function", n2)],
            CAPTURE_PRIORITY,
        )

        assert len(result) == 2

    def test_sorted_by_start_byte(self) -> None:
        content = "def first(): pass\ndef second(): pass\n"
        second = node_for(content, "def second(): pass", node_id=2)
        first = node_for(content, "def first(): pass", node_id=1)

        result = _deduplicate_by_priority(
            [cap("definition.function", second), cap("definition.function", first)],
            CAPTURE_PRIORITY,
        )

        assert [c.node.id for c in result] == [1, 2]


# ── _find_class_init ──────────────────────────────────────────────────────


class TestFindClassInit:
    def test_finds_init_nested_inside_class(self) -> None:
        content = "class Foo:\n    def __init__(self): pass\n"
        class_node = node_for(content, content.rstrip("\n"))
        init_node = node_for(content, "def __init__(self): pass")
        init_cap = cap("definition.class_init", init_node)

        assert _find_class_init([init_cap], class_node, skip=set()) is init_cap

    def test_skips_already_used_init(self) -> None:
        content = "class Foo:\n    def __init__(self): pass\n"
        class_node = node_for(content, content.rstrip("\n"))
        init_node = node_for(content, "def __init__(self): pass")
        init_cap = cap("definition.class_init", init_node)

        assert _find_class_init([init_cap], class_node, skip={init_node.id}) is None

    def test_ignores_non_class_init_captures(self) -> None:
        content = "class Foo:\n    def method(self): pass\n"
        class_node = node_for(content, content.rstrip("\n"))
        method_node = node_for(content, "def method(self): pass")

        result = _find_class_init(
            [cap("definition.method", method_node)], class_node, skip=set()
        )
        assert result is None

    def test_returns_none_when_captures_empty(self) -> None:
        content = "class Foo: pass"
        class_node = node_for(content, content)
        assert _find_class_init([], class_node, skip=set()) is None


# ── AstChunkingStrategy ───────────────────────────────────────────────────


class TestAstChunkingStrategy:
    strategy = AstChunkingStrategy()

    def test_mode_name(self) -> None:
        assert self.strategy.mode_name == "ast"

    def test_function_gets_own_chunk(self) -> None:
        content = "def foo():\n    return 1\n"
        p = parsed(content)
        node = node_for(content, "def foo():\n    return 1")

        chunks = self.strategy.chunk(p, [cap("definition.function", node)])

        assert len(chunks) == 1
        assert chunks[0].chunk_kind == "function"
        assert "def foo" in chunks[0].text

    def test_function_chunk_carries_its_symbol_name(self) -> None:
        content = "def foo():\n    return 1\n"
        p = parsed(content)
        node = node_for(content, "def foo():\n    return 1")

        chunks = self.strategy.chunk(p, [cap("definition.function", node, key="foo")])

        assert chunks[0].symbol == "foo"

    def test_every_window_of_a_split_function_keeps_the_symbol(self) -> None:
        body = "\n".join(f"    line_{n} = {n}" for n in range(40))
        content = f"def wide():\n{body}\n"
        p = parsed(content)
        node = node_for(content, f"def wide():\n{body}")

        chunks = AstChunkingStrategy(max_chars=120).chunk(
            p, [cap("definition.function", node, key="wide")]
        )

        assert len(chunks) > 1
        assert {chunk.symbol for chunk in chunks} == {"wide"}

    def test_grouped_constants_have_no_single_symbol(self) -> None:
        content = "A = 1\nB = 2\n"
        p = parsed(content)

        chunks = self.strategy.chunk(
            p,
            [
                cap("definition.constant", node_for(content, "A = 1"), key="A"),
                cap("definition.constant", node_for(content, "B = 2"), key="B"),
            ],
        )

        assert [chunk.symbol for chunk in chunks] == [None]

    def test_class_init_absorbed_into_class_chunk(self) -> None:
        content = "class Foo:\n    def __init__(self):\n        pass\n"
        p = parsed(content)
        class_text = "class Foo:\n    def __init__(self):\n        pass"
        class_node = node_for(content, class_text)
        init_node = node_for(content, "def __init__(self):\n        pass")

        chunks = self.strategy.chunk(
            p,
            [
                cap("definition.class", class_node),
                cap("definition.class_init", init_node),
            ],
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_kind == "class"
        assert "__init__" in chunks[0].text

    def test_class_without_init_still_produces_chunk(self) -> None:
        content = "class Bar:\n    x = 1\n"
        p = parsed(content)
        class_node = node_for(content, "class Bar:\n    x = 1")

        chunks = self.strategy.chunk(p, [cap("definition.class", class_node)])

        assert len(chunks) == 1
        assert chunks[0].chunk_kind == "class"

    def test_enum_dedup_wins_over_class(self) -> None:
        content = "class Color(str, Enum):\n    RED = 'red'\n"
        p = parsed(content)
        node = node_for(content, content.rstrip("\n"), node_id=99)

        chunks = self.strategy.chunk(
            p,
            [cap("definition.class", node), cap("definition.enum", node)],
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_kind == "enum"

    def test_constants_grouped_into_one_chunk(self) -> None:
        content = "X = 1\nY = 2\n"
        p = parsed(content)
        n1 = node_for(content, "X = 1")
        n2 = node_for(content, "Y = 2")

        chunks = self.strategy.chunk(
            p,
            [cap("definition.constant", n1), cap("definition.constant", n2)],
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_kind == "constant"
        assert "X = 1" in chunks[0].text and "Y = 2" in chunks[0].text

    def test_class_init_skip_is_not_emitted_standalone(self) -> None:
        content = "class Foo:\n    def __init__(self): pass\n"
        p = parsed(content)
        init_node = node_for(content, "def __init__(self): pass")

        chunks = self.strategy.chunk(p, [cap("definition.class_init", init_node)])

        assert chunks == []

    def test_chunk_ids_are_stable_across_calls(self) -> None:
        content = "def foo(): pass\n"
        p = parsed(content)
        node = node_for(content, "def foo(): pass")
        caps = [cap("definition.function", node)]

        assert self.strategy.chunk(p, caps)[0].id == self.strategy.chunk(p, caps)[0].id

    def test_chunk_id_is_stable_for_same_location_and_text(self) -> None:
        content = "def foo(): pass\n"
        p = parsed(content)
        node = node_for(content, "def foo(): pass")

        chunks = self.strategy.chunk(p, [cap("definition.function", node)])

        assert chunks[0].id == cid(chunks[0].text)

    def test_returns_empty_list_for_no_captures(self) -> None:
        assert self.strategy.chunk(parsed("# nothing\n"), []) == []
