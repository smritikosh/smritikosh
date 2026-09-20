"""Tests for SectionChunkingStrategy."""

from __future__ import annotations

from smritikosh.indexing.strategies._helpers import build_chunk
from smritikosh.indexing.strategies.section import (
    SectionChunkingStrategy,
    _merge_markdown_fragments,
)
from tests.indexing.conftest import cap, node_for, parsed


class TestSectionChunkingStrategy:
    strategy = SectionChunkingStrategy()

    def test_mode_name(self) -> None:
        assert self.strategy.mode_name == "section"

    def test_one_chunk_per_capture(self) -> None:
        content = "# Heading\n\nSome text.\n"
        p = parsed(content, "doc.md")
        node = node_for(content, "# Heading")

        chunks = self.strategy.chunk(p, [cap("definition.section", node)])

        assert len(chunks) == 1
        assert chunks[0].chunk_kind == "section"

    def test_multiple_captures_produce_multiple_chunks(self) -> None:
        content = "# A\n\ntext\n\n# B\n\nmore\n"
        p = parsed(content, "doc.md")
        n1 = node_for(content, "# A")
        n2 = node_for(content, "# B")

        chunks = self.strategy.chunk(
            p, [cap("definition.section", n1), cap("definition.section", n2)]
        )

        assert len(chunks) == 2

    def test_no_captures_falls_back_to_whole_file(self) -> None:
        content = "just some text\n"
        p = parsed(content, "doc.md")

        chunks = self.strategy.chunk(p, [])

        assert len(chunks) == 1
        assert chunks[0].text == content
        assert chunks[0].chunk_kind == "section"

    def test_empty_file_no_captures_returns_one_chunk(self) -> None:
        p = parsed("", "empty.md")

        chunks = self.strategy.chunk(p, [])

        assert len(chunks) == 1
        assert chunks[0].start_line == 1


class TestJsonSectionChunking:
    """JSON: depth chosen by size, plus a key-path label on every chunk."""

    def test_should_keep_a_small_config_flat(self) -> None:
        """Top-level sections that fit are emitted; their children are skipped.

        Both keys here are below the fragment floor, so they merge into one
        run rather than becoming two chunks too small to retrieve — see
        test_should_not_merge_a_section_that_is_already_big_enough for the
        stand-alone case.
        """
        content = '{\n  "a": {"inner": 1},\n  "b": 2\n}'
        p = parsed(content, "c.json", language="json")
        captures = [
            cap("definition.section", node_for(content, '"a": {"inner": 1}'), key="a"),
            cap("definition.subsection", node_for(content, '"inner": 1'), key="inner"),
            cap("definition.section", node_for(content, '"b": 2'), key="b"),
        ]

        chunks = SectionChunkingStrategy(max_chars=500).chunk(p, captures)

        assert len(chunks) == 1
        assert chunks[0].text.splitlines()[0] == "// a+b"
        assert '"inner": 1' in chunks[0].text  # child travelled with its parent

    def test_should_not_merge_a_section_that_is_already_big_enough(self) -> None:
        """A capture at or above the floor stands alone with its own path."""
        big_a = '"a": "' + "x" * 120 + '"'
        big_b = '"b": "' + "y" * 120 + '"'
        content = "{\n  " + big_a + ",\n  " + big_b + "\n}"
        p = parsed(content, "c.json", language="json")
        captures = [
            cap("definition.section", node_for(content, big_a), key="a"),
            cap("definition.section", node_for(content, big_b), key="b"),
        ]

        chunks = SectionChunkingStrategy(max_chars=500).chunk(p, captures)

        assert [c.text.splitlines()[0] for c in chunks] == ["// a", "// b"]

    def test_should_descend_into_an_oversized_section(self) -> None:
        """An oversized parent is discarded in favour of the keys inside it."""
        inner_a = '"inner_a": "' + "x" * 200 + '"'
        inner_b = '"inner_b": "' + "y" * 200 + '"'
        outer = f'"outer": {{{inner_a}, {inner_b}}}'
        content = "{\n  " + outer + "\n}"
        p = parsed(content, "c.json", language="json")
        captures = [
            cap("definition.section", node_for(content, outer), key="outer"),
            cap("definition.subsection", node_for(content, inner_a), key="inner_a"),
            cap("definition.subsection", node_for(content, inner_b), key="inner_b"),
        ]

        chunks = SectionChunkingStrategy(max_chars=300).chunk(p, captures)

        first_lines = [c.text.splitlines()[0] for c in chunks]
        assert "// outer.inner_a" in first_lines
        assert "// outer.inner_b" in first_lines
        assert not any(line == "// outer" for line in first_lines)

    def test_should_emit_an_oversized_leaf_rather_than_dropping_it(self) -> None:
        """No deeper capture exists, so line-window splitting is the backstop."""
        big = '"solo": "' + "z" * 900 + '"'
        content = "{\n  " + big + "\n}"
        p = parsed(content, "c.json", language="json")

        chunks = SectionChunkingStrategy(max_chars=300).chunk(
            p, [cap("definition.section", node_for(content, big), key="solo")]
        )

        assert chunks
        assert all("z" in c.text for c in chunks)

    def test_should_not_label_markdown_sections(self) -> None:
        """A heading is already natural language; a path would add noise."""
        content = "# Heading\n\nSome text.\n"
        p = parsed(content, "doc.md", language="markdown")

        heading = cap(
            "definition.section", node_for(content, "# Heading"), key="Heading"
        )

        chunks = SectionChunkingStrategy(max_chars=500).chunk(p, [heading])

        assert not chunks[0].text.startswith("//")

    def test_should_partition_nested_markdown_without_repeating_body_text(self) -> None:
        parent = "parent detail " * 12
        child = "child detail " * 12
        sibling = "sibling detail " * 12
        content = (
            f"# Parent\n\n{parent}\n## Child\n\n{child}\n## Sibling\n\n{sibling}\n"
        )
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap("definition.section", node_for(content, "# Parent"), key="Parent"),
            cap("definition.section", node_for(content, "## Child"), key="Child"),
            cap("definition.section", node_for(content, "## Sibling"), key="Sibling"),
        ]

        chunks = SectionChunkingStrategy(max_chars=500).chunk(p, captures)

        combined = "\n".join(chunk.text for chunk in chunks)
        assert combined.count(parent.strip()) == 1
        assert combined.count(child.strip()) == 1
        assert combined.count(sibling.strip()) == 1
        child_chunk = next(chunk for chunk in chunks if child.strip() in chunk.text)
        assert child_chunk.text.startswith("# Parent\n## Child\n")

    def test_should_split_a_long_markdown_paragraph_within_budget(self) -> None:
        paragraph = "retrievable documentation sentence. " * 100
        content = f"# Long section\n\n{paragraph}\n"
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap(
                "definition.section",
                node_for(content, "# Long section"),
                key="Long section",
            )
        ]

        chunks = SectionChunkingStrategy(max_chars=200).chunk(p, captures)

        assert len(chunks) > 1
        assert all(chunk.text for chunk in chunks)
        assert all(len(chunk.text) <= 200 for chunk in chunks)

    def test_should_keep_a_markdown_table_row_intact_when_it_fits(self) -> None:
        rows = "\n".join(
            f"| capability {index} | description {index} |" for index in range(20)
        )
        content = f"# Capabilities\n\n{rows}\n"
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap(
                "definition.section",
                node_for(content, "# Capabilities"),
                key="Capabilities",
            )
        ]

        chunks = SectionChunkingStrategy(max_chars=200).chunk(p, captures)

        emitted_lines = [line for chunk in chunks for line in chunk.text.splitlines()]
        for index in range(20):
            assert (
                emitted_lines.count(f"| capability {index} | description {index} |")
                == 1
            )

    def test_should_not_emit_a_fenced_code_block_twice(self) -> None:
        code = "```python\nprint('unique marker')\n```"
        content = f"# Example\n\n{code}\n"
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap("definition.section", node_for(content, "# Example"), key="Example"),
            cap("definition.code_block", node_for(content, code), key="python"),
        ]

        chunks = SectionChunkingStrategy(max_chars=500).chunk(p, captures)

        assert "\n".join(chunk.text for chunk in chunks).count("unique marker") == 1

    def test_should_not_emit_a_heading_only_parent_as_its_own_chunk(self) -> None:
        content = "# Parent\n\n## Child\n\nUseful child content.\n"
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap("definition.section", node_for(content, "# Parent"), key="Parent"),
            cap("definition.section", node_for(content, "## Child"), key="Child"),
        ]

        chunks = SectionChunkingStrategy(max_chars=500).chunk(p, captures)

        assert len(chunks) == 1
        assert chunks[0].text.startswith("# Parent\n## Child\n")

    def test_should_pack_complete_blocks_using_the_model_token_counter(self) -> None:
        blocks = [
            " ".join(f"purpose{i}" for i in range(10)),
            " ".join(f"table{i}" for i in range(15)),
            " ".join(f"state{i}" for i in range(15)),
            " ".join(f"detail{i}" for i in range(20)),
        ]
        content = "# Lifecycle\n\n" + "\n\n".join(blocks)
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap(
                "definition.section",
                node_for(content, "# Lifecycle"),
                key="Lifecycle",
            )
        ]

        def count_words(text: str) -> int:
            return len(text.split())

        chunks = SectionChunkingStrategy(
            max_chars=20,
            max_tokens=50,
            token_counter=count_words,
        ).chunk(p, captures)

        assert len(chunks) == 2
        assert all(count_words(chunk.text) <= 47 for chunk in chunks)
        assert blocks[0] in chunks[0].text
        assert blocks[2] in chunks[0].text
        assert blocks[3] in chunks[1].text

    def test_should_cap_the_breadcrumb_against_the_token_budget(self) -> None:
        """A deep hierarchy of long headings must leave room for the body.

        Token mode has no character budget for the breadcrumb to inherit, so
        without a cap of its own the breadcrumb grows with nesting depth until
        every chunk is heading text, splits one character at a time, and still
        lands over the window it was supposed to respect.
        """
        heading = "Operational " * 60
        content = "".join(
            f"{'#' * level} {heading} L{level}\n\nBody prose at level {level}.\n\n"
            for level in range(1, 7)
        )
        p = parsed(content, "deep.md", language="markdown")
        captures = [
            cap(
                "definition.section",
                node_for(content, f"{'#' * level} {heading} L{level}"),
                key=f"{heading} L{level}",
            )
            for level in range(1, 7)
        ]

        chunks = SectionChunkingStrategy(
            max_chars=2_000,
            max_tokens=512,
            token_counter=len,
        ).chunk(p, captures)

        assert len(chunks) == 6, "one chunk per section, not one per character"
        assert all(len(chunk.text) <= 512 for chunk in chunks)
        for chunk in chunks:
            breadcrumb = chunk.text.split("\n\n", 1)[0]
            assert len(breadcrumb) <= 512 // 3

    def test_should_not_merge_a_small_section_into_the_next_one(self) -> None:
        """A merged pair reports the later heading, losing the earlier one.

        Fragments are merged across the whole file, so a section below the
        floor would absorb the section after it and be reported under that
        section's symbol -- attributing its prose to the wrong heading in the
        outline and in every citation built from it.
        """
        content = (
            "# Handbook\n\n"
            "## Ledger\n\nShort.\n\n"
            "## Scope\n\nScope covers what the tool reads and writes.\n"
        )
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap("definition.section", node_for(content, "# Handbook"), key="Handbook"),
            cap("definition.section", node_for(content, "## Ledger"), key="Ledger"),
            cap("definition.section", node_for(content, "## Scope"), key="Scope"),
        ]

        chunks = SectionChunkingStrategy(max_chars=2_000).chunk(p, captures)

        assert [chunk.symbol for chunk in chunks] == [
            "Handbook > Ledger",
            "Handbook > Scope",
        ]
        assert "Short." in chunks[0].text
        assert "Scope covers" not in chunks[0].text

    def test_should_still_merge_two_fragments_of_one_section(self) -> None:
        """Keeping headings apart must not switch fragment merging off."""
        fragments = [
            build_chunk("doc.md", "# Only\n\nfirst", "section", 3, 3, "Only"),
            build_chunk("doc.md", "# Only\n\nsecond", "section", 5, 5, "Only"),
        ]

        merged = _merge_markdown_fragments(fragments, 2_000)

        assert len(merged) == 1
        assert merged[0].symbol == "Only"
        assert (merged[0].start_line, merged[0].end_line) == (3, 5)

    def test_should_measure_each_block_once_while_packing(self) -> None:
        """The pending run is measured once, when the block that grew it arrives.

        Re-deriving it on every block re-tokenizes the whole buffer each time,
        and a real tokenizer pass is the most expensive thing indexing does.
        """
        blocks = [
            f"Paragraph {index} with a little filler prose." for index in range(20)
        ]
        content = "# Section\n\n" + "\n\n".join(blocks) + "\n"
        p = parsed(content, "doc.md", language="markdown")
        captures = [
            cap("definition.section", node_for(content, "# Section"), key="Section")
        ]
        measured: list[str] = []

        def counting(text: str) -> int:
            measured.append(text)
            return len(text.split())

        SectionChunkingStrategy(
            max_chars=2_000,
            max_tokens=5_000,
            token_counter=counting,
        ).chunk(p, captures)

        # Two fixed calls measure the breadcrumb — once to cap it, once to
        # prime the running total — and then one call admits each block.
        # Nothing here flushes, so any further call is the pending run being
        # measured a second time, which is what re-tokenizes the whole buffer.
        assert len(measured) <= len(blocks) + 2

    def test_should_split_one_oversized_block_by_exact_token_count(self) -> None:
        paragraph = " ".join(f"token{i}" for i in range(100))
        content = f"# Long\n\n{paragraph}"
        p = parsed(content, "doc.md", language="markdown")
        captures = [cap("definition.section", node_for(content, "# Long"), key="Long")]

        def count_words(text: str) -> int:
            return len(text.split())

        chunks = SectionChunkingStrategy(
            max_tokens=20,
            token_counter=count_words,
        ).chunk(p, captures)

        assert len(chunks) > 1
        assert all(count_words(chunk.text) <= 19 for chunk in chunks)

    def test_should_reserve_budget_for_the_key_path_prefix(self) -> None:
        """The prefix eats the window, so the budget must account for it.

        Uses a multi-line body: a single over-long line is deliberately never
        broken (see split_oversized), so it is the wrong fixture for a budget
        assertion.
        """
        body = '"k": [\n' + "\n".join(f'    "v{i}",' for i in range(60)) + "\n  ]"
        content = "{\n  " + body + "\n}"
        p = parsed(content, "c.json", language="json")

        chunks = SectionChunkingStrategy(max_chars=200).chunk(
            p, [cap("definition.section", node_for(content, body), key="k")]
        )

        assert len(chunks) > 1
        assert all(len(c.text) <= 200 for c in chunks)
        assert all(c.text.startswith("// k\n") for c in chunks)

    def test_should_keep_an_unsplittable_line_whole(self) -> None:
        """One giant line is a literal; cutting it would corrupt the snippet."""
        body = '"k": "' + "q" * 400 + '"'
        content = "{\n  " + body + "\n}"
        p = parsed(content, "c.json", language="json")

        chunks = SectionChunkingStrategy(max_chars=200).chunk(
            p, [cap("definition.section", node_for(content, body), key="k")]
        )

        assert len(chunks) == 1
        assert "q" * 400 in chunks[0].text

    def test_should_not_descend_into_fragment_sized_children(self) -> None:
        """Tiny keys retrieve worse than one coarse chunk, so keep the parent."""
        kids = ", ".join(f'"k{i}": {i}' for i in range(40))
        outer = f'"outer": {{{kids}}}'
        content = "{\n  " + outer + "\n}"
        p = parsed(content, "c.json", language="json")
        captures = [cap("definition.section", node_for(content, outer), key="outer")]
        captures += [
            cap("definition.subsection", node_for(content, f'"k{i}": {i}'), key=f"k{i}")
            for i in range(40)
        ]

        chunks = SectionChunkingStrategy(max_chars=200).chunk(p, captures)

        # Every chunk is labelled with the parent, not the fragment-sized keys.
        assert all(c.text.startswith("// outer\n") for c in chunks)
