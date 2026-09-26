"""Tests for tree-sitter capture extraction."""

from collections.abc import Iterator
from operator import attrgetter
from unittest.mock import ANY

import pytest
from tree_sitter import Parser, Query
from tree_sitter_language_pack import get_language

import smritikosh.indexing.extractor as extractor
from smritikosh.models import Capture, ParsedFile


@pytest.fixture(autouse=True)
def clear_query_cache() -> Iterator[None]:
    """Keep the process-local query cache isolated between tests."""
    extractor._QUERY_CACHE.clear()
    yield
    extractor._QUERY_CACHE.clear()


def _parse(language: str, source: str, *, path: str) -> ParsedFile:
    tree = Parser(get_language(language)).parse(source.encode("utf-8"))
    return ParsedFile(path=path, language=language, content=source, tree=tree)


@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        ("python", "def run():\n    pass\n", ("definition.function", "run")),
        ("java", "class Client {}\n", ("definition.class", "Client")),
        ("kotlin", "fun run() {}\n", ("definition.function", "run")),
        (
            "typescript",
            "function run(): void {}\n",
            ("definition.function", "run"),
        ),
        ("javascript", "function run() {}\n", ("definition.function", "run")),
        (
            "markdown",
            "# Introduction\n\nBody.\n",
            ("definition.section", "Introduction"),
        ),
        (
            "json",
            '{"name": "demo"}\n',
            ("definition.section", "name"),
        ),
        (
            "yaml",
            "kind: CronJob\n",
            ("definition.section", "kind"),
        ),
    ],
)
def test_should_extract_expected_definition_for_each_query_language(
    language: str,
    source: str,
    expected: tuple[str, str],
) -> None:
    parsed = _parse(language, source, path=f"example.{language}")

    captures = extractor.extract_file(parsed, has_tags_scm=True)

    assert captures == [
        Capture(
            capture_name=expected[0],
            node=ANY,
            name=expected[1],
            path=f"example.{language}",
        )
    ]


def test_should_return_empty_when_file_has_no_tags_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = ParsedFile(
        path="pyproject.toml",
        language="toml",
        content="[project]\n",
        tree=object(),
    )

    def fail_if_called(language: str) -> Query:
        pytest.fail(f"query loading should be skipped for {language}")

    monkeypatch.setattr(extractor, "_load_query", fail_if_called)

    assert extractor.extract_file(parsed, has_tags_scm=False) == []


def test_should_return_complete_capture_metadata_for_python_definitions() -> None:
    source = """
MAX_RETRIES: Final[int] = 3

class Status(str, Enum):
    ACTIVE = "active"

class Client:
    async def fetch(self):
        pass
"""
    parsed = _parse("python", source, path="src/client.py")

    captures = extractor.extract_file(parsed, has_tags_scm=True)

    expected = [
        Capture(
            capture_name="definition.constant",
            node=ANY,
            name="MAX_RETRIES",
            path="src/client.py",
        ),
        Capture(
            capture_name="definition.class",
            node=ANY,
            name="Status",
            path="src/client.py",
        ),
        Capture(
            capture_name="definition.enum",
            node=ANY,
            name="Status",
            path="src/client.py",
        ),
        Capture(
            capture_name="definition.class_constant",
            node=ANY,
            name="ACTIVE",
            path="src/client.py",
        ),
        Capture(
            capture_name="definition.class",
            node=ANY,
            name="Client",
            path="src/client.py",
        ),
        Capture(
            capture_name="definition.method",
            node=ANY,
            name="fetch",
            path="src/client.py",
        ),
    ]
    by_definition = attrgetter("capture_name", "name")
    assert sorted(captures, key=by_definition) == sorted(expected, key=by_definition)
    assert all(capture.node.text for capture in captures)


def test_should_reuse_compiled_query_when_language_is_already_cached() -> None:
    first = extractor._load_query("python")

    second = extractor._load_query("python")

    assert second is first
    assert list(extractor._QUERY_CACHE) == ["python"]


def test_should_register_extractor_logic_for_memo_invalidation() -> None:
    assert hasattr(extractor.extract_file, "_logic_fingerprint")
