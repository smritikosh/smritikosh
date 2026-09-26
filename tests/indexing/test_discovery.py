"""Tests for source-file discovery and its three gates."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from smritikosh.indexing.discovery import iter_source_files
from smritikosh.indexing.file_router import FileRouter
from smritikosh.ports.file_source import FileSource
from tests.indexing.conftest import FakeStrategy

STRATEGY = FakeStrategy()


class FakeFileSource(FileSource):
    """In-memory source that records which files were actually read."""

    def __init__(self, files: dict[str, str], ignored: set[str] | None = None) -> None:
        self._files = files
        self._ignored = ignored or set()
        self.reads: list[str] = []

    def iter_paths(self) -> Iterator[str]:
        return iter(self._files)

    def read_text(self, path: str) -> str:
        self.reads.append(path)
        return self._files[path]

    def is_ignored(self, path: str) -> bool:
        return path in self._ignored


@pytest.fixture
def router() -> FileRouter:
    router = FileRouter()
    router.register_extension(".py", "python", STRATEGY)
    router.register_extension(".json", "json", STRATEGY)
    return router


def _paths(source: FileSource, router: FileRouter) -> list[str]:
    return [f.path for f in iter_source_files(source, router)]


# ── Gates ─────────────────────────────────────────────────────────────────────


def test_should_drop_files_under_excluded_directories(router: FileRouter) -> None:
    source = FakeFileSource(
        {
            "src/app.py": "keep",
            ".venv/lib/dep.py": "drop",
            "node_modules/pkg/index.py": "drop",
            "build/out.py": "drop",
        }
    )

    assert _paths(source, router) == ["src/app.py"]


def test_should_drop_numbered_virtualenvs_and_claude_config(
    router: FileRouter,
) -> None:
    """``.venv2`` and ``.claude`` are not source, and both inflate the file count."""
    source = FakeFileSource(
        {
            "src/app.py": "keep",
            "src/venv2.py": "keep",
            ".venv2/lib/site-packages/pkg/__init__.py": "drop",
            "svc/venv2/lib/dep.py": "drop",
            ".claude/skills/oncall/helper.py": "drop",
        }
    )

    assert _paths(source, router) == ["src/app.py", "src/venv2.py"]


def test_should_drop_files_the_source_reports_ignored(router: FileRouter) -> None:
    source = FakeFileSource(
        {"src/app.py": "keep", "secrets/key.py": "drop"},
        ignored={"secrets/key.py"},
    )

    assert _paths(source, router) == ["src/app.py"]


def test_should_drop_files_the_router_rejects(router: FileRouter) -> None:
    source = FakeFileSource(
        {
            "config/eval_config.json": "keep",
            "package-lock.json": "drop",
            "assets/logo.png": "drop",
        }
    )

    assert _paths(source, router) == ["config/eval_config.json"]


def test_should_skip_the_ignore_gate_when_source_has_no_rules(
    router: FileRouter,
) -> None:
    """A source that does not override is_ignored inherits False from the port."""

    class MinimalFileSource(FileSource):
        def iter_paths(self) -> Iterator[str]:
            return iter(["src/app.py"])

        def read_text(self, path: str) -> str:  # noqa: ARG002
            return "content"

    assert _paths(MinimalFileSource(), router) == ["src/app.py"]


# ── Output ────────────────────────────────────────────────────────────────────


def test_should_copy_route_config_onto_the_source_file(router: FileRouter) -> None:
    source = FakeFileSource({"src/app.py": "import os\n"})

    (file,) = list(iter_source_files(source, router))

    assert file.path == "src/app.py"
    assert file.language == "python"
    assert file.content == "import os\n"
    assert file.has_tags_scm is True
    assert file.strategy is STRATEGY


def test_should_not_read_files_that_fail_a_gate(router: FileRouter) -> None:
    source = FakeFileSource(
        {
            "src/app.py": "keep",
            "package-lock.json": "drop",
            ".venv/dep.py": "drop",
        }
    )

    list(iter_source_files(source, router))

    assert source.reads == ["src/app.py"]
