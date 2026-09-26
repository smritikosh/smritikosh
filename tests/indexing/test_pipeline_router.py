"""Tests for _build_router — FileRouter factory and @functools.cache behaviour."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from smritikosh.indexing.file_router import FileRouter
from smritikosh.indexing.pipeline._router import _build_router


@pytest.fixture(autouse=True)
def clear_cache() -> Iterator[None]:
    """Isolate each test from the process-level @functools.cache."""
    _build_router.cache_clear()
    yield
    _build_router.cache_clear()


# ── Return type ───────────────────────────────────────────────────────────────


def test_returns_a_file_router() -> None:
    assert isinstance(_build_router(), FileRouter)


# ── @functools.cache ──────────────────────────────────────────────────────────


def test_same_instance_returned_on_repeated_calls() -> None:
    assert _build_router() is _build_router()


# ── Every registered extension ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "src/app.py",
        "client.ts",
        "component.tsx",
        "index.js",
        "page.jsx",
        "Service.java",
        "Main.kt",
        "build.kts",
        "README.md",
        "guide.mdx",
        "pyproject.toml",
        "config.json",
        "deploy/app.yaml",
        "deploy/app.yml",
    ],
)
def test_known_extension_is_routed(path: str) -> None:
    assert _build_router().route(path) is not None


def test_unknown_extension_returns_none() -> None:
    assert _build_router().route("data.csv") is None


# ── TOML has no tags.scm ──────────────────────────────────────────────────────


def test_yaml_has_tags_scm_true() -> None:
    config = _build_router().route("deploy/app.yaml")
    assert config is not None
    assert config.language == "yaml"
    assert config.has_tags_scm is True


def test_toml_has_tags_scm_false() -> None:
    config = _build_router().route("pyproject.toml")
    assert config is not None
    assert config.has_tags_scm is False


# ── JSON exclude filter wired ─────────────────────────────────────────────────


def test_package_lock_json_is_excluded() -> None:
    assert _build_router().route("package-lock.json") is None


def test_regular_json_is_indexed() -> None:
    assert _build_router().route("config/settings.json") is not None
