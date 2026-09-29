"""Tests for smritikosh.adapters.embedder.make_embedder factory.

These tests exercise the *adapter layer* in isolation — no Click, no CLI.
The CLI wrapper (_make_embedder in cli.py) is tested separately in test_cli.py.
"""

from __future__ import annotations

import pytest

from smritikosh.adapters.embedder import (
    EMBEDDER_CHOICES,
    EMBEDDER_ENV_VAR,
    make_embedder,
)

# ── EMBEDDER_CHOICES ──────────────────────────────────────────────────────────


def test_embedder_choices_contains_all_supported_backends() -> None:
    assert set(EMBEDDER_CHOICES) == {"fastembed", "mps"}


def test_embedder_choices_is_a_list_of_strings() -> None:
    assert isinstance(EMBEDDER_CHOICES, list)
    assert all(isinstance(c, str) for c in EMBEDDER_CHOICES)


# ── make_embedder — fastembed ─────────────────────────────────────────────────


def test_make_embedder_fastembed_returns_fast_embed_embedder() -> None:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    assert isinstance(make_embedder("fastembed"), FastEmbedEmbedder)


def test_make_embedder_defaults_to_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    monkeypatch.delenv(EMBEDDER_ENV_VAR, raising=False)

    assert isinstance(make_embedder(), FastEmbedEmbedder)


# ── make_embedder — mps ───────────────────────────────────────────────────────


def test_make_embedder_mps_returns_mps_embedder() -> None:
    from smritikosh.adapters.embedder.mps import MpsEmbedder

    assert isinstance(make_embedder("mps"), MpsEmbedder)


def test_make_embedder_mps_does_not_load_torch() -> None:
    """Constructing must stay cheap — the GPU model loads on first encode."""
    assert make_embedder("mps")._model is None  # type: ignore[attr-defined]


# ── make_embedder — env var ───────────────────────────────────────────────────


def test_env_var_selects_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every entry point must agree, so the choice lives in the environment."""
    from smritikosh.adapters.embedder.mps import MpsEmbedder

    monkeypatch.setenv(EMBEDDER_ENV_VAR, "mps")

    assert isinstance(make_embedder(), MpsEmbedder)


def test_explicit_name_beats_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    monkeypatch.setenv(EMBEDDER_ENV_VAR, "mps")

    assert isinstance(make_embedder("fastembed"), FastEmbedEmbedder)


def test_blank_env_var_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    monkeypatch.setenv(EMBEDDER_ENV_VAR, "")

    assert isinstance(make_embedder(), FastEmbedEmbedder)


def test_make_embedder_fastembed_is_case_insensitive() -> None:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    for variant in ("FastEmbed", "FASTEMBED", "fAsTeMbEd"):
        assert isinstance(make_embedder(variant), FastEmbedEmbedder), variant


# ── make_embedder — unknown name ──────────────────────────────────────────────


def test_make_embedder_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown embedder"):
        make_embedder("bert")


def test_make_embedder_value_error_mentions_valid_choices() -> None:
    with pytest.raises(ValueError) as exc_info:
        make_embedder("gpt")
    message = str(exc_info.value)
    for choice in EMBEDDER_CHOICES:
        assert choice in message
