"""Tests for automatic embedder selection and exact identity restoration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from smritikosh.adapters.embedder import (
    EmbedderUnavailableError,
    resolve_embedder,
    resolver,
    select_embedder,
)
from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder
from smritikosh.adapters.embedder.mps import MpsEmbedder


def _set_mps_available(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
) -> None:
    mps_provider, portable_provider = resolver._PROVIDERS
    monkeypatch.setattr(
        resolver,
        "_PROVIDERS",
        (replace(mps_provider, available=lambda: available), portable_provider),
    )


def test_should_select_mps_when_auto_and_hardware_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps_available(monkeypatch, True)

    embedder = select_embedder()

    assert isinstance(embedder, MpsEmbedder)


def test_should_fall_back_to_portable_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps_available(monkeypatch, False)

    embedder = select_embedder()

    assert isinstance(embedder, FastEmbedEmbedder)


def test_should_restore_adapter_matching_stored_identity() -> None:
    expected = FastEmbedEmbedder()

    restored = resolve_embedder(expected.identity)

    assert isinstance(restored, FastEmbedEmbedder)


def test_should_reject_accelerated_index_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_mps_available(monkeypatch, False)

    with pytest.raises(EmbedderUnavailableError, match="Apple GPU"):
        resolve_embedder(MpsEmbedder().identity)


def test_should_reject_unknown_vector_space() -> None:
    from smritikosh.ports.embedder import EmbeddingIdentity

    identity = EmbeddingIdentity("unknown", "unknown-v1", 768)

    with pytest.raises(EmbedderUnavailableError, match="No installed embedder"):
        resolve_embedder(identity)
