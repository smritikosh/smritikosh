"""Select embedding adapters from user intent and stored vector-space identity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from smritikosh.ports.embedder import Embedder, EmbeddingIdentity

__all__ = [
    "EmbedderUnavailableError",
    "resolve_embedder",
    "select_embedder",
]


class EmbedderUnavailableError(RuntimeError):
    """Raised when no installed adapter can serve a requested vector space."""


@dataclass(frozen=True)
class _Provider:
    available: Callable[[], bool]
    create: Callable[[], Embedder]


def _fastembed_available() -> bool:
    return True


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def _fastembed() -> Embedder:
    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

    return FastEmbedEmbedder()


def _mps() -> Embedder:
    from smritikosh.adapters.embedder.mps import MpsEmbedder

    return MpsEmbedder()


_PROVIDERS = (
    _Provider(_mps_available, _mps),
    _Provider(_fastembed_available, _fastembed),
)


def select_embedder() -> Embedder:
    """Select the best available adapter for a new index."""
    for provider in _PROVIDERS:
        if provider.available():
            return provider.create()
    raise EmbedderUnavailableError("No embedding runtime is available.")


def resolve_embedder(identity: EmbeddingIdentity) -> Embedder:
    """Restore an available adapter compatible with an existing index."""
    for provider in _PROVIDERS:
        candidate = provider.create()
        if candidate.identity != identity:
            continue
        if provider.available():
            return candidate
        raise EmbedderUnavailableError(
            "This index requires Apple GPU acceleration, which is unavailable "
            "on this machine. Rebuild it with the portable profile."
        )
    raise EmbedderUnavailableError(
        f"No installed embedder supports vector space {identity.vector_space!r}."
    )
