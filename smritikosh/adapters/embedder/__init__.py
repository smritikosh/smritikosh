"""Embedding adapter selection and restoration."""

from __future__ import annotations

from smritikosh.adapters.embedder.resolver import (
    EmbedderUnavailableError,
    resolve_embedder,
    select_embedder,
)

__all__ = [
    "EmbedderUnavailableError",
    "resolve_embedder",
    "select_embedder",
]
