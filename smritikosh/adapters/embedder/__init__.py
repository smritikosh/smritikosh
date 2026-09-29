"""smritikosh.adapters.embedder — embedder adapters and factory.

Public API
----------
All symbols are importable directly from ``smritikosh.adapters.embedder``::

    from smritikosh.adapters.embedder import make_embedder, EMBEDDER_CHOICES

``FastEmbedEmbedder`` is also importable from its own submodule::

    from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smritikosh.ports.embedder import Embedder

__all__ = ["EMBEDDER_CHOICES", "EMBEDDER_ENV_VAR", "make_embedder"]

#: Canonical list of supported embedder names.
#: Keep in sync with ``make_embedder``.
EMBEDDER_CHOICES: list[str] = ["fastembed", "mps"]

#: Selects the backend when no name is passed.  An env var rather than a flag
#: alone because *every* entry point must agree — indexing with one backend
#: and searching with another compares vectors from different spaces.
EMBEDDER_ENV_VAR = "SMRITIKOSH_EMBEDDER"

DEFAULT_EMBEDDER = "fastembed"


def make_embedder(name: str | None = None) -> Embedder:
    """Instantiate the :class:`~smritikosh.ports.embedder.Embedder` for *name*.

    *name* is normalised to lowercase so ``"FastEmbed"`` and ``"FASTEMBED"``
    both work.  ``None`` reads :data:`EMBEDDER_ENV_VAR`, falling back to
    ``fastembed``.

    ``fastembed``
        CodeRankEmbed as an INT8 ONNX export on CPU.  Runs anywhere, no extra
        install — the default.
    ``mps``
        The same model in fp32 on the Apple GPU via PyTorch.  Roughly 2x
        faster on Apple silicon, needs ``pip install 'smritikosh[mps]'``.

    The two produce *different* vectors for the same text, so an index has to
    be built and queried with one of them, not both.

    Raises
    ------
    ValueError
        If *name* is not one of :data:`EMBEDDER_CHOICES`.
    """
    if name is None:
        name = os.environ.get(EMBEDDER_ENV_VAR) or DEFAULT_EMBEDDER
    name = name.lower()

    if name == "fastembed":
        from smritikosh.adapters.embedder.fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder()
    if name == "mps":
        from smritikosh.adapters.embedder.mps import MpsEmbedder

        return MpsEmbedder()
    raise ValueError(
        f"Unknown embedder {name!r}. Choose: {', '.join(EMBEDDER_CHOICES)}."
    )
