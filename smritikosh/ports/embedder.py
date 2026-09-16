"""The embedding contract -- implementations live in adapters/embedder/."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod

__all__ = ["Embedder"]

#: Fallback context window for an adapter that does not declare one.  512 is
#: the smallest window in common use (BERT-family encoders), so assuming it
#: never over-fills a model — it only splits more than strictly necessary.
_CONSERVATIVE_MAX_TOKENS = 512


class Embedder(ABC):
    """Turns text into dense vectors for indexing and search."""

    @property
    @abstractmethod
    def dims(self) -> int:
        """Vector width, which drives VectorStore.setup()."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identity; a change invalidates every cached embedding."""

    @property
    def max_tokens(self) -> int:
        """Tokens the model reads before it silently truncates the rest.

        Chunking uses this as a byte budget so no chunk is longer than what
        the model will actually encode.  Encoders drop the overflow without
        raising, so a chunk over the limit is stored whole but embedded from
        its prefix only — the tail becomes unsearchable.  Adapters that know
        their context window should override this.
        """
        return _CONSERVATIVE_MAX_TOKENS

    def count_tokens(self, text: str) -> int:
        """Count model tokens, conservatively when no tokenizer is exposed."""
        return math.ceil(len(text) / 2.5) + 2

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_document(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self.encode_documents, [text])
        return vectors[0]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self.encode_queries, [text])
        return vectors[0]

    async def embed_documents_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.encode_documents, texts)
