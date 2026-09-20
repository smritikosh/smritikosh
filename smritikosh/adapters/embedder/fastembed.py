"""Local embedding backend — fastembed + ONNX Runtime, no PyTorch, CPU-only.

Uses :data:`~smritikosh.constants.DEFAULT_MODEL` (CodeRankEmbed), a 768-dim
code-retrieval encoder.  Encoder models process all tokens in parallel,
which is what makes them viable on CPU.

CodeRankEmbed is not in fastembed's catalog, so the adapter registers an
INT8 ONNX export on first use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smritikosh.constants import DEFAULT_MODEL
from smritikosh.ports.embedder import Embedder

if TYPE_CHECKING:
    from fastembed import TextEmbedding

__all__ = ["FastEmbedEmbedder"]

logger = logging.getLogger(__name__)

#: CodeRankEmbed's own instruction.  Its model card marks this as *required*:
#: without it the query lands in document space and retrieval degrades badly.
_CODE_INSTRUCTION = "Represent this query for searching relevant code: "

# INT8 build chosen over FP32 (139 MB vs 548 MB) using the `reduce_range=True`
# export — naive INT8 produces degenerate embeddings on pre-VNNI x86 CPUs.
# This export's tokenizer truncates at 512 despite the model card's 8192;
# max_tokens reads it.
_MODEL_REPO = "mrsladoje/CodeRankEmbed-onnx-int8"
_MODEL_DIM = 768
_MODEL_FILE = "onnx/model.onnx"


class FastEmbedEmbedder(Embedder):
    """Embeds via fastembed + ONNX Runtime — no PyTorch, runs on any CPU.

    Parameters
    ----------
    threads:
        Number of ONNX intra-op threads.  ``None`` lets ONNX Runtime
        choose (usually one thread per physical core).
    """

    def __init__(self, *, threads: int | None = None) -> None:
        self._model_name = DEFAULT_MODEL
        self._threads = threads
        self._model: TextEmbedding | None = None
        self._max_tokens: int | None = None
        self._query_prefix = _CODE_INSTRUCTION

    # ── Embedder protocol ─────────────────────────────────────────────────────

    @property
    def dims(self) -> int:
        """Embedding width.  Known from the model spec — no ONNX load."""
        return _MODEL_DIM

    @property
    def model_id(self) -> str:
        return f"{self._model_name}:{self.dims}"

    @property
    def max_tokens(self) -> int:
        """The tokenizer's own truncation limit — the exact cutoff to respect.

        fastembed's static catalog carries no context-window field, so this
        reads the limit off the loaded tokenizer, where it is authoritative:
        ``truncation["max_length"]`` *is* the setting that silently drops the
        tail of an over-long chunk.  Cached because the chunker asks per file.
        """
        if self._max_tokens is None:
            truncation = self._load().model.tokenizer.truncation or {}
            self._max_tokens = int(truncation.get("max_length") or super().max_tokens)
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens with the exact tokenizer used for document embedding."""
        return len(self._load().model.tokenizer.encode(text).ids)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        # fastembed's query_embed() is a no-op alias for embed(), so the
        # query instruction has to be applied here rather than delegated.
        return self._encode([self._query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().embed(texts)]

    # ── Pickle support ────────────────────────────────────────────────────────
    # The engine's change-detection fingerprinting calls pickle.dumps(embedder).
    # onnxruntime.InferenceSession (held inside TextEmbedding) is not picklable,
    # so we exclude it.  _load() recreates the session lazily after unpickling.

    def __getstate__(self) -> dict:
        return {
            "_model_name": self._model_name,
            "_threads": self._threads,
            # Changing the query instruction changes the document-query
            # geometry, so it must invalidate every memoised embedding.
            "_query_prefix": self._query_prefix,
            "_model": None,
            "_max_tokens": None,  # re-read from the tokenizer after unpickling
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> TextEmbedding:
        """Lazy-load the ONNX model on first call."""
        if self._model is None:
            import truststore
            from fastembed import TextEmbedding

            # Corporate proxies (Zscaler etc.) MITM TLS; inject the OS trust
            # store so model downloads succeed without extra CA bundles.
            truststore.inject_into_ssl()

            _register_model()
            logger.info("Loading fastembed model %s", self._model_name)
            self._model = TextEmbedding(
                self._model_name,
                threads=self._threads,
            )
        return self._model


def _register_model() -> None:
    """Teach fastembed about CodeRankEmbed.  Idempotent across embedders."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    try:
        TextEmbedding.add_custom_model(
            model=DEFAULT_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=_MODEL_REPO),
            dim=_MODEL_DIM,
            model_file=_MODEL_FILE,
        )
    except ValueError:
        pass  # already registered by an earlier embedder in this process
