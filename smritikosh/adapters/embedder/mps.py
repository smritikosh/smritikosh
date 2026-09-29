"""Apple GPU embedding backend — PyTorch on Metal (MPS), opt-in.

Same model as the default CPU adapter (:data:`DEFAULT_MODEL`, CodeRankEmbed),
run through PyTorch's Metal backend instead of ONNX Runtime.  Measured on an
M4 Max (16 CPU cores, 40 GPU cores): 77 chunks/s against 35 for eight parallel
INT8 ONNX sessions, and 12 for a single one.

Not the default, and deliberately so:

* it only exists on Apple silicon, while the ONNX path runs anywhere;
* it pulls in PyTorch (~2.5 GB), which the base install avoids on purpose;
* loading the checkpoint costs a few seconds per process, which a one-shot
  ``search`` pays in full — it earns its keep on a long indexing run.

Install with ``pip install 'smritikosh[mps]'``.

This runs the fp32 checkpoint, whereas the ONNX path runs an INT8 export.
The two are *not* interchangeable: measured cosine between their vectors is
0.87-0.90 and they rank differently, so an index must be built and searched
with the same backend.  ``model_id`` carries the backend for that reason.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from smritikosh.constants import CODE_QUERY_INSTRUCTION, DEFAULT_MODEL
from smritikosh.ports.embedder import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["MpsEmbedder"]

logger = logging.getLogger(__name__)

_MODEL_DIM = 768

#: Token window applied to the PyTorch checkpoint.  The model card advertises
#: 8192, and unlike the INT8 ONNX export nothing here forces 512 — but the
#: budget feeds the chunker, and a wider window means fewer, longer chunks and
#: correspondingly coarser line citations.  512 keeps ranges as tight as the
#: CPU path; raise it only if you want bigger passages.
_MAX_SEQ_LENGTH = 512

#: Sequences per forward pass.  Measured on an M4 Max: 16 gave 76.6 chunks/s,
#: 64 gave 75.6, 128 gave 68.6 — the GPU saturates early, and a larger batch
#: only pads more aggressively.
_BATCH_SIZE = 16

_INSTALL_HINT = (
    "The mps embedder needs PyTorch and sentence-transformers. "
    "Install them with: pip install 'smritikosh[mps]'"
)


class MpsEmbedder(Embedder):
    """Embeds CodeRankEmbed on the Apple GPU via PyTorch MPS.

    Parameters
    ----------
    batch_size:
        Sequences per forward pass.  Defaults to 16, the measured optimum.
    max_seq_length:
        Token window, and therefore the chunker's size budget.  Defaults to
        512 so chunks match the CPU path.
    """

    def __init__(
        self,
        *,
        batch_size: int = _BATCH_SIZE,
        max_seq_length: int = _MAX_SEQ_LENGTH,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._model_name = DEFAULT_MODEL
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._model: SentenceTransformer | None = None
        # A tokenizer of our own for counting — see count_tokens.
        self._counter: object | None = None
        self._query_prefix = CODE_QUERY_INSTRUCTION
        # One Metal device, reached from many threads: the pipeline runs
        # chunking under @sm.threaded, and the chunker holds count_tokens.
        self._lock = threading.Lock()

    # ── Embedder protocol ─────────────────────────────────────────────────────

    @property
    def dims(self) -> int:
        """Embedding width.  Known from the model spec — no model load."""
        return _MODEL_DIM

    @property
    def model_id(self) -> str:
        """Identity including the backend — fp32 and INT8 are different spaces."""
        return f"{self._model_name}:mps:{self.dims}"

    @property
    def max_tokens(self) -> int:
        """The window this adapter enforces, which the chunker budgets against."""
        return self._max_seq_length

    def count_tokens(self, text: str) -> int:
        """Count tokens with the same tokenizer used for document embedding.

        Deliberately counts on a *separate* tokenizer instance rather than the
        model's own.  The chunker calls this from a thread pool while encoding
        runs, and ``SentenceTransformer.encode`` reconfigures truncation on its
        tokenizer for every batch.  Sharing one meant a Rust-side mutable
        borrow landing mid-read: ``RuntimeError: Already borrowed``.

        Reading the raw backend also keeps the count honest — the wrapper would
        report the truncated length, and the chunker is asking how long the
        text *is*, precisely so it can stay under the limit.
        """
        self._load()
        return len(self._counter.encode(text).ids)  # type: ignore[attr-defined]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        # CodeRankEmbed is asymmetric: the instruction is what puts a query in
        # the same space as the documents, and omitting it degrades retrieval.
        return self._encode([self._query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        # Serialise the forward pass: a torch module on Metal is not safe for
        # concurrent use, and overlapping encodes abort the process with
        # "A command encoder is already encoding to this command buffer".
        # Costs nothing in throughput — there is one GPU either way.
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vectors]

    # ── Pickle support ────────────────────────────────────────────────────────
    # The engine's change-detection fingerprinting calls pickle.dumps(embedder).
    # Neither a live torch module nor a Lock is picklable, so both are excluded
    # and rebuilt on the far side.

    def __getstate__(self) -> dict:
        return {
            "_model_name": self._model_name,
            "_batch_size": self._batch_size,
            "_max_seq_length": self._max_seq_length,
            # Changing the instruction changes the document-query geometry,
            # so it must invalidate every memoised embedding.
            "_query_prefix": self._query_prefix,
            "_model": None,
            "_counter": None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._model = None
        self._counter = None
        self._lock = threading.Lock()  # a Lock does not survive pickling

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> SentenceTransformer:
        """Lazy-load the checkpoint onto the GPU on first call.

        Double-checked under a lock.  Without it two worker threads both see
        ``None`` and both build a model: the second one initialises Metal
        while the first is already encoding, which aborts the process.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                self._model, self._counter = self._build()
        return self._model

    def _build(self) -> tuple[SentenceTransformer, object]:
        """Build the model and its own counting tokenizer.

        Call only while holding ``_lock``.
        """
        import truststore

        # Corporate proxies (Zscaler etc.) MITM TLS; inject the OS trust store
        # so the weights download succeeds without extra CA bundles.
        truststore.inject_into_ssl()

        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Metal (MPS) is not available on this machine — "
                "the mps embedder needs Apple silicon. Use the default "
                "fastembed embedder instead."
            )

        logger.info("Loading %s onto MPS", self._model_name)
        model = SentenceTransformer(
            self._model_name,
            trust_remote_code=True,
            device="mps",
        )
        model.max_seq_length = self._max_seq_length

        # A second, private tokenizer for count_tokens.  Its raw Rust backend
        # is safe to read from many threads as long as nothing reconfigures
        # it, and nothing here ever does.
        counter = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        ).backend_tokenizer
        return model, counter
