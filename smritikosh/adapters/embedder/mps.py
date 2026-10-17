"""Apple GPU embedding adapter using PyTorch on Metal — and its warm server.

This adapter runs the fp32 CodeRankEmbed checkpoint. Its vectors are not
compatible with the portable adapter's quantized ONNX vectors, so each adapter
declares a distinct vector-space identity.

Loading that checkpoint costs seconds even from the local cache, and every CLI
process paid it before exiting after one search. The warm server at the bottom
of this module keeps one loaded model in a per-user background process;
searches ask it for query vectors over a private Unix socket and fall back to
in-process loading whenever it is disabled, missing, or misbehaving.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from smritikosh.constants import CODE_QUERY_INSTRUCTION, DEFAULT_MODEL
from smritikosh.ports.embedder import Embedder, EmbeddingIdentity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["MpsEmbedder"]

logger = logging.getLogger(__name__)

_MODEL_DIM = 768
_VECTOR_SPACE = "coderank-pytorch-fp32-v1"

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
    "Apple GPU acceleration needs PyTorch and sentence-transformers. "
    "Reinstall smritikosh to restore its platform dependencies."
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
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            model=self._model_name,
            vector_space=_VECTOR_SPACE,
            dimensions=self.dims,
        )

    @property
    def display_name(self) -> str:
        return "CodeRankEmbed (Apple GPU accelerated)"

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
        # A warm per-user server may already hold this model in RAM; asking it
        # skips the multi-second load every CLI process would otherwise pay.
        if _warm_enabled():
            try:
                return _encode_via_server(texts)
            except Exception:
                logger.info(
                    "warm server unavailable; embedding in-process", exc_info=True
                )
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

        # Cache-first: a warm local load skips the hub's freshness checks,
        # which through a corporate proxy cost more than the load itself
        # (measured ~13.5s vs ~5s). Any failure — a genuine cache miss or a
        # half-populated cache — falls through to the original online path,
        # so first-time installation still works.
        try:
            return self._load_checkpoint(
                SentenceTransformer, AutoTokenizer, local_only=True
            )
        except Exception:
            logger.info("No usable local cache for %s — downloading", self._model_name)

        import truststore

        # Corporate proxies (Zscaler etc.) MITM TLS; inject the OS trust store
        # so the weights download succeeds without extra CA bundles. Only the
        # online path needs it — the local cache never touches the network.
        truststore.inject_into_ssl()
        return self._load_checkpoint(
            SentenceTransformer, AutoTokenizer, local_only=False
        )

    def _load_checkpoint(
        self,
        sentence_transformer_cls: type[SentenceTransformer],
        auto_tokenizer_cls,
        *,
        local_only: bool,
    ) -> tuple[SentenceTransformer, object]:
        """Load the model and its private counting tokenizer.

        ``local_only=True`` refuses all network access (hub freshness checks
        included) and raises when the checkpoint is not cached.
        """
        logger.info("Loading %s onto MPS (local_only=%s)", self._model_name, local_only)
        model = sentence_transformer_cls(
            self._model_name,
            trust_remote_code=True,
            device="mps",
            local_files_only=local_only,
        )
        model.max_seq_length = self._max_seq_length

        # A second, private tokenizer for count_tokens.  Its raw Rust backend
        # is safe to read from many threads as long as nothing reconfigures
        # it, and nothing here ever does.
        counter = auto_tokenizer_cls.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            local_files_only=local_only,
        ).backend_tokenizer
        return model, counter


# ── Warm embedding server ─────────────────────────────────────────────────────
# A per-user background process keeps one loaded model warm between CLI
# searches, reachable over a Unix socket private to the current user. The wire
# format is one JSON object per line each way; the socket name carries the
# package version, so a client and a server always run identical code and no
# protocol negotiation is needed. The daemon is serial by design: encoding
# holds the only GPU, so serving one connection at a time costs nothing and
# removes every concurrency primitive.

_WARM_ENV: Final = "SMRITIKOSH_EMBED_SERVER"  # "0" = off, "1" = forced on
_IDLE_ENV: Final = "SMRITIKOSH_EMBED_IDLE_MINUTES"
_DEFAULT_IDLE_MINUTES: Final = 30.0

#: Server: a client that connects but sends nothing is dropped after this.
_READ_TIMEOUT_S: Final = 5.0
#: Client: bound on waiting for vectors; covers the server's first model load.
_RESPONSE_TIMEOUT_S: Final = 120.0
#: Client: bound on waiting for a spawned server to accept connections.
_SPAWN_TIMEOUT_S: Final = 10.0


def _warm_enabled() -> bool:
    """Whether searches should try the warm server.

    On by default on Unix; off in CI (one-shot machines gain nothing from a
    daemon) unless explicitly forced on.
    """
    setting = os.environ.get(_WARM_ENV)
    if setting:
        return setting == "1"
    return not os.environ.get("CI") and hasattr(socket, "AF_UNIX")


def _idle_minutes_default() -> float:
    try:
        return float(os.environ.get(_IDLE_ENV, _DEFAULT_IDLE_MINUTES))
    except ValueError:
        return _DEFAULT_IDLE_MINUTES


def _socket_path() -> Path:
    """User-private, version-stamped socket path (created on demand)."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    run_dir = (Path(base) if base else Path.home() / ".cache") / "smritikosh"
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    version = importlib.metadata.version("smritikosh")
    return run_dir / f"embed-{version}.sock"


# ── Client ────────────────────────────────────────────────────────────────────


def _connect() -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(_socket_path()))
    return sock


def _connect_or_spawn() -> socket.socket:
    """Connect to the warm server, starting it first when necessary."""
    try:
        return _connect()
    except OSError:
        pass
    _spawn()
    deadline = time.monotonic() + _SPAWN_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            return _connect()
        except OSError:
            time.sleep(0.1)
    raise OSError("warm embedding server did not start in time")


def _spawn() -> None:
    """Start the server detached, so it outlives this CLI process.

    The spawn target is this module itself — see the ``__main__`` block at
    the bottom, which keeps the daemon's launch contract next to the daemon.
    """
    with open(_socket_path().with_suffix(".log"), "ab") as log:
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "smritikosh.adapters.embedder.mps"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _encode_via_server(texts: list[str]) -> list[list[float]]:
    """Ask the warm server for query vectors; raise on any failure."""
    sock = _connect_or_spawn()
    try:
        sock.settimeout(_RESPONSE_TIMEOUT_S)
        sock.sendall(json.dumps({"texts": texts}).encode() + b"\n")
        response = json.loads(sock.makefile("rb").readline())
    finally:
        sock.close()
    if "error" in response:
        raise OSError(response["error"])
    return response["vectors"]


def ensure_server() -> None:
    """Start the warm server if it is not already running."""
    _connect_or_spawn().close()


def stop_server() -> bool:
    """Ask a running warm server to exit; False when none was running."""
    try:
        with _connect() as sock:
            sock.settimeout(_READ_TIMEOUT_S)
            sock.sendall(b'{"stop": true}\n')
        return True
    except OSError:
        return False


# ── Daemon ────────────────────────────────────────────────────────────────────


class _WarmHandler(socketserver.StreamRequestHandler):
    """One connection is one request: texts in, vectors out."""

    def handle(self) -> None:
        server: _WarmServer = self.server  # type: ignore[assignment]
        self.connection.settimeout(_READ_TIMEOUT_S)
        try:
            request = json.loads(self.rfile.readline())
            server.deadline = time.monotonic() + server.idle_seconds
            if "stop" in request:
                server.stop_requested = True
                return
            if server.embedder is None:
                # First request pays the cache-first load; every later one
                # reuses this instance — this reference is the warmth.
                server.embedder = MpsEmbedder()
            response = {"vectors": server.embedder.encode_queries(request["texts"])}
        except Exception as exc:  # stalled client, bad JSON, encode failure
            response = {"error": str(exc)}
        self.wfile.write(json.dumps(response).encode() + b"\n")


class _WarmServer(socketserver.UnixStreamServer):
    """A serial Unix-socket server holding one loaded embedder."""

    embedder: MpsEmbedder | None = None
    stop_requested = False
    idle_seconds = _DEFAULT_IDLE_MINUTES * 60
    deadline = 0.0

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        """A client that dies mid-exchange is logged, never fatal."""
        logger.info("warm client connection failed", exc_info=True)


def run_server(*, idle_minutes: float | None = None) -> int:
    """Run the warm embedding server; see the ``__main__`` block below."""
    # The daemon must never route its own encodes through another server.
    os.environ[_WARM_ENV] = "0"
    import fcntl

    with open(_socket_path().with_suffix(".lock"), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # The lock is held for a server's lifetime, so exactly one lives.
            logger.info("another warm embedding server is already running")
            return 0
        path = _socket_path()
        path.unlink(missing_ok=True)  # stale socket; safe while holding the lock
        server = _WarmServer(str(path), _WarmHandler)
        os.chmod(path, 0o600)
        server.timeout = 1.0  # idle-tick for handle_request
        server.idle_seconds = (idle_minutes or _idle_minutes_default()) * 60
        server.deadline = time.monotonic() + server.idle_seconds
        logger.info("warm embedding server listening on %s", path)
        try:
            while not server.stop_requested and time.monotonic() < server.deadline:
                server.handle_request()
        finally:
            server.server_close()
            path.unlink(missing_ok=True)
    logger.info("warm embedding server exiting")
    return 0


if __name__ == "__main__":
    # Daemon entry point: _spawn() execs this module detached as
    # `python -m smritikosh.adapters.embedder.mps [--idle-minutes N]`.
    import argparse

    _parser = argparse.ArgumentParser(description="Warm embedding server (daemon).")
    _parser.add_argument("--idle-minutes", type=float, default=None)
    raise SystemExit(run_server(idle_minutes=_parser.parse_args().idle_minutes))
