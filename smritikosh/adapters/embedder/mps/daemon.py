"""The warm embedding daemon: one process, one loaded model, one socket.

Run directly (``python -m smritikosh.adapters.embedder.mps``) or through
:func:`run_server`. This module owns the runtime paths — socket, lock, log —
because it creates them; the client in ``client.py`` only needs to know where
to knock.

The wire format is one JSON object per line each way; the socket name carries
the package version, so a client and a server always run identical code and no
protocol negotiation is needed. The daemon is serial by design: encoding holds
the only GPU, so serving one connection at a time costs nothing and removes
every concurrency primitive.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import socketserver
import time
from pathlib import Path
from typing import Final

from smritikosh.adapters.embedder.mps.embedder import MpsEmbedder

__all__ = ["run_server"]

logger = logging.getLogger(__name__)

_WARM_ENV: Final = "SMRITIKOSH_EMBED_SERVER"  # "0" = off, "1" = forced on
_IDLE_ENV: Final = "SMRITIKOSH_EMBED_IDLE_MINUTES"
_DEFAULT_IDLE_MINUTES: Final = 30.0

#: A client that connects but sends nothing is dropped after this.
_READ_TIMEOUT_S: Final = 5.0


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
    """Run the warm embedding server until it idles out or is stopped."""
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
