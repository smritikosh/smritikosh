"""The warm-server client: how a short-lived search process gets query vectors.

Runs inside every CLI search. Every function here either returns vectors or
raises — :meth:`MpsEmbedder.encode_queries` treats any raise as "embed
in-process instead", so a missing, hung, or crashed server only ever costs
speed, never correctness. The daemon module owns the paths; the client merely
needs to know where to knock.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Final

from smritikosh.adapters.embedder.mps import daemon

__all__ = ["encode_via_server", "ensure_server", "stop_server", "warm_enabled"]

logger = logging.getLogger(__name__)

#: Bound on waiting for vectors; covers the server's first-ever model load.
_RESPONSE_TIMEOUT_S: Final = 120.0
#: Bound on waiting for a spawned server to accept connections.
_SPAWN_TIMEOUT_S: Final = 10.0


def warm_enabled() -> bool:
    """Whether searches should try the warm server.

    On by default on Unix; off in CI (one-shot machines gain nothing from a
    daemon) unless explicitly forced on.
    """
    setting = os.environ.get(daemon._WARM_ENV)
    if setting:
        return setting == "1"
    return not os.environ.get("CI") and hasattr(socket, "AF_UNIX")


def encode_via_server(texts: list[str]) -> list[list[float]]:
    """Ask the warm server for query vectors; raise on any failure.

    Texts go raw — the server's own MpsEmbedder applies the query instruction.
    """
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
            sock.settimeout(daemon._READ_TIMEOUT_S)
            sock.sendall(b'{"stop": true}\n')
        return True
    except OSError:
        return False


def _connect() -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(daemon._socket_path()))
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

    The spawn target is this package itself — see ``__main__.py``, which keeps
    the daemon's launch contract next to the daemon.
    """
    with open(daemon._socket_path().with_suffix(".log"), "ab") as log:
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "smritikosh.adapters.embedder.mps"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
