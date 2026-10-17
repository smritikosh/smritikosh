"""Tests for the warm embedding server — real sockets, fake embedder, no model.

The daemon runs in a thread on a tmp-path socket, so every test exercises the
real wire protocol without ever spawning a process or loading a checkpoint.

Coverage map
────────────
warm_enabled      – default on, off in CI, forced on/off by env
_socket_path      – version-stamped, user-private runtime dir
encode_queries    – warm path sends raw texts, falls back in-process on any
                    failure, documents never touch the server
daemon            – roundtrip vectors, parallel clients share one server,
                    concurrent startups keep exactly one, stalled client is
                    dropped and later clients served, server error reaches the
                    client, idle exit removes the socket, stop command
stop_server       – False when nothing is running
__main__          – the package runs as the daemon entry point
"""

from __future__ import annotations

import importlib.metadata
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from smritikosh.adapters.embedder.mps import client, daemon
from smritikosh.adapters.embedder.mps.embedder import MpsEmbedder
from smritikosh.constants import CODE_QUERY_INSTRUCTION


class _FakeEmbedder:
    """Stands in for MpsEmbedder inside the daemon — records, never loads."""

    instances: list[_FakeEmbedder] = []

    def __init__(self) -> None:
        type(self).instances.append(self)
        self.queries: list[list[str]] = []

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        self.queries.append(list(texts))
        return [[1.0, 2.0]] * len(texts)


class _StubSession:
    """In-process encode stub for the fallback path (as in test_mps.py)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append(list(texts))
        return [_StubVector() for _ in texts]


class _StubVector:
    def tolist(self) -> list[float]:
        return [0.5] * 768


@pytest.fixture()
def warm(monkeypatch) -> Path:
    """Run the warm machinery against a tmp socket with a fake embedder.

    The socket lives in a short /tmp path: pytest's tmp_path easily exceeds the
    104-byte AF_UNIX path limit on macOS.
    """
    _FakeEmbedder.instances = []
    sock_path = Path(tempfile.mkdtemp(prefix="smri-warm-")) / "t.sock"
    monkeypatch.setattr(daemon, "_socket_path", lambda: sock_path)
    monkeypatch.setattr(daemon, "_READ_TIMEOUT_S", 1.0)
    monkeypatch.setattr(daemon, "MpsEmbedder", _FakeEmbedder)
    monkeypatch.setenv(daemon._WARM_ENV, "1")
    return sock_path


def _start(sock_path: Path, **kwargs) -> threading.Thread:
    thread = threading.Thread(target=daemon.run_server, kwargs=kwargs, daemon=True)
    thread.start()
    for _ in range(200):
        if sock_path.exists():
            return thread
        time.sleep(0.01)
    raise AssertionError("warm server did not start")


def _stop(sock_path: Path, thread: threading.Thread) -> None:
    assert client.stop_server()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not sock_path.exists()  # removed on exit


# ── warm_enabled ──────────────────────────────────────────────────────────────


def test_warm_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(daemon._WARM_ENV, raising=False)
    monkeypatch.delenv("CI", raising=False)

    assert client.warm_enabled()


def test_warm_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv(daemon._WARM_ENV, "0")

    assert not client.warm_enabled()


def test_warm_defaults_off_in_ci(monkeypatch) -> None:
    """One-shot CI machines gain nothing from a daemon holding 1 GB."""
    monkeypatch.delenv(daemon._WARM_ENV, raising=False)
    monkeypatch.setenv("CI", "true")

    assert not client.warm_enabled()


def test_warm_forced_on_overrides_ci(monkeypatch) -> None:
    monkeypatch.setenv(daemon._WARM_ENV, "1")
    monkeypatch.setenv("CI", "true")

    assert client.warm_enabled()


# ── _socket_path ──────────────────────────────────────────────────────────────


def test_socket_path_is_versioned_and_user_private(monkeypatch, tmp_path) -> None:
    """The version in the name keeps upgraded clients off an old server."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    path = daemon._socket_path()

    assert importlib.metadata.version("smritikosh") in path.name
    assert path.parent == tmp_path / "smritikosh"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


# ── encode_queries routing (client side, no real server) ─────────────────────


@pytest.fixture()
def warm_client(monkeypatch):
    monkeypatch.setenv(daemon._WARM_ENV, "1")
    return monkeypatch


def test_encode_queries_sends_raw_texts_to_the_server(warm_client) -> None:
    """The server applies the query instruction — the client must not."""
    sent: list[list[str]] = []

    def fake_encode_via_server(texts):
        sent.append(list(texts))
        return [[9.9]]

    warm_client.setattr(client, "encode_via_server", fake_encode_via_server)
    embedder = MpsEmbedder()
    warm_client.setattr(
        embedder, "_encode", lambda texts: pytest.fail("must not embed in-process")
    )

    assert embedder.encode_queries(["q1", "q2"]) == [[9.9]]
    assert sent == [["q1", "q2"]]


def test_encode_queries_falls_back_in_process(warm_client) -> None:
    """A dead or hung server must never break search — just slow it down."""

    def raise_oserror(texts):
        raise OSError("no server")

    warm_client.setattr(client, "encode_via_server", raise_oserror)
    embedder = MpsEmbedder()
    session = _StubSession()
    embedder._model = session  # type: ignore[assignment]

    assert embedder.encode_queries(["retry failed uploads"]) == [[0.5] * 768]
    assert session.calls == [[CODE_QUERY_INSTRUCTION + "retry failed uploads"]]


def test_encode_documents_never_touches_the_server(warm_client) -> None:
    """Indexing batches stay in-process; the server is a search-time shortcut."""
    warm_client.setattr(
        client,
        "encode_via_server",
        lambda texts: pytest.fail("documents stay in-process"),
    )
    embedder = MpsEmbedder()
    embedder._model = _StubSession()  # type: ignore[assignment]

    assert embedder.encode_documents(["x"]) == [[0.5] * 768]


# ── The daemon over a real socket ─────────────────────────────────────────────


def test_roundtrip_returns_vectors(warm) -> None:
    thread = _start(warm, idle_minutes=5)
    try:
        assert client.encode_via_server(["retry failed uploads"]) == [[1.0, 2.0]]
        assert _FakeEmbedder.instances[0].queries == [["retry failed uploads"]]
    finally:
        _stop(warm, thread)


def test_parallel_clients_share_one_server(warm) -> None:
    thread = _start(warm, idle_minutes=5)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(lambda i: client.encode_via_server([f"q{i}"]), range(8))
            )
        assert all(r == [[1.0, 2.0]] for r in results)
        served = [q for inst in _FakeEmbedder.instances for q in inst.queries]
        assert sorted(q[0] for q in served) == [f"q{i}" for i in range(8)]
    finally:
        _stop(warm, thread)


def test_concurrent_startups_keep_exactly_one_server(warm) -> None:
    """The flock is held for a server's lifetime; a loser exits at once."""
    thread = _start(warm, idle_minutes=5)
    try:
        assert daemon.run_server(idle_minutes=5) == 0  # returns immediately
        assert len(_FakeEmbedder.instances) == 0  # and never served anything
    finally:
        _stop(warm, thread)


def test_stalled_client_is_dropped_and_others_served(warm) -> None:
    """A client that connects but never sends costs one read timeout, no more."""
    thread = _start(warm, idle_minutes=5)
    stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stalled.connect(str(warm))
    try:
        started = time.monotonic()
        assert client.encode_via_server(["q"]) == [[1.0, 2.0]]
        # _READ_TIMEOUT_S is patched to 1s; a hang would look like 5x that.
        assert time.monotonic() - started < 5
    finally:
        stalled.close()
        _stop(warm, thread)


def test_server_side_error_reaches_the_client(warm, monkeypatch) -> None:
    """The client falls back in-process when the server reports a failure."""

    class _Failing:
        def encode_queries(self, texts):
            raise RuntimeError("GPU wedged")

    monkeypatch.setattr(daemon, "MpsEmbedder", lambda: _Failing())
    thread = _start(warm, idle_minutes=5)
    try:
        with pytest.raises(OSError, match="GPU wedged"):
            client.encode_via_server(["q"])
    finally:
        _stop(warm, thread)


def test_idle_exit_removes_the_socket(warm) -> None:
    thread = _start(warm, idle_minutes=0.01)  # 0.6 s

    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not warm.exists()


def test_stop_server_returns_false_without_a_server(warm) -> None:
    assert not client.stop_server()


# ── __main__ entry point ──────────────────────────────────────────────────────


def test_package_runs_as_the_daemon_entry_point() -> None:
    """`python -m smritikosh.adapters.embedder.mps` is what _spawn execs."""
    result = subprocess.run(
        [sys.executable, "-m", "smritikosh.adapters.embedder.mps", "--help"],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0
