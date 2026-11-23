"""Labels for the Apple silicon warm server versus in-process ONNX."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from smritikosh.adapters.embedder.mps.client import warm_server_label


def test_windows_and_other_platforms_use_in_process_onnx() -> None:
    with patch("smritikosh.adapters.embedder.mps.client.sys.platform", "win32"):
        assert warm_server_label() == "in-process ONNX"
    with patch("smritikosh.adapters.embedder.mps.client.sys.platform", "linux"):
        assert warm_server_label() == "in-process ONNX"
    with patch("smritikosh.adapters.embedder.mps.client.hasattr", return_value=False):
        assert warm_server_label() == "in-process ONNX"


def test_apple_silicon_reports_listening_idle_or_off(tmp_path: Path) -> None:
    socket_path = tmp_path / "embed.sock"
    with (
        patch("smritikosh.adapters.embedder.mps.client.sys.platform", "darwin"),
        patch(
            "smritikosh.adapters.embedder.mps.client.platform.machine",
            return_value="arm64",
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.warm_enabled",
            return_value=True,
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.daemon._idle_minutes_default",
            return_value=30.0,
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.daemon._socket_path",
            return_value=socket_path,
        ),
    ):
        assert warm_server_label() == "off"
        socket_path.write_text("")
        assert warm_server_label() == "listening, idle 30 minutes"

    with (
        patch("smritikosh.adapters.embedder.mps.client.sys.platform", "darwin"),
        patch(
            "smritikosh.adapters.embedder.mps.client.platform.machine",
            return_value="arm64",
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.warm_enabled",
            return_value=False,
        ),
    ):
        assert warm_server_label() == "off"

    with (
        patch("smritikosh.adapters.embedder.mps.client.sys.platform", "darwin"),
        patch(
            "smritikosh.adapters.embedder.mps.client.platform.machine",
            return_value="arm64",
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.warm_enabled",
            return_value=True,
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.daemon._idle_minutes_default",
            return_value=1.5,
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.daemon._socket_path",
            return_value=socket_path,
        ),
    ):
        assert warm_server_label() == "listening, idle 1.5 minutes"

    with (
        patch("smritikosh.adapters.embedder.mps.client.sys.platform", "darwin"),
        patch(
            "smritikosh.adapters.embedder.mps.client.platform.machine",
            return_value="arm64",
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.warm_enabled",
            return_value=True,
        ),
        patch(
            "smritikosh.adapters.embedder.mps.client.daemon._socket_path",
            side_effect=RuntimeError("no package"),
        ),
    ):
        assert warm_server_label() == "off"
