"""Tests for ``python -m smritikosh``.

The console script only lands on PATH when the installing environment's ``bin``
directory is already there, so ``python -m`` is the fallback reached for when it
is not.  It broke by simply not existing, which no in-process test of
``cli.main`` would have caught — the subprocess case below is the only one that
runs the package the way a user types it.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_should_run_the_cli_as_a_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "smritikosh", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage: python -m smritikosh" in result.stdout


def test_should_hand_off_to_cli_main_when_run_as_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same entry point in-process, so the module is not a coverage blind spot."""
    monkeypatch.setattr(sys, "argv", ["smritikosh", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("smritikosh", run_name="__main__")

    assert exit_info.value.code == 0
    assert "semantic vector search over a codebase" in capsys.readouterr().out
