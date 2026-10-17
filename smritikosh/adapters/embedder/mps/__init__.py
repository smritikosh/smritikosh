"""Apple GPU embedding on Metal, and the warm server that keeps it loaded.

- ``embedder`` — :class:`MpsEmbedder`, the adapter the codebase knows.
- ``client`` — runs inside each short-lived search process; gets query
  vectors from the warm server, spawning it when needed.
- ``daemon`` — the per-user background process holding one loaded model.
"""

from smritikosh.adapters.embedder.mps.client import ensure_server, stop_server
from smritikosh.adapters.embedder.mps.daemon import run_server
from smritikosh.adapters.embedder.mps.embedder import MpsEmbedder

__all__ = ["MpsEmbedder", "ensure_server", "run_server", "stop_server"]
