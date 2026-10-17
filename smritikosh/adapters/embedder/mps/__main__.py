"""Daemon entry point: ``client._spawn`` execs this as a detached process."""

import argparse

from smritikosh.adapters.embedder.mps.daemon import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm embedding server (daemon).")
    parser.add_argument("--idle-minutes", type=float, default=None)
    raise SystemExit(run_server(idle_minutes=parser.parse_args().idle_minutes))
