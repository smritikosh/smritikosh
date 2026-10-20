#!/usr/bin/env bash
# Arm: bare — agentic grep, nothing else.
#
# --no-extensions is the off switch that makes this arm honest. With no
# extension loaded there is no smritikosh tool to call, no bash enforcement
# redirecting grep into semantic search, and no background indexer quietly
# building an index for the arm whose whole purpose is to run without one.
# Nothing needs disabling because nothing gets loaded.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ARM_FLAGS=(--no-extensions)

run_pi
