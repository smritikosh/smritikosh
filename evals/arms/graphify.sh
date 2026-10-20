#!/usr/bin/env bash
# Arm: graphify — the comparator, driven as a CLI through bash.
#
# No extension is written for graphify on purpose. It is a command-line tool
# and the agent already has bash, so a context block naming the commands is
# enough to put it on equal footing. Building an extension first would mean
# grading our wrapper quality rather than their retrieval quality. If the
# graphify arms earn their place in the matrix, a proper extension can come
# later and the arm script is the only thing that changes.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! command -v graphify > /dev/null 2>&1; then
    echo "graphify arm: graphify is not on PATH" >&2
    exit 2
fi

ARM_FLAGS=(--no-extensions)
ARM_CTX="$(dirname "${BASH_SOURCE[0]}")/ctx/graphify.md"

run_pi
