#!/usr/bin/env bash
# Arm: smritikosh — the CLI, driven through bash.
#
# Same shape as graphify.sh on purpose. Both tools are command-line programs
# and every arm already has bash, so giving each one a context block naming
# its commands puts them on identical footing. Wiring smritikosh through a pi
# extension while graphify went through bash would have compared our
# integration quality against their retrieval quality.
#
# It also keeps the arm honest in a quieter way: no extension means no
# background indexer, no bash enforcement redirecting grep, and no editor
# chrome loaded into a headless run. Every arm is `pi --no-extensions`, and
# the only thing that differs between them is the text appended to the
# system prompt.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! command -v smritikosh > /dev/null 2>&1; then
    echo "smritikosh arm: smritikosh is not on PATH" >&2
    exit 2
fi

ARM_FLAGS=(--no-extensions)
ARM_CTX="$(dirname "${BASH_SOURCE[0]}")/ctx/smritikosh.md"

run_pi
