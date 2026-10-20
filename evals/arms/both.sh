#!/usr/bin/env bash
# Arm: both — the smritikosh and graphify CLIs together.
#
# Exists to answer whether the two are complementary or redundant. If `both`
# does not beat the better of the two alone, one of them is not earning its
# place in the context window.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for binary in smritikosh graphify; do
    if ! command -v "$binary" > /dev/null 2>&1; then
        echo "both arm: $binary is not on PATH" >&2
        exit 2
    fi
done

ARM_FLAGS=(--no-extensions)
# Both blocks, in the order the arm table lists them.
ARM_CTX="$(dirname "${BASH_SOURCE[0]}")/ctx/smritikosh.md"
ARM_CTX2="$(dirname "${BASH_SOURCE[0]}")/ctx/graphify.md"

run_pi
