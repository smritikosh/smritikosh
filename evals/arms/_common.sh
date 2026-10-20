#!/usr/bin/env bash
# Shared pi invocation. Sourced by every arm script; never run directly.
#
# An arm script's whole job is to set ARM_FLAGS (and optionally ARM_CTX /
# ARM_CTX2) and then call run_pi. Everything that must be identical across
# arms — model, thinking level, tool loadout, isolation flags, timing — lives
# here, so an arm cannot accidentally differ in a dimension the experiment is
# not testing.
#
# Every arm runs `pi --no-extensions`. Retrieval tools are command-line
# programs reached through bash, so the only thing that varies between arms
# is the text appended to the system prompt. That keeps smritikosh and
# graphify on identical footing and keeps editor chrome, background indexers
# and bash enforcement out of a headless benchmark.
#
# Inputs (environment, set by harness/run.py):
#   EVAL_QUESTION   the question text
#   EVAL_RUN_DIR    output directory for this single run
#   EVAL_CWD        worktree to run in (pinned to the golden set's commit)
#   EVAL_DOCS       optional PATH-separated list of docs files, appended in
#                   order (the tuned arm stacks a workflow block on the same
#                   overview the other docs arms get)
#   EVAL_MODEL      provider/model for the agent
#   EVAL_THINKING   thinking level

set -euo pipefail

run_pi() {
    : "${EVAL_QUESTION:?EVAL_QUESTION is required}"
    : "${EVAL_RUN_DIR:?EVAL_RUN_DIR is required}"
    : "${EVAL_CWD:?EVAL_CWD is required}"
    : "${EVAL_MODEL:?EVAL_MODEL is required}"
    local thinking="${EVAL_THINKING:-low}"

    mkdir -p "$EVAL_RUN_DIR/session"

    # Isolation. Nothing is discovered implicitly, so a stray file in ~/.pi or
    # in the worktree cannot change an arm's behaviour between runs:
    #   --no-extensions         no auto-loaded extensions (explicit -e still works)
    #   --no-skills             no skill files
    #   --no-prompt-templates   no prompt templates
    #   --no-context-files      no AGENTS.md / CLAUDE.md pickup from disk
    #   --no-mcp                no MCP servers
    #
    # Tools are an exclusion, not an allowlist: --tools is an allowlist that
    # would also drop extension-provided tools, which is exactly what the
    # smritikosh arms need to keep. Denying edit and write leaves read + bash
    # for every arm, plus whatever the arm's extension registers. The agent is
    # answering questions, so it never needs to mutate the tree.
    local -a argv=(
        pi -p --mode json
        --session-dir "$EVAL_RUN_DIR/session"
        --no-skills --no-prompt-templates --no-context-files --no-mcp
        --exclude-tools edit,write
        --model "$EVAL_MODEL"
        --thinking "$thinking"
    )

    argv+=("${ARM_FLAGS[@]}")

    # Docs first, then arm context: an arm's tool instructions should be read
    # against the project overview, not before it.
    if [[ -n "${EVAL_DOCS:-}" ]]; then
        local doc
        while IFS= read -r -d: doc; do
            [[ -n "$doc" ]] && argv+=(--append-system-prompt "$doc")
        done <<< "${EVAL_DOCS}:"
    fi
    for ctx in "${ARM_CTX:-}" "${ARM_CTX2:-}"; do
        [[ -n "$ctx" ]] && argv+=(--append-system-prompt "$ctx")
    done

    # -- ends option parsing: a question beginning with a dash is a question,
    # not a flag.
    argv+=(-- "$EVAL_QUESTION")

    printf '%s\n' "${argv[@]}" > "$EVAL_RUN_DIR/argv.txt"

    # Wall time is measured around the subprocess as well as derived from
    # session timestamps later; the two disagreeing means something stalled
    # outside the model loop, which is worth knowing.
    local start_ns end_ns status=0
    start_ns=$(python3 -c 'import time; print(time.time_ns())')

    (cd "$EVAL_CWD" && "${argv[@]}") \
        > "$EVAL_RUN_DIR/stream.jsonl" \
        2> "$EVAL_RUN_DIR/stderr.txt" || status=$?

    end_ns=$(python3 -c 'import time; print(time.time_ns())')

    python3 - "$EVAL_RUN_DIR" "$start_ns" "$end_ns" "$status" <<'PY'
import json, sys
run_dir, start_ns, end_ns, status = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(f"{run_dir}/subprocess.json", "w") as fh:
    json.dump({"wall_ms": (end_ns - start_ns) / 1e6, "exit_code": status}, fh, indent=2)
PY

    return "$status"
}
