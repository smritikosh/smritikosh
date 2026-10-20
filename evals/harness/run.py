"""Execute the question x arm x repeat matrix.

Three properties matter more than speed here:

* **Resumable.** A run costs money. Anything already completed is skipped,
  so a crash, a rate limit, or a laptop lid closing costs you the current
  run and nothing else.
* **Randomized.** Running all of one arm before the next lets provider-side
  prompt caching and time-of-day latency land on whichever arm went last.
  The order is shuffled under a fixed seed: unbiased, but reproducible.
* **Budgeted.** `--max-spend` stops the matrix when cumulative cost crosses
  a line, because a prompt-loop bug should cost a few dollars, not a few
  hundred.

Usage:
    python run.py --dry-run
    python run.py --questions q08 --arms bare --repeats 1
    python run.py --max-spend 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

from config import (
    RUNS_DIR,
    WORKTREES_DIR,
    load_arms,
    load_golden,
    parse_run_id,
    run_id,
)
from extract import extract_run


def select(items, wanted, key=lambda x: x.id):
    if not wanted:
        return items
    names = {w.strip() for w in wanted.split(",")}
    chosen = [i for i in items if key(i) in names]
    missing = names - {key(i) for i in chosen}
    if missing:
        raise SystemExit(f"unknown: {', '.join(sorted(missing))}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="golden")
    parser.add_argument("--arms", help="comma-separated arm ids (default: all enabled)")
    parser.add_argument(
        "--questions", help="comma-separated question ids (default: all)"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--model", help="agent model (overrides EVAL_MODEL and arms.yaml)"
    )
    parser.add_argument(
        "--thinking", help="thinking level (overrides EVAL_THINKING and arms.yaml)"
    )
    parser.add_argument("--seed", type=int, default=20251020, help="run-order seed")
    parser.add_argument("--max-spend", type=float, default=10.0, help="USD ceiling")
    parser.add_argument("--timeout", type=int, default=900, help="per-run seconds")
    parser.add_argument(
        "--only",
        metavar="RUN_ID",
        help="run exactly one cell, e.g. q08__bare__r1",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the matrix and exit"
    )
    parser.add_argument("--redo", action="store_true", help="re-run completed cells")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    all_arms, defaults = load_arms()

    # --only is the one-cell invocation: the thing you reach for while
    # debugging the harness, or when re-running a single suspicious result.
    # It exists as its own flag because spelling the same cell out as three
    # separate options invites the typo that quietly runs forty.
    wanted_questions, wanted_arms, repeats, first_repeat = (
        args.questions,
        args.arms,
        args.repeats,
        1,
    )
    if args.only:
        question_id, arm_id, repeat = parse_run_id(args.only)
        wanted_questions, wanted_arms = question_id, arm_id
        # Pin to that exact repeat rather than running 1..n.
        first_repeat = repeats = repeat

    arms = select(
        [a for a in all_arms if a.enabled] if not wanted_arms else all_arms, wanted_arms
    )
    questions = select(golden.questions, wanted_questions)

    model = args.model or defaults.get("model")
    thinking = args.thinking or defaults.get("thinking", "low")

    cells = [
        (q, a, r)
        for q in questions
        for a in arms
        for r in range(first_repeat, repeats + 1)
    ]
    # Shuffle across the whole matrix, not within an arm: the thing being
    # de-biased is the position of an arm in wall-clock time.
    random.Random(args.seed).shuffle(cells)

    print(f"commit {golden.commit} | model {model} | thinking {thinking}")
    print(
        f"{len(questions)} questions x {len(arms)} arms x {repeats} = {len(cells)} runs"
    )
    print(f"order seed {args.seed} | budget ${args.max_spend:.2f}\n")

    if args.dry_run:
        for q, arm, repeat in cells:
            print(
                f"  {run_id(q.id, arm.id, repeat)}  [{arm.script}] "
                f"docs={arm.docs_label}"
            )
        return 0

    missing = {a.profile for a in arms if not (WORKTREES_DIR / a.profile).exists()}
    if missing:
        raise SystemExit(
            f"missing worktrees for profile(s) {sorted(missing)} — run prepare.py"
        )

    spent = 0.0
    done = failed = skipped = 0

    for index, (question, arm, repeat) in enumerate(cells, 1):
        rid = run_id(question.id, arm.id, repeat)
        run_dir = RUNS_DIR / rid
        record_path = run_dir / "record.json"

        if record_path.exists() and not args.redo:
            previous = json.loads(record_path.read_text())
            spent += (previous.get("cost") or {}).get("total", 0.0)
            skipped += 1
            continue

        if spent >= args.max_spend:
            print(
                f"\nbudget reached (${spent:.2f}) — stopping with "
                f"{len(cells) - index + 1} left"
            )
            break

        run_dir.mkdir(parents=True, exist_ok=True)
        # meta is written before the run so a crashed cell still says what it
        # was trying to do.
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "run_id": rid,
                    "question_id": question.id,
                    # The text, not just the id: a record should be readable
                    # on its own, without opening the golden file beside it.
                    "question": question.question,
                    "category": question.category,
                    "arm": arm.id,
                    "repeat": repeat,
                    "model": model,
                    "thinking": thinking,
                    "docs": arm.docs_files,
                    "profile": arm.profile,
                    "commit": golden.commit,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
        )

        env = {
            **os.environ,
            "EVAL_QUESTION": question.question,
            "EVAL_RUN_DIR": str(run_dir),
            "EVAL_CWD": str(WORKTREES_DIR / arm.profile),
            # os.pathsep-joined so the arm script can append several docs in
            # order; a single file is just a list of one.
            "EVAL_DOCS": os.pathsep.join(str(p) for p in arm.docs_paths),
            "EVAL_MODEL": model,
            "EVAL_THINKING": thinking,
        }

        print(f"[{index}/{len(cells)}] {rid} ", end="", flush=True)
        started = time.time()
        try:
            proc = subprocess.run(
                [str(arm.script_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            status = proc.returncode
            if status != 0:
                (run_dir / "harness_error.txt").write_text(proc.stderr[-4000:])
        except subprocess.TimeoutExpired:
            status = 124
            (run_dir / "harness_error.txt").write_text(f"timeout after {args.timeout}s")

        record = extract_run(run_dir)
        record_path.write_text(json.dumps(record, indent=2))
        spent += record["cost"]["total"]

        # A provider error is not a result. Marking it failed keeps the cell
        # out of the report and lets a later resume retry it, instead of
        # freezing an empty answer into the record as a zero score.
        if record.get("provider_error"):
            failed += 1
            (run_dir / "harness_error.txt").write_text(record["provider_error"])
            print(f"PROVIDER ERROR: {record['provider_error'][:120]}")
            continue

        if status == 0 and record["answer"]:
            done += 1
            used = "/".join(f"{k}:{v}" for k, v in record["tool_counts"].items() if v)
            print(
                f"ok {time.time() - started:.0f}s "
                f"${record['cost']['total']:.4f} "
                f"tools={len(record['tool_calls'])} ({used}) "
                f"ctx={record['context']['peak_tok']} "
                f"| spent ${spent:.2f}"
            )
        else:
            failed += 1
            print(f"FAILED exit={status} (see {run_dir}/harness_error.txt)")

    print(f"\n{done} ok, {failed} failed, {skipped} skipped | total ${spent:.2f}")
    print(f"records in {RUNS_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
