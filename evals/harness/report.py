"""Aggregate records and grades into runs.csv plus report.md.

The report leads with the contrasts the arm matrix was built to expose, not
with a leaderboard. "Smritikosh scores 0.82" is unactionable; "the index
adds 0.21 coverage once docs are held fixed" is a claim someone can argue
with.

Medians and IQR rather than means: with a handful of repeats one timed-out
run drags a mean somewhere misleading, and the spread is the more honest
summary of a small sample anyway.

Usage:
    python report.py                 # writes evals/results/<date>/
    python report.py --out results/pilot
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import statistics
import sys
from collections import defaultdict

from config import RESULTS_DIR, RUNS_DIR, load_arms, load_golden

# Contrasts worth printing on their own line, as (a, b, what it isolates).
CONTRASTS = [
    ("claude_md", "bare", "documentation alone"),
    ("smritikosh", "bare", "the index alone"),
    ("smritikosh_general", "claude_md", "the index, documentation held fixed"),
    ("smritikosh_tuned", "smritikosh_general", "the tuned retrieval prompt"),
    ("both", "smritikosh", "adding graphify to smritikosh"),
]

# The CSV is the audit trail and carries everything, including the cache
# split that the rendered report deliberately leaves out. Flat on purpose:
# one row per run, no nesting, pivotable in a spreadsheet.
CSV_COLUMNS = [
    # identity
    "run_id",
    "question_id",
    # The text travels with every row. A CSV whose rows say only "q03" is
    # unreadable without the golden file open beside it.
    "question",
    "category",
    "arm",
    "repeat",
    "model",
    "exit_code",
    "provider_error",
    # quality
    "coverage",
    "citation_hit",
    "success",
    "coverage_spread",
    "unverified_quotes",
    "covered_facts",
    # economics
    "cost_total",
    "cost_per_covered_fact",
    "cost_input",
    "cost_output",
    "cost_cacheRead",
    "cost_cacheWrite",
    "wall_ms",
    "turns",
    # context composition: the mechanism
    "ctx_peak_tok",
    "ctx_system_tok",
    "ctx_tool_result_tok",
    "ctx_assistant_tok",
    "ctx_share_results",
    "max_result_chars",
    # tokens
    "tokens_input",
    "tokens_output",
    "tokens_cacheRead",
    "tokens_cacheWrite",
    # behaviour
    "tool_calls_total",
    "tool_smritikosh",
    "tool_graphify",
    "tool_read",
    "tool_grep_find",
    "tool_other",
    "answer_chars",
    "n_citations",
]


def load_rows() -> list[dict]:
    rows = []
    for record_path in sorted(RUNS_DIR.glob("*/record.json")):
        record = json.loads(record_path.read_text())
        grade_path = record_path.parent / "grade.json"
        grade = json.loads(grade_path.read_text()) if grade_path.exists() else {}
        counts = record.get("tool_counts", {})
        tokens = record.get("tokens", {})
        cost = record.get("cost", {})
        ctx = record.get("context", {})

        # Derived here rather than stored, so one definition exists and old
        # records never carry a stale version of it.
        tally = grade.get("tally") or {}
        covered = tally.get("covered", 0) + 0.5 * tally.get("partial", 0)
        total_cost = cost.get("total")
        # Cost per covered fact folds spend and quality into one number: the
        # cheap arm that answers nothing and the correct arm that burns ten
        # dollars both look bad, which is right.
        per_fact = (
            round(total_cost / covered, 4)
            if total_cost is not None and covered
            else None
        )
        peak = ctx.get("peak_tok") or 0
        rows.append(
            {
                "run_id": record["run_id"],
                "question_id": record.get("question_id"),
                "question": record.get("question"),
                "category": record.get("category"),
                "arm": record.get("arm"),
                "repeat": record.get("repeat"),
                "model": record.get("model"),
                "exit_code": record.get("exit_code"),
                "provider_error": record.get("provider_error"),
                "coverage": grade.get("coverage"),
                "citation_hit": grade.get("citation_hit"),
                "success": grade.get("success"),
                "coverage_spread": grade.get("coverage_spread"),
                "unverified_quotes": grade.get("unverified_quotes"),
                "covered_facts": covered or None,
                "cost_total": total_cost,
                "cost_per_covered_fact": per_fact,
                "cost_input": cost.get("input"),
                "cost_output": cost.get("output"),
                "cost_cacheRead": cost.get("cacheRead"),
                "cost_cacheWrite": cost.get("cacheWrite"),
                "wall_ms": record.get("wall_ms"),
                "turns": record.get("turns"),
                "ctx_peak_tok": peak or None,
                "ctx_system_tok": ctx.get("system_tok"),
                "ctx_tool_result_tok": ctx.get("tool_result_tok"),
                "ctx_assistant_tok": ctx.get("assistant_tok"),
                "ctx_share_results": (
                    round((ctx.get("tool_result_tok") or 0) / peak, 3) if peak else None
                ),
                "max_result_chars": record.get("max_result_chars"),
                "tokens_input": tokens.get("input", 0),
                "tokens_output": tokens.get("output", 0),
                "tokens_cacheRead": tokens.get("cacheRead", 0),
                "tokens_cacheWrite": tokens.get("cacheWrite", 0),
                "tool_calls_total": len(record.get("tool_calls", [])),
                "tool_smritikosh": counts.get("smritikosh", 0),
                "tool_graphify": counts.get("graphify", 0),
                "tool_read": counts.get("read", 0),
                "tool_grep_find": counts.get("grep_find", 0),
                "tool_other": counts.get("other", 0),
                "answer_chars": len(record.get("answer") or ""),
                "n_citations": len(record.get("citations", [])),
                "_record": record,
                "_grade": grade,
            }
        )
    return rows


def med(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def iqr(values) -> tuple[float, float] | None:
    values = sorted(v for v in values if v is not None)
    if len(values) < 4:
        return (values[0], values[-1]) if values else None
    quartiles = statistics.quantiles(values, n=4)
    return quartiles[0], quartiles[2]


def fmt(value, spec=".2f") -> str:
    return "—" if value is None else format(value, spec)


def arm_table(rows: list[dict], arms: list[str]) -> list[str]:
    """Eight columns, chosen so every one is a claim a reader can check.

    The cacheRead/cacheWrite split, wall time and raw token counts live in
    runs.csv, not here. They are real, and on the page they invite the wrong
    conclusion: fresh input tokens are typically 0 % of the bill, so a column
    of them reads as "this run was almost free".
    """
    out = [
        "| arm | n | coverage | success | cost $ | $/fact | ctx peak "
        "| results tok | results % | tools |",
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            continue
        successes = [r for r in group if r["success"] is not None]
        rate = (
            sum(1 for r in successes if r["success"]) / len(successes)
            if successes
            else None
        )
        share = med(r["ctx_share_results"] for r in group)
        out.append(
            "| `{arm}` | {n} | {cov} | {succ} | {cost} | {perfact} | {peak} "
            "| {results} | {share} | {tools} |".format(
                arm=arm,
                n=len(group),
                cov=fmt(med(r["coverage"] for r in group)),
                succ=fmt(rate, ".0%") if rate is not None else "—",
                cost=fmt(med(r["cost_total"] for r in group), ".4f"),
                perfact=fmt(med(r["cost_per_covered_fact"] for r in group), ".4f"),
                peak=fmt(med(r["ctx_peak_tok"] for r in group), ",.0f"),
                results=fmt(med(r["ctx_tool_result_tok"] for r in group), ",.0f"),
                share=fmt(share, ".0%") if share is not None else "—",
                tools=fmt(med(r["tool_calls_total"] for r in group), ".0f"),
            )
        )
    out.append("")
    out.append(
        "`$/fact` is cost divided by covered facts: spend and quality in one "
        "number. `results tok` is how much of the context was tool output, "
        "which is what an agent re-reads every turn and therefore what it "
        "mostly pays for."
    )
    return out


def mechanism_table(rows: list[dict], arms: list[str]) -> list[str]:
    """Where each arm's context went, and the largest single dump into it.

    `max result` is the point of the table. grep itself is cheap; the `read`
    it leads to is not, and a 27 000-character file fetched to answer a
    ten-line question is the cost this experiment is about. A median hides
    that, a maximum does not.
    """
    out = [
        "| arm | system tok | results tok | assistant tok | peak tok | max result ch |",
        "| --- | --: | --: | --: | --: | --: |",
    ]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            continue
        out.append(
            f"| `{arm}` "
            f"| {fmt(med(r['ctx_system_tok'] for r in group), ',.0f')} "
            f"| {fmt(med(r['ctx_tool_result_tok'] for r in group), ',.0f')} "
            f"| {fmt(med(r['ctx_assistant_tok'] for r in group), ',.0f')} "
            f"| {fmt(med(r['ctx_peak_tok'] for r in group), ',.0f')} "
            f"| {fmt(max((r['max_result_chars'] or 0) for r in group), ',.0f')} |"
        )
    out.append("")
    out.append(
        "`system tok` is the fixed overhead an arm pays before answering "
        "anything: its tool schemas plus any appended docs. An arm with a "
        "larger system prompt has to earn it back in smaller tool results."
    )
    return out


def breakeven_table(rows: list[dict], focus: str, baseline: str) -> list[str]:
    """Where the retrieval arm's up-front cost starts paying for itself.

    An index arm pays a larger system prompt before answering anything and
    recovers it through smaller tool results. Whether that trade wins depends
    on how many turns the conversation runs, which makes "36 % cheaper" a
    half-truth \u2014 true at this length, not at every length.

    Cost per turn is crude on purpose: total divided by turns. It is enough
    to show the direction and to say honestly when the tool is not worth it.
    """
    mine = [r for r in rows if r["arm"] == focus and r["turns"]]
    theirs = [r for r in rows if r["arm"] == baseline and r["turns"]]
    if not mine or not theirs:
        return []

    my_system = med(r["ctx_system_tok"] for r in mine) or 0
    their_system = med(r["ctx_system_tok"] for r in theirs) or 0
    my_per_turn = med(r["cost_total"] / r["turns"] for r in mine)
    their_per_turn = med(r["cost_total"] / r["turns"] for r in theirs)
    my_turns = med(r["turns"] for r in mine)

    out = [
        f"`{focus}` carries {my_system:,.0f} tokens of system prompt against "
        f"`{baseline}`'s {their_system:,.0f} \u2014 an up-front cost of "
        f"{my_system - their_system:+,.0f} tokens it has to earn back.",
        "",
        "| | " + f"`{baseline}`" + " | " + f"`{focus}`" + " |",
        "| --- | --: | --: |",
        f"| system prompt (tok) | {their_system:,.0f} | {my_system:,.0f} |",
        f"| cost per turn | {fmt(their_per_turn, '.4f')} | {fmt(my_per_turn, '.4f')} |",
        f"| turns observed | {fmt(med(r['turns'] for r in theirs), '.0f')} "
        f"| {fmt(my_turns, '.0f')} |",
        "",
    ]
    if my_per_turn is not None and their_per_turn is not None:
        if my_per_turn < their_per_turn:
            out.append(
                f"Cheaper per turn, so the trade already pays at the "
                f"{my_turns:.0f} turns observed. The break-even sits below "
                "that; a shorter-question tier would locate it."
            )
        else:
            out.append(
                "More expensive per turn: on these questions the larger "
                "system prompt is not being earned back."
            )
    return out


def contrast_table(rows: list[dict]) -> list[str]:
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    out = [
        "| contrast | isolates | Δ coverage | Δ cost $ | Δ wall s |",
        "| --- | --- | --: | --: | --: |",
    ]
    for a, b, what in CONTRASTS:
        if a not in by_arm or b not in by_arm:
            continue
        deltas = []
        for field, scale in (("coverage", 1), ("cost_total", 1), ("wall_ms", 1 / 1000)):
            left, right = (
                med(r[field] for r in by_arm[a]),
                med(r[field] for r in by_arm[b]),
            )
            deltas.append(
                None if left is None or right is None else (left - right) * scale
            )
        out.append(
            f"| `{a}` − `{b}` | {what} | {fmt(deltas[0], '+.2f')} "
            f"| {fmt(deltas[1], '+.4f')} | {fmt(deltas[2], '+.0f')} |"
        )
    return out


def per_question_matrix(rows: list[dict], arms: list[str], questions) -> list[str]:
    out = [
        "| question | category | " + " | ".join(f"`{a}`" for a in arms) + " |",
        "| --- | --- |" + " --: |" * len(arms),
    ]
    for question in questions:
        cells = []
        for arm in arms:
            group = [
                r for r in rows if r["question_id"] == question.id and r["arm"] == arm
            ]
            coverage = med(r["coverage"] for r in group)
            hit = any(r["citation_hit"] for r in group)
            cells.append(
                "—" if coverage is None else f"{coverage:.2f}{'✓' if hit else ''}"
            )
        out.append(
            f"| {question.id} | {question.category} | " + " | ".join(cells) + " |"
        )
    out.append("")
    out.append("Median coverage; `✓` marks a citation overlapping a gold range.")
    out.append("")
    # Folded rather than inline: the table is what people scan, but the ids
    # in it mean nothing without the questions one click away.
    out.append("<details><summary>The questions</summary>")
    out.append("")
    for question in questions:
        out.append(f"**{question.id}** ({question.category}) - {question.question}")
        out.append("")
    out.append("</details>")
    return out


def losses(rows: list[dict], focus: str = "smritikosh_tuned") -> list[dict]:
    """Questions where the focus arm was beaten, and by what.

    Beaten means lower coverage, or equal coverage at higher cost. The
    second case matters: an arm that gets there by reading ten files is not
    doing as well as one that read two, even when both answer correctly.
    """
    by_question = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_question[row["question_id"]][row["arm"]].append(row)

    out = []
    for question_id, arms in sorted(by_question.items()):
        if focus not in arms:
            continue
        mine_cov = med(r["coverage"] for r in arms[focus])
        mine_cost = med(r["cost_total"] for r in arms[focus])
        if mine_cov is None:
            continue
        for arm, group in arms.items():
            if arm == focus:
                continue
            their_cov = med(r["coverage"] for r in group)
            their_cost = med(r["cost_total"] for r in group)
            if their_cov is None:
                continue
            worse_quality = their_cov > mine_cov + 1e-9
            same_quality_cheaper = (
                abs(their_cov - mine_cov) <= 1e-9
                and their_cost is not None
                and mine_cost is not None
                and their_cost < mine_cost * 0.8
            )
            if worse_quality or same_quality_cheaper:
                out.append(
                    {
                        "question_id": question_id,
                        "question": arms[focus][0]["_record"].get("question", ""),
                        "lost_to": arm,
                        "reason": "coverage" if worse_quality else "cost",
                        "mine_coverage": mine_cov,
                        "their_coverage": their_cov,
                        "mine_cost": mine_cost,
                        "their_cost": their_cost,
                        "my_queries": [
                            q
                            for r in arms[focus]
                            for q in r["_record"].get("retrieval_queries", [])
                        ][:12],
                        "their_files": sorted(
                            {
                                f
                                for r in group
                                for f in r["_record"].get("files_read", [])
                            }
                        )[:12],
                    }
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="golden")
    parser.add_argument("--out", help="output directory (default results/<date>)")
    parser.add_argument(
        "--focus", default="smritikosh_tuned", help="arm the loss list is about"
    )
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        raise SystemExit(f"no records under {RUNS_DIR}")

    golden = load_golden(args.golden)
    all_arms, defaults = load_arms()
    order = [a.id for a in all_arms if any(r["arm"] == a.id for r in rows)]

    out_dir = (
        pathlib.Path(args.out)
        if args.out
        else RESULTS_DIR / dt.date.today().isoformat()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Report what actually ran, not what the config currently says. A sweep
    # can be re-pointed at another provider between running and reporting,
    # and a header naming the wrong model misattributes the whole result.
    observed = {r["model"] for r in rows if r["model"]}
    agent_model = ", ".join(sorted(observed)) or defaults.get("model", "?")
    judged_by = {
        (r["_grade"] or {}).get("judge_model")
        for r in rows
        if (r["_grade"] or {}).get("judge_model")
    }
    judge_model = ", ".join(sorted(judged_by)) or defaults.get("judge_model", "?")

    with (out_dir / "runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    provider_errors = [
        r["run_id"] for r in rows if (r["_record"] or {}).get("provider_error")
    ]
    ungraded = sum(1 for r in rows if r["coverage"] is None)
    unstable = [r["run_id"] for r in rows if (r["coverage_spread"] or 0) > 0.2]
    bad_quotes = [r["run_id"] for r in rows if (r["unverified_quotes"] or 0) > 0]
    loss_list = losses(rows, args.focus)

    md = [
        f"# Eval report — {dt.date.today().isoformat()}",
        "",
        f"Commit `{golden.commit}` · agent `{agent_model}` · "
        f"judge `{judge_model}` · {len(rows)} runs",
        "",
        "## Per-arm medians",
        "",
        *arm_table(rows, order),
        "",
        "## Where the context went",
        "",
        *mechanism_table(rows, order),
        "",
        "## Contrasts",
        "",
        "Each row changes one knob. This is the part of the report that makes a claim.",
        "",
        *contrast_table(rows),
        "",
        "## Break-even",
        "",
        *(breakeven_table(rows, args.focus, "bare") or ["Not enough arms to compare."]),
        "",
        "## Per question",
        "",
        *per_question_matrix(rows, order, golden.questions),
        "",
        f"## Where `{args.focus}` lost",
        "",
    ]

    if not loss_list:
        md += [f"No losses recorded for `{args.focus}` in this run.", ""]
    else:
        md += ["This is the improvement backlog, not a scoreboard.", ""]
        for loss in loss_list:
            md += [
                f"### {loss['question_id']} — lost to `{loss['lost_to']}` "
                f"on {loss['reason']}",
                "",
                f"> {loss['question']}",
                "",
                f"- coverage: {loss['mine_coverage']:.2f} "
                f"vs {loss['their_coverage']:.2f}",
                f"- cost: {fmt(loss['mine_cost'], '.4f')} "
                f"vs {fmt(loss['their_cost'], '.4f')}",
                "- queries issued: "
                + (", ".join(repr(q) for q in loss["my_queries"]) or "—"),
                "- what the winner read: " + (", ".join(loss["their_files"]) or "—"),
                "",
            ]

    if provider_errors or ungraded or unstable or bad_quotes:
        md += ["## Caveats", ""]
        if provider_errors:
            md.append(
                f"- provider errors (not agent failures): "
                f"{', '.join(provider_errors)}. Re-run those cells; they are "
                "excluded from grading, not scored as zero."
            )
        if ungraded:
            md.append(f"- {ungraded} run(s) ungraded — run `grade.py`.")
        if unstable:
            md.append(
                f"- unstable grades (spread > 0.2): {', '.join(unstable)}. "
                "Treat those rows as provisional."
            )
        if bad_quotes:
            md.append(
                "- verdicts whose quote is absent from the answer: "
                f"{', '.join(bad_quotes)}. The judge invented support; "
                "re-grade before trusting them."
            )
        md.append("")

    (out_dir / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out_dir}/report.md and runs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
