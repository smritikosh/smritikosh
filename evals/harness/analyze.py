"""Second judge: turn losses into named retrieval gaps.

`report.py` says smritikosh lost q03 to `claude_md`. That is a fact, not a
task. This pass reads the full dossier for each loss — the queries issued,
the ranges retrieval returned, the files the winner read instead, the facts
the answer missed — and names the mechanism, so the next change to
retrieval has a target.

This judge is the opposite of the grading judge by design. The grader is
blind to arms and costs because knowing them would bias a score. The
analyst must see everything, because the diagnosis *is* the relationship
between them. Its output is prose for humans to argue with, never a number
that feeds back into the tables.

Usage:
    python analyze.py                         # analyze every loss
    python analyze.py --focus smritikosh      # a different arm under test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

from config import RESULTS_DIR, RUNS_DIR, load_arms, load_golden
from report import load_rows, losses, med

SUMMARY_PROMPT = """You are writing the standing improvement backlog for a
code-search tool called Smritikosh, based on a full benchmark sweep.

You are shown every question in the set, how each context arm scored on it,
what each arm spent, and the per-loss diagnoses already made. Unlike the
per-loss analysis, you see the whole sweep at once, so your job is the thing
no single question can show: patterns across questions.

Write:

1. "verdict" - 3 to 4 sentences. What does this sweep actually establish?
   Name the strongest claim the data supports, and the strongest objection a
   sceptic would raise. If the sample is too small to establish anything, say
   that first and plainly.

2. "backlog" - ranked list of concrete gaps, highest value first. Each entry:
   - "gap": one line naming the problem
   - "evidence": the specific questions and numbers that show it. Cite
     question ids. No evidence means no entry.
   - "severity": "high" | "medium" | "low"
   - "affects": which question categories or arms it appears in
   - "fix": one concrete change, to Smritikosh or to the eval itself
   - "kind": "retrieval" | "ranking" | "chunking" | "prompt" | "eval_design"

Rules:
- Patterns, not restatements. "q03 lost" is already known; "every cross-file
  question loses, and all of them needed a caller lookup" is the insight.
- A gap in the *eval* counts and should be named. An unfair question, a leaky
  docs file, or a judge grading style over substance is worth more than a
  ranking tweak.
- Rank by expected value, not by how interesting the bug is.
- An empty backlog is a valid answer when the sweep shows no clear gap. Say
  so rather than inventing work.

Return ONLY JSON, no prose, no code fences:

{"verdict": "...", "backlog": [{"gap": "...", "evidence": "...",
"severity": "high", "affects": "...", "fix": "...", "kind": "retrieval"}]}

SWEEP:
{sweep}
"""


ANALYST_PROMPT = """You are diagnosing retrieval failures in a code-search tool called
Smritikosh. It indexes a repository and returns `path:start-end` ranges for a
coding agent to read.

Below is one question where an agent using Smritikosh produced a worse or more
expensive answer than an agent using a different setup. You have both sides:
the queries Smritikosh was given, the ranges it returned, what the winning
setup read instead, and which key facts the Smritikosh answer missed.

Name the mechanism of the failure. Choose the single best-fitting cause:

- "retrieval_miss": the needed code was never returned by any query.
- "ranking": the needed code was returned, but buried below noise.
- "query_formulation": the agent asked for the wrong things; the index would
  have found it given better facets.
- "chunk_boundary": the right file came back with the wrong span — too narrow
  to contain the fact, or too wide to be usable.
- "agent_behaviour": retrieval returned the right evidence and the agent still
  failed to use it.
- "not_a_tool_problem": the loss is explained by answer style, verbosity, or
  luck rather than retrieval.

Be specific and concrete. If the evidence does not support a confident
diagnosis, say so in "confidence". Never invent file contents.

Return ONLY JSON, no prose, no code fences:

{"cause": "...", "confidence": "high|medium|low", "mechanism": "2-3 sentences
on what actually happened", "fix": "one concrete change to Smritikosh, or
'none' if this is not a tool problem"}

DOSSIER:
{dossier}
"""


def dossier_for(loss: dict, rows: list[dict], question, focus: str) -> dict:
    mine = [
        r for r in rows if r["question_id"] == loss["question_id"] and r["arm"] == focus
    ]
    theirs = [
        r
        for r in rows
        if r["question_id"] == loss["question_id"] and r["arm"] == loss["lost_to"]
    ]

    def missed_facts(group):
        out = []
        for row in group:
            for verdict in (row["_grade"] or {}).get("verdicts", []):
                if verdict.get("verdict") in {"missing", "partial"}:
                    index = verdict.get("fact_index")
                    if isinstance(index, int) and 0 <= index < len(question.key_facts):
                        out.append(
                            f"[{verdict['verdict']}] {question.key_facts[index]}"
                        )
        return sorted(set(out))

    return {
        "question": question.question,
        "category": question.category,
        "gold_citations": question.gold_citations,
        "lost_to": loss["lost_to"],
        "reason": loss["reason"],
        "smritikosh": {
            "coverage": loss["mine_coverage"],
            "cost": loss["mine_cost"],
            "queries_issued": sorted(
                {q for r in mine for q in r["_record"]["retrieval_queries"]}
            ),
            "ranges_returned": sorted(
                {x for r in mine for x in r["_record"]["ranges_returned"]}
            )[:40],
            "files_read": sorted({f for r in mine for f in r["_record"]["files_read"]}),
            "citations_emitted": sorted(
                {c for r in mine for c in r["_record"]["citations"]}
            ),
            "facts_missed": missed_facts(mine),
            "tool_counts": mine[0]["_record"]["tool_counts"] if mine else {},
        },
        "winner": {
            "coverage": loss["their_coverage"],
            "cost": loss["their_cost"],
            "files_read": sorted(
                {f for r in theirs for f in r["_record"]["files_read"]}
            ),
            "citations_emitted": sorted(
                {c for r in theirs for c in r["_record"]["citations"]}
            ),
            "tool_counts": theirs[0]["_record"]["tool_counts"] if theirs else {},
        },
    }


def sweep_digest(rows: list[dict], golden, findings: list[dict]) -> dict:
    """Compact the whole sweep into something a model can reason over.

    Not the raw rows: forty records with full tool-call detail would be most
    of a context window spent on noise. This keeps one line per question per
    arm, plus the diagnoses already made, which is what a pattern lives in.
    """
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    arms = {}
    for arm, group in by_arm.items():
        done = [r for r in group if r["success"] is not None]
        arms[arm] = {
            "runs": len(group),
            "median_coverage": med(r["coverage"] for r in group),
            "success_rate": (
                round(sum(1 for r in done if r["success"]) / len(done), 3)
                if done
                else None
            ),
            "median_cost": med(r["cost_total"] for r in group),
            "median_cost_per_covered_fact": med(
                r["cost_per_covered_fact"] for r in group
            ),
            "median_ctx_peak_tok": med(r["ctx_peak_tok"] for r in group),
            "median_tool_result_tok": med(r["ctx_tool_result_tok"] for r in group),
            "median_tool_calls": med(r["tool_calls_total"] for r in group),
        }

    questions = []
    for question in golden.questions:
        per_arm = {}
        for arm, group in by_arm.items():
            cells = [r for r in group if r["question_id"] == question.id]
            if not cells:
                continue
            per_arm[arm] = {
                "coverage": med(r["coverage"] for r in cells),
                "citation_hit": any(r["citation_hit"] for r in cells),
                "cost": med(r["cost_total"] for r in cells),
                "tool_calls": med(r["tool_calls_total"] for r in cells),
            }
        if per_arm:
            questions.append(
                {
                    "id": question.id,
                    "category": question.category,
                    "question": question.question,
                    "arms": per_arm,
                }
            )

    return {
        "commit": golden.commit,
        "n_questions": len(questions),
        "arms": arms,
        "questions": questions,
        "loss_diagnoses": [
            {
                "question_id": f["question_id"],
                "lost_to": f["lost_to"],
                "reason": f["reason"],
                "cause": f["diagnosis"].get("cause"),
                "mechanism": f["diagnosis"].get("mechanism"),
                "facts_missed": f["dossier"]["smritikosh"]["facts_missed"],
            }
            for f in findings
        ],
    }


def ask(model: str, prompt: str, timeout: int, thinking: str = "low") -> dict:
    """Run one tool-less model call and parse a single JSON object out of it."""
    proc = subprocess.run(
        [
            "pi",
            "-p",
            "--mode",
            "json",
            "--no-session",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-mcp",
            "--model",
            model,
            "--thinking",
            thinking,
            "--",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    text = ""
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("type") == "message_end"
            and (event.get("message") or {}).get("role") == "assistant"
        ):
            content = event["message"].get("content")
            if isinstance(content, list):
                chunk = "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                if chunk.strip():
                    text = chunk
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"analyst returned no JSON: {text[:300]!r}")
    return json.loads(text[start : end + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="golden")
    parser.add_argument("--focus", default="smritikosh_tuned")
    parser.add_argument("--model", help="override the per-loss analyst model")
    parser.add_argument(
        "--summary-model",
        help="model for the sweep-wide backlog (defaults to the analyst model)",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="skip the sweep-wide backlog"
    )
    parser.add_argument("--out", help="output directory (default results/<date>)")
    parser.add_argument(
        "--only", metavar="QUESTION_ID", help="analyze losses for one question only"
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        raise SystemExit(f"no records under {RUNS_DIR}")
    golden = load_golden(args.golden)
    questions = {q.id: q for q in golden.questions}
    _, defaults = load_arms()
    # The analyst reasons over a long dossier, so it falls back to the agent
    # model rather than the cheap grading one.
    model = args.model or defaults.get("analyst_model") or defaults.get("model")
    # The backlog is written once per sweep and is the artefact a human acts
    # on, so it is worth the strongest model available rather than the
    # cheapest that parses.
    summary_model = args.summary_model or defaults.get("summary_model") or model

    loss_list = losses(rows, args.focus)
    if args.only:
        loss_list = [loss for loss in loss_list if loss["question_id"] == args.only]
    out_dir = (
        pathlib.Path(args.out)
        if args.out
        else RESULTS_DIR / dt.date.today().isoformat()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # No losses is not nothing to report: the backlog pass still runs, because
    # "this sweep establishes very little, the sample is too small" is itself
    # a finding worth writing down.
    if not loss_list:
        print(f"no losses for {args.focus}")

    findings = []
    for loss in loss_list:
        question = questions[loss["question_id"]]
        dossier = dossier_for(loss, rows, question, args.focus)
        print(
            f"analyzing {loss['question_id']} vs {loss['lost_to']} ... ",
            end="",
            flush=True,
        )
        try:
            prompt = ANALYST_PROMPT.replace("{dossier}", json.dumps(dossier, indent=2))
            verdict = ask(model, prompt, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"failed: {exc}")
            continue
        print(verdict.get("cause", "?"))
        findings.append({**loss, "dossier": dossier, "diagnosis": verdict})

    # ---- the standing backlog -------------------------------------------
    # The per-loss pass sees one question at a time, which is the right scope
    # for diagnosing a single failure and the wrong one for deciding what to
    # work on. This pass sees the whole sweep, so it can say "every
    # cross-file question loses" \u2014 a sentence no single dossier contains.
    summary: dict = {}
    if not args.no_summary:
        print("writing the backlog ... ", end="", flush=True)
        try:
            prompt = SUMMARY_PROMPT.replace(
                "{sweep}", json.dumps(sweep_digest(rows, golden, findings), indent=2)
            )
            # A higher thinking level here than for a single loss: this call
            # is once per sweep and is the one a human acts on.
            summary = ask(summary_model, prompt, args.timeout * 2, thinking="medium")
            print(f"{len(summary.get('backlog', []))} item(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"failed: {exc}")

    (out_dir / "gaps.json").write_text(
        json.dumps(
            {
                "focus": args.focus,
                "analyst_model": model,
                "summary_model": summary_model,
                "summary": summary,
                "findings": findings,
            },
            indent=2,
        )
    )

    by_cause: dict[str, int] = {}
    for finding in findings:
        cause = finding["diagnosis"].get("cause", "unknown")
        by_cause[cause] = by_cause.get(cause, 0) + 1

    md = [
        f"# Retrieval gaps — {dt.date.today().isoformat()}",
        "",
        f"Arm under test: `{args.focus}` · analyst `{model}` · "
        f"backlog `{summary_model}` · {len(findings)} loss(es)",
        "",
    ]

    if summary.get("verdict"):
        md += ["## Verdict", "", summary["verdict"], ""]

    backlog = summary.get("backlog") or []
    if backlog:
        md += [
            "## Backlog",
            "",
            "Ranked by expected value. Each item cites the questions that show "
            "it; an item without evidence should not be here.",
            "",
            "| # | severity | kind | gap |",
            "| --: | --- | --- | --- |",
        ]
        for index, item in enumerate(backlog, 1):
            md.append(
                f"| {index} | {item.get('severity', '?')} "
                f"| {item.get('kind', '?')} | {item.get('gap', '')} |"
            )
        md.append("")
        for index, item in enumerate(backlog, 1):
            md += [
                f"### {index}. {item.get('gap', '')}",
                "",
                f"**{item.get('severity', '?')}** · {item.get('kind', '?')} "
                f"· affects {item.get('affects', '?')}",
                "",
                "*Evidence:* " + str(item.get("evidence", "—")),
                "",
                "*Fix:* " + str(item.get("fix", "—")),
                "",
            ]
    elif summary:
        md += [
            "## Backlog",
            "",
            "The backlog pass returned no items. Taken at face value that means "
            "the sweep shows no clear gap — which on a small sample is the "
            "expected answer, not a clean bill of health.",
            "",
        ]

    md += [
        "## Causes",
        "",
        "| cause | count |",
        "| --- | --: |",
        *[f"| {c} | {n} |" for c, n in sorted(by_cause.items(), key=lambda kv: -kv[1])],
        "",
        "## Per-loss findings",
        "",
    ]
    for finding in findings:
        diagnosis = finding["diagnosis"]
        mine = finding["dossier"]["smritikosh"]
        md += [
            f"### {finding['question_id']} — lost to `{finding['lost_to']}` "
            f"({finding['reason']})",
            "",
            f"**{diagnosis.get('cause')}** · confidence {diagnosis.get('confidence')}",
            "",
            diagnosis.get("mechanism", ""),
            "",
            f"**Fix:** {diagnosis.get('fix')}",
            "",
            "- queries issued: "
            + (", ".join(repr(q) for q in mine["queries_issued"]) or "—"),
            f"- facts missed: "
            f"{'; '.join(finding['dossier']['smritikosh']['facts_missed']) or '—'}",
            f"- winner read: "
            f"{', '.join(finding['dossier']['winner']['files_read']) or '—'}",
            "",
        ]

    (out_dir / "gaps.md").write_text("\n".join(md))
    print(f"\nwrote {out_dir}/gaps.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
