"""Render the run tree as one self-contained HTML page.

The tables in `report.md` are correct and hard to absorb. A number like
"tool results were 70 % of context" is a fact you have to assemble in your
head; the same thing as a stacked bar is a fact you see. This script exists
for that difference and nothing else — it computes no new numbers, it only
draws the ones `extract.py` and `grade.py` already produced.

Design constraints, in order:

* **One file, no network.** No CDN, no framework, no build step. The output
  is a single .html you can mail to someone or open on a plane. A visualiser
  that breaks when a CDN moves is worse than a table.
* **No JavaScript for anything structural.** Collapsing uses `<details>`,
  which browsers implement natively. JS would be one more thing to debug
  when the only job is drawing rectangles.
* **Every pixel traces to a field.** Bar widths are proportions of real
  measurements. Nothing is smoothed, interpolated or rounded for looks.

Usage:
    python visualise.py                      # -> results/<date>/report.html
    python visualise.py --open               # and open it
    python visualise.py --out /tmp/eval.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pathlib
import subprocess
import sys

from config import RESULTS_DIR, RUNS_DIR, load_golden
from report import load_rows, med

# One colour per tool kind, used everywhere the kind appears so the legend
# only has to be learned once.
KIND_COLOURS = {
    "smritikosh": "#4EFAAA",
    "graphify": "#8B9DFF",
    "read": "#F59E0B",
    "grep_find": "#EF4444",
    "other": "#6B7280",
}

# Cost segments. Deliberately ordered cacheWrite first, because on a real
# run it is the largest share and the ordering should not bury it.
COST_SEGMENTS = [
    ("cacheWrite", "#F59E0B", "cache write"),
    ("output", "#4EFAAA", "output"),
    ("cacheRead", "#8B9DFF", "cache read"),
    ("input", "#6B7280", "fresh input"),
]

# Context segments, in the order they stack. System first because it is the
# fixed cost an arm pays before doing anything.
CONTEXT_SEGMENTS = [
    ("system_tok", "#8B9DFF", "system + tools"),
    ("tool_result_tok", "#EF4444", "tool results"),
    ("assistant_tok", "#4EFAAA", "assistant"),
    ("user_tok", "#6B7280", "user"),
]

CSS = """
:root {
  --bg: #0B0B0C; --panel: #141416; --line: #26262B;
  --fg: #E4E4E7; --dim: #8A8A93; --accent: #4EFAAA; --warn: #F59E0B;
  --bad: #EF4444;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.5rem 4rem; background: var(--bg); color: var(--fg);
  font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1rem; margin: 2.5rem 0 .4rem; color: var(--fg);
     text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
h2 + .note { margin-top: 0; }
.sub { color: var(--dim); margin: 0 0 .5rem; }
.note { color: var(--dim); margin: .3rem 0 1rem; font-size: 13px; }
.panel { background: var(--panel); border: 1px solid var(--line);
         border-radius: 8px; padding: 1rem 1.1rem; margin-bottom: .6rem; }

/* ---- arm cards ---- */
.cards { display: grid; gap: .6rem;
         grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.card-name { color: var(--accent); font-weight: 600; margin-bottom: .6rem; }
.stat { display: flex; justify-content: space-between; gap: 1rem;
        padding: .18rem 0; }
.stat span:first-child { color: var(--dim); }
.stat span:last-child { font-variant-numeric: tabular-nums; }
.win { color: var(--accent); }
.lose { color: var(--bad); }

/* ---- bars ---- */
.row { display: grid; grid-template-columns: 170px 1fr 110px;
       align-items: center; gap: .7rem; padding: .25rem 0; }
.row-label { color: var(--dim); overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }
.row-value { text-align: right; color: var(--dim);
             font-variant-numeric: tabular-nums; }
.bar { display: flex; height: 20px; border-radius: 3px; overflow: hidden;
       background: #1C1C20; }
.seg { height: 100%; }
.seg:hover { filter: brightness(1.35); }

/* ---- legend ---- */
.legend { display: flex; flex-wrap: wrap; gap: 1rem; margin: .7rem 0 0;
          color: var(--dim); font-size: 12px; }
.swatch { display: inline-block; width: 10px; height: 10px;
          border-radius: 2px; margin-right: .35rem; vertical-align: baseline; }

/* ---- grid ---- */
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: .4rem .55rem;
         border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--dim); font-weight: 500; }
td.q { color: var(--fg); }
.cell-good { color: var(--accent); }
.cell-mid { color: var(--warn); }
.cell-bad { color: var(--bad); }
.cell-none { color: #3A3A42; }

/* ---- run detail ---- */
details { border: 1px solid var(--line); border-radius: 6px;
          margin-bottom: .4rem; background: var(--panel); }
summary { cursor: pointer; padding: .6rem .8rem; list-style: none;
          display: flex; justify-content: space-between; gap: 1rem; }
summary::-webkit-details-marker { display: none; }
summary:hover { background: #18181B; }
.summary-id { color: var(--fg); }
.summary-meta { color: var(--dim); font-variant-numeric: tabular-nums; }
.detail { padding: 0 .8rem .9rem; border-top: 1px solid var(--line); }
.detail h4 { margin: .9rem 0 .35rem; color: var(--dim); font-size: 12px;
             text-transform: uppercase; letter-spacing: .06em; font-weight: 500; }
.chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.chip { background: #1C1C20; border: 1px solid var(--line); border-radius: 3px;
        padding: .1rem .45rem; font-size: 12px; color: var(--dim); }
.answer { background: #0E0E10; border: 1px solid var(--line); border-radius: 4px;
          padding: .7rem .8rem; white-space: pre-wrap; max-height: 300px;
          overflow: auto; font-size: 12.5px; color: #C8C8CF; }
.verdict { display: grid; grid-template-columns: 86px 1fr; gap: .6rem;
           padding: .25rem 0; font-size: 12.5px; }
.v-covered { color: var(--accent); }
.v-partial { color: var(--warn); }
.v-missing { color: var(--bad); }
.timeline { display: flex; flex-direction: column; gap: 2px; }
.tl-row { display: grid; grid-template-columns: 92px 1fr 70px;
          gap: .5rem; align-items: center; font-size: 12px; }
.tl-kind { color: var(--dim); }
.tl-size { text-align: right; color: var(--dim);
           font-variant-numeric: tabular-nums; }
.tl-bar { height: 13px; border-radius: 2px; min-width: 2px; }
/* ---- filter ---- */
.filters { display: flex; gap: .8rem; flex-wrap: wrap; align-items: center;
           margin-bottom: .7rem; }
.filters label { color: var(--dim); font-size: 12px; }
select { background: #1C1C20; color: var(--fg); border: 1px solid var(--line);
         border-radius: 4px; padding: .3rem .5rem; font: inherit;
         font-size: 12.5px; }
.hidden { display: none; }
.count { color: var(--dim); font-size: 12px; }

/* ---- backlog ---- */
.verdict-box { border-left: 3px solid var(--accent); background: #0E1512;
               padding: .8rem 1rem; border-radius: 4px; margin-bottom: 1rem;
               color: #CFE8DC; }
.item { border: 1px solid var(--line); border-radius: 6px; padding: .8rem 1rem;
        margin-bottom: .5rem; background: var(--panel); }
.item-head { display: flex; gap: .5rem; align-items: baseline;
             margin-bottom: .45rem; flex-wrap: wrap; }
.sev { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
       padding: .1rem .4rem; border-radius: 3px; font-weight: 600; }
.sev-high { background: #3A1518; color: #FFB4B4; }
.sev-medium { background: #3A2A10; color: #FFD79A; }
.sev-low { background: #1C2A30; color: #A8D0E0; }
.kind { color: var(--dim); font-size: 12px; }
.item-gap { color: var(--fg); margin-bottom: .4rem; }
.item-field { color: var(--dim); font-size: 12.5px; margin-top: .3rem; }
.item-field b { color: #A8A8B2; font-weight: 500; }

.banner { border-left: 3px solid var(--bad); padding: .5rem .8rem;
          background: #1A1012; color: #FFB4B4; border-radius: 4px;
          margin-bottom: .6rem; }
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def bar(segments: list[tuple[float, str, str]], total: float) -> str:
    """A stacked bar. `segments` is (value, colour, tooltip)."""
    if total <= 0:
        return '<div class="bar"></div>'
    parts = []
    for value, colour, label in segments:
        if value <= 0:
            continue
        pct = value / total * 100
        parts.append(
            f'<div class="seg" style="width:{pct:.3f}%;background:{colour}" '
            f'title="{esc(label)}"></div>'
        )
    return f'<div class="bar">{"".join(parts)}</div>'


def arm_cards(rows: list[dict], arms: list[str]) -> str:
    """Headline numbers per arm, with the best value in each row highlighted.

    Highlighting is per-metric and direction-aware, because "lower is better"
    for cost and "higher is better" for coverage, and a reader should not
    have to remember which is which.
    """
    metrics = [
        ("coverage", "coverage", "high", ".2f"),
        ("success", "success", "high", ".0%"),
        ("cost_total", "cost $", "low", ".4f"),
        ("cost_per_covered_fact", "$ / fact", "low", ".4f"),
        ("ctx_peak_tok", "ctx peak", "low", ",.0f"),
        ("ctx_tool_result_tok", "results tok", "low", ",.0f"),
        ("tool_calls_total", "tool calls", "low", ".0f"),
        ("wall_ms", "wall s", "low", ".0f"),
    ]

    values: dict[str, dict[str, float | None]] = {}
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        done = [r for r in group if r["success"] is not None]
        values[arm] = {
            "coverage": med(r["coverage"] for r in group),
            "success": (
                sum(1 for r in done if r["success"]) / len(done) if done else None
            ),
            "cost_total": med(r["cost_total"] for r in group),
            "cost_per_covered_fact": med(r["cost_per_covered_fact"] for r in group),
            "ctx_peak_tok": med(r["ctx_peak_tok"] for r in group),
            "ctx_tool_result_tok": med(r["ctx_tool_result_tok"] for r in group),
            "tool_calls_total": med(r["tool_calls_total"] for r in group),
            "wall_ms": (med(r["wall_ms"] for r in group) or 0) / 1000 or None,
            "_n": len(group),
        }

    best: dict[str, float | None] = {}
    for key, _, direction, _spec in metrics:
        present = [values[a][key] for a in arms if values[a][key] is not None]
        if present:
            best[key] = max(present) if direction == "high" else min(present)

    cards = []
    for arm in arms:
        lines = [
            f'<div class="card-name">{esc(arm)}</div>',
            f'<div class="stat"><span>runs</span>'
            f"<span>{values[arm]['_n']}</span></div>",
        ]
        for key, label, _direction, spec in metrics:
            value = values[arm][key]
            if value is None:
                shown, cls = "—", ""
            else:
                shown = format(value, spec)
                # Only mark a winner when arms actually differ.
                differs = len({values[a][key] for a in arms}) > 1
                cls = " win" if differs and value == best.get(key) else ""
            lines.append(
                f'<div class="stat"><span>{esc(label)}</span>'
                f'<span class="{cls.strip()}">{esc(shown)}</span></div>'
            )
        cards.append(f'<div class="panel">{"".join(lines)}</div>')
    return f'<div class="cards">{"".join(cards)}</div>'


def context_bars(rows: list[dict], arms: list[str]) -> str:
    """Stacked context composition, one bar per arm, on a shared scale.

    The shared scale is the whole point. Normalising each bar to 100 % would
    hide that one arm's context is twice the size of another's, which is the
    finding.
    """
    peaks = [med(r["ctx_peak_tok"] for r in rows if r["arm"] == a) or 0 for a in arms]
    scale = max(peaks) or 1

    out = []
    for arm, peak in zip(arms, peaks, strict=True):
        group = [r for r in rows if r["arm"] == arm]
        segments = []
        for key, colour, label in CONTEXT_SEGMENTS:
            # Read from the record rather than the flattened row: user_tok is
            # tiny and has no CSV column, and it should not gain one just to
            # draw a sliver.
            value = (
                med((r["_record"].get("context") or {}).get(key) for r in group) or 0
            )
            if value:
                segments.append((value, colour, f"{label}: {value:,.0f} tok"))
        out.append(
            f'<div class="row">'
            f'<div class="row-label">{esc(arm)}</div>'
            f"{bar(segments, scale)}"
            f'<div class="row-value">{peak:,.0f} tok</div>'
            f"</div>"
        )

    legend = "".join(
        f'<span><span class="swatch" style="background:{colour}"></span>'
        f"{esc(label)}</span>"
        for _key, colour, label in CONTEXT_SEGMENTS
    )
    return f'<div class="panel">{"".join(out)}<div class="legend">{legend}</div></div>'


def cost_bars(rows: list[dict], arms: list[str]) -> str:
    """Cost composition per arm, on a shared scale, with the dollars named.

    The twin of `context_bars`. That one shows what the context was made of;
    this one shows what it cost to carry, and the two are the same story told
    in tokens and then in money. A single "cost $" column cannot show that
    cache writes dominate, which is the fact that makes a smaller context
    worth having.
    """
    totals = [med(r["cost_total"] for r in rows if r["arm"] == a) or 0 for a in arms]
    scale = max(totals) or 1

    out = []
    for arm, total in zip(arms, totals, strict=True):
        group = [r for r in rows if r["arm"] == arm]
        segments = []
        for key, colour, label in COST_SEGMENTS:
            value = med(r[f"cost_{key}"] for r in group) or 0
            if value:
                segments.append(
                    (value, colour, f"{label}: ${value:.4f} ({value / total:.0%})")
                )
        out.append(
            f'<div class="row">'
            f'<div class="row-label">{esc(arm)}</div>'
            f"{bar(segments, scale)}"
            f'<div class="row-value">${total:.4f}</div>'
            f"</div>"
        )

    # Per-arm dollar figures under the bars, because a hover tooltip is not
    # something a reader can scan or copy into a note.
    head = "".join(f"<th>{esc(a)}</th>" for a in arms)

    # The dominant line, per arm, computed once so the table can mark it and
    # the sentence underneath can name it. Everyone guesses `output`.
    dominant: dict[str, str] = {}
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        per_line = {
            key: (med(r[f"cost_{key}"] for r in group) or 0)
            for key, _c, _l in COST_SEGMENTS
        }
        if any(per_line.values()):
            dominant[arm] = max(per_line, key=lambda k: per_line[k])

    body = []
    for key, _colour, label in COST_SEGMENTS:
        cells = []
        for arm in arms:
            group = [r for r in rows if r["arm"] == arm]
            dollars = med(r[f"cost_{key}"] for r in group)
            tokens = med(r[f"tokens_{key}"] for r in group)
            if dollars is None:
                cells.append("<td>—</td>")
                continue
            top = dominant.get(arm) == key
            cells.append(
                f'<td class="{"dominant-cell" if top else ""}">'
                f'<span class="{"dominant" if top else ""}">${dollars:.4f}</span>'
                f'<br><span style="color:#55555E">{tokens or 0:,.0f} tok</span></td>'
            )
        body.append(f"<tr><td>{esc(label)}</td>{''.join(cells)}</tr>")

    # Say the answer in words. A reader should not have to compare four
    # numbers across a row to learn which line is the expensive one.
    labels = {key: label for key, _c, label in COST_SEGMENTS}
    if dominant and len(set(dominant.values())) == 1:
        only = labels[next(iter(dominant.values()))]
        verdict = f"On every arm the largest line is <strong>{esc(only)}</strong>."
    elif dominant:
        verdict = "Largest line per arm: " + ", ".join(
            f"<strong>{esc(arm)}</strong> {esc(labels[key])}"
            for arm, key in dominant.items()
        )
    else:
        verdict = "No cost recorded."

    legend = "".join(
        f'<span><span class="swatch" style="background:{colour}"></span>'
        f"{esc(label)}</span>"
        for _key, colour, label in COST_SEGMENTS
    )
    return (
        f'<div class="panel">{"".join(out)}'
        f'<div class="legend">{legend}</div>'
        f'<table style="margin-top:.9rem"><tr><th>line</th>{head}</tr>'
        f"{''.join(body)}</table>"
        f'<div class="note" style="margin:.8rem 0 0">{verdict} '
        "Not output, which is what everyone guesses.</div></div>"
    )


def question_grid(rows: list[dict], arms: list[str], questions) -> str:
    head = "".join(f"<th>{esc(a)}</th>" for a in arms)
    body = []
    for question in questions:
        cells = []
        for arm in arms:
            group = [
                r for r in rows if r["question_id"] == question.id and r["arm"] == arm
            ]
            coverage = med(r["coverage"] for r in group)
            if coverage is None:
                cells.append('<td class="cell-none">·</td>')
                continue
            hit = any(r["citation_hit"] for r in group)
            cls = (
                "cell-good"
                if coverage >= 0.8
                else "cell-mid"
                if coverage >= 0.5
                else "cell-bad"
            )
            mark = " ✓" if hit else ""
            cells.append(f'<td class="{cls}">{coverage:.2f}{mark}</td>')
        body.append(
            f'<tr><td class="q" title="{esc(question.question)}">'
            f'{esc(question.id)} <span style="color:var(--dim)">'
            f"{esc(question.category)}</span></td>{''.join(cells)}</tr>"
        )
    return (
        f'<div class="panel"><table><tr><th>question</th>{head}</tr>'
        f"{''.join(body)}</table></div>"
    )


def tool_timeline(record: dict) -> str:
    """Every tool call as a bar sized by how much it put into the context.

    This is the most useful single picture in the page. A run that greps five
    times and then reads one enormous file looks completely different from a
    run of evenly sized retrieval calls, and no table conveys that.
    """
    calls = record.get("tool_calls") or []
    if not calls:
        return '<div class="note">No tool calls recorded.</div>'
    widest = max(c["result_chars"] for c in calls) or 1

    rows = []
    for call in calls:
        colour = KIND_COLOURS.get(call["kind"], KIND_COLOURS["other"])
        width = max(call["result_chars"] / widest * 100, 0.5)
        hint = call["kind"]
        if call["queries"]:
            hint += ": " + ", ".join(call["queries"][:3])
        rows.append(
            f'<div class="tl-row">'
            f'<div class="tl-kind">{esc(call["kind"])}</div>'
            f'<div class="tl-bar" style="width:{width:.2f}%;background:{colour}" '
            f'title="{esc(hint)}"></div>'
            f'<div class="tl-size">{call["result_chars"]:,}</div>'
            f"</div>"
        )
    return f'<div class="timeline">{"".join(rows)}</div>'


def verdict_list(grade: dict) -> str:
    verdicts = grade.get("verdicts") or []
    facts = grade.get("key_facts") or []
    if not verdicts:
        return '<div class="note">Not graded.</div>'
    out = []
    for verdict in verdicts:
        index = verdict.get("fact_index")
        fact = facts[index] if isinstance(index, int) and index < len(facts) else "?"
        name = verdict.get("verdict", "?")
        flag = "" if verdict.get("quote_ok", True) else " ⚠ unverified quote"
        out.append(
            f'<div class="verdict"><span class="v-{esc(name)}">{esc(name)}</span>'
            f"<span>{esc(fact)}"
            f'<span style="color:var(--bad)">{esc(flag)}</span></span></div>'
        )
    return "".join(out)


def cost_breakdown(record: dict) -> str:
    """Where one run's money went, and the token counts behind it.

    Worth a picture because the shape surprises everyone: fresh input is
    typically a rounding error and cache *writes* are usually the largest
    line. The agent is paying to store and re-read its own accumulated
    reading, not to think about the question.
    """
    cost = record.get("cost") or {}
    tokens = record.get("tokens") or {}
    total = cost.get("total") or 0
    if not total:
        return '<div class="note">No cost recorded.</div>'

    segments = [
        (cost.get(key) or 0, colour, f"{label}: ${cost.get(key) or 0:.4f}")
        for key, colour, label in COST_SEGMENTS
    ]
    rows = [bar(segments, total)]

    # Which line actually dominates. Naming it is the whole point: everyone
    # assumes it is `output`, and it is almost always `cacheWrite`.
    biggest = max(COST_SEGMENTS, key=lambda s: cost.get(s[0]) or 0)[0]

    parts = []
    for key, colour, label in COST_SEGMENTS:
        dollars = cost.get(key) or 0
        mark = ' class="dominant"' if key == biggest else ""
        tag = '<span class="dominant-tag">largest</span>' if key == biggest else ""
        parts.append(
            f'<span><span class="swatch" style="background:{colour}"></span>'
            f"<span{mark}>{esc(label)} ${dollars:.4f}</span> "
            f'<span style="color:#55555E">'
            f"({dollars / total * 100:.0f}%, {tokens.get(key, 0):,} tok)</span>"
            f"{tag}</span>"
        )
    rows.append(f'<div class="legend">{"".join(parts)}</div>')
    rows.append(
        f'<div class="note" style="margin:.5rem 0 0">total '
        f"<strong>${total:.4f}</strong> over {record.get('turns', 0)} turns "
        f"· {sum(tokens.values()):,} tokens handled</div>"
    )
    return "".join(rows)


def backlog_section(gaps: dict) -> str:
    """The sweep-wide verdict and ranked backlog, if analyze.py has run.

    Read from gaps.json rather than recomputed: this page draws, it does not
    judge, and a second opinion rendered here would be a third one to
    reconcile.
    """
    if not gaps:
        return (
            '<div class="note">No <code>gaps.json</code> yet — run '
            "<code>analyze.py</code> to generate the backlog.</div>"
        )

    out = []
    summary = gaps.get("summary") or {}
    if summary.get("verdict"):
        out.append(f'<div class="verdict-box">{esc(summary["verdict"])}</div>')

    backlog = summary.get("backlog") or []
    if not backlog:
        out.append(
            '<div class="note">The backlog pass returned no items. On a small '
            "sample that is the expected answer, not a clean bill of health.</div>"
        )
    for index, item in enumerate(backlog, 1):
        severity = str(item.get("severity", "low")).lower()
        out.append(
            f'<div class="item">'
            f'<div class="item-head">'
            f'<span class="sev sev-{esc(severity)}">{esc(severity)}</span>'
            f'<span class="kind">{esc(item.get("kind", "?"))}</span>'
            f'<span class="kind">· affects {esc(item.get("affects", "?"))}</span>'
            f"</div>"
            f'<div class="item-gap">{index}. {esc(item.get("gap", ""))}</div>'
            f'<div class="item-field"><b>Evidence:</b> '
            f"{esc(item.get('evidence', '—'))}</div>"
            f'<div class="item-field"><b>Fix:</b> '
            f"{esc(item.get('fix', '—'))}</div>"
            f"</div>"
        )

    findings = gaps.get("findings") or []
    if findings:
        causes: dict[str, int] = {}
        for finding in findings:
            cause = (finding.get("diagnosis") or {}).get("cause", "unknown")
            causes[cause] = causes.get(cause, 0) + 1
        chips = "".join(
            f'<span class="chip">{esc(cause)} ×{count}</span>'
            for cause, count in sorted(causes.items(), key=lambda kv: -kv[1])
        )
        out.append(
            f'<div class="note" style="margin-top:1rem">Per-loss causes '
            f'diagnosed: </div><div class="chips">{chips}</div>'
        )
    return "".join(out)


def run_details(rows: list[dict]) -> str:
    out = []
    for row in sorted(rows, key=lambda r: (r["question_id"] or "", r["arm"] or "")):
        record, grade = row["_record"], row["_grade"]
        coverage = (
            f"{row['coverage']:.2f}" if row["coverage"] is not None else "ungraded"
        )
        summary = (
            f'<summary><span class="summary-id">{esc(row["run_id"])}</span>'
            f'<span class="summary-meta">'
            f"cov {coverage} · ${row['cost_total'] or 0:.4f} · "
            f"ctx {row['ctx_peak_tok'] or 0:,} · "
            f"{row['tool_calls_total']} tools</span></summary>"
        )

        blocks = []
        if record.get("provider_error"):
            blocks.append(
                f'<div class="banner">Provider error, not an agent failure: '
                f"{esc(record['provider_error'])}</div>"
            )
        blocks.append(f"<h4>Question</h4><div>{esc(row['question'] or '')}</div>")

        queries = record.get("retrieval_queries") or []
        if queries:
            chips = "".join(f'<span class="chip">{esc(q)}</span>' for q in queries)
            blocks.append(f'<h4>Queries issued</h4><div class="chips">{chips}</div>')

        blocks.append(f"<h4>Tool calls</h4>{tool_timeline(record)}")
        blocks.append(f"<h4>Cost</h4>{cost_breakdown(record)}")
        blocks.append(f"<h4>Key facts</h4>{verdict_list(grade)}")

        citations = record.get("citations") or []
        chips = (
            "".join(f'<span class="chip">{esc(c)}</span>' for c in citations)
            or '<span class="chip">none</span>'
        )
        blocks.append(f'<h4>Citations</h4><div class="chips">{chips}</div>')
        blocks.append(
            f'<h4>Answer</h4><div class="answer">'
            f"{esc(record.get('answer') or '')}</div>"
        )

        out.append(
            f'<details class="run" '
            f'data-question="{esc(row["question_id"] or "")}" '
            f'data-arm="{esc(row["arm"] or "")}">'
            f'{summary}<div class="detail">{"".join(blocks)}</div></details>'
        )
    return "".join(out)


def filter_controls(rows: list[dict], questions, arms: list[str]) -> str:
    """Two dropdowns that hide run rows client-side.

    This is the one place the page uses JavaScript, and it is deliberately an
    *enhancement*: every run is already in the DOM, so with scripting off the
    page still shows everything, just unfiltered. Nothing is fetched, nothing
    is rendered lazily, and Ctrl-F still finds every run.
    """
    seen = {r["question_id"] for r in rows}
    question_options = "".join(
        f'<option value="{esc(q.id)}">{esc(q.id)} — {esc(q.category)}</option>'
        for q in questions
        if q.id in seen
    )
    arm_options = "".join(f'<option value="{esc(a)}">{esc(a)}</option>' for a in arms)

    return f"""<div class="filters">
  <label for="fq">question</label>
  <select id="fq"><option value="">all ({len(seen)})</option>
    {question_options}</select>
  <label for="fa">arm</label>
  <select id="fa"><option value="">all ({len(arms)})</option>
    {arm_options}</select>
  <span class="count" id="fcount"></span>
</div>
<script>
(function () {{
  const q = document.getElementById("fq");
  const a = document.getElementById("fa");
  const count = document.getElementById("fcount");
  const runs = Array.from(document.querySelectorAll("details.run"));
  function apply() {{
    let shown = 0;
    for (const run of runs) {{
      const ok =
        (!q.value || run.dataset.question === q.value) &&
        (!a.value || run.dataset.arm === a.value);
      run.classList.toggle("hidden", !ok);
      if (ok) shown++;
    }}
    count.textContent = shown + " of " + runs.length + " runs";
  }}
  q.addEventListener("change", apply);
  a.addEventListener("change", apply);
  apply();
}})();
</script>"""


def build(rows: list[dict], golden, arms: list[str], gaps: dict) -> str:
    models = ", ".join(sorted({r["model"] for r in rows if r["model"]})) or "?"
    judges = ", ".join(
        sorted({(r["_grade"] or {}).get("judge_model") for r in rows if r["_grade"]})
    )
    errors = [r["run_id"] for r in rows if (r["_record"] or {}).get("provider_error")]

    banner = ""
    if errors:
        banner = (
            f'<div class="banner">{len(errors)} run(s) hit a provider error and '
            f"are excluded from grading: {esc(', '.join(errors))}</div>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smritikosh eval · {esc(dt.date.today().isoformat())}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>Agent eval</h1>
<p class="sub">commit <strong>{esc(golden.commit)}</strong> ·
agent {esc(models)} · judge {esc(judges or "—")} ·
{len(rows)} runs · {esc(dt.datetime.now().strftime("%Y-%m-%d %H:%M"))}</p>
{banner}

<h2>Arms</h2>
<p class="note">Median per arm. Green marks the best value in a row, with
direction accounted for: lower is better for cost, higher for coverage.</p>
{arm_cards(rows, arms)}

<h2>Where the context went</h2>
<p class="note">Bars share one scale, so a longer bar is a bigger context.
Tool results are the segment that grows without bound, and an agent re-reads
its whole context every turn, so that segment is most of what a run costs.</p>
{context_bars(rows, arms)}

<h2>Where the money went</h2>
<p class="note">The same story in dollars. Cache <em>writes</em> are usually
the largest line and fresh input a rounding error: a run pays to store and
re-read its own accumulated reading, not to think about the question. Cache
reads move the most tokens by far and cost the least per token, which is why
a smaller context is worth more than a cheaper model.</p>
{cost_bars(rows, arms)}

<h2>Per question</h2>
<p class="note">Median coverage. ✓ marks a citation overlapping a gold
range. Hover a question to read it.</p>
{question_grid(rows, arms, golden.questions)}

<h2>Summary and backlog</h2>
<p class="note">Written by a second, stronger model that sees the whole
sweep: the per-loss analysis can only diagnose one question at a time, and a
pattern across questions is a different kind of finding. It is allowed to
conclude that the eval itself is the problem.</p>
{backlog_section(gaps)}

<h2>Runs</h2>
<p class="note">One row per run; open for the queries it issued, every tool
call sized by what it returned, where the money went, and the graded facts.
Hover any bar for detail.</p>
{filter_controls(rows, golden.questions, arms)}
{run_details(rows)}
</div></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="golden")
    parser.add_argument(
        "--out", help="output path (default results/<date>/report.html)"
    )
    parser.add_argument("--gaps", help="path to gaps.json (default alongside out)")
    parser.add_argument("--open", action="store_true", help="open in a browser")
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        raise SystemExit(f"no records under {RUNS_DIR} — run run.py first")

    golden = load_golden(args.golden)
    # Arm order follows first appearance in the data, not arms.yaml, so a
    # partial sweep does not render a row of empty columns.
    arms: list[str] = []
    for row in rows:
        if row["arm"] and row["arm"] not in arms:
            arms.append(row["arm"])

    out = (
        pathlib.Path(args.out)
        if args.out
        else RESULTS_DIR / dt.date.today().isoformat() / "report.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # The backlog is read, never recomputed. This page draws; analyze.py
    # judges. A second opinion rendered here would be a third one to
    # reconcile.
    gaps_path = pathlib.Path(args.gaps) if args.gaps else out.parent / "gaps.json"
    gaps: dict = {}
    if gaps_path.exists():
        loaded = json.loads(gaps_path.read_text())
        # Older sweeps wrote a bare list of findings before the summary pass
        # existed; accept both shapes rather than crashing on history.
        gaps = loaded if isinstance(loaded, dict) else {"findings": loaded}

    out.write_text(build(rows, golden, arms, gaps))

    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.0f} KB, {len(rows)} runs, {len(arms)} arms)")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
