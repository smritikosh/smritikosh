"""Grade answers: key-fact coverage by judge, citation hit by arithmetic.

Two halves, deliberately different in kind.

**Citation hit is deterministic.** Does any `PATH:START-END` in the answer
overlap a gold range? That is interval arithmetic, so no model is consulted
and the result is identical on every pass.

**Coverage is judged, and judged blind.** The judge sees the question, the
key facts, and the answer. It does not see the arm, the cost, the tool
calls, or the other arms' answers. A judge that knows which answer came
from the tool under test will find reasons to like it; withholding that is
cheaper than correcting for it afterwards.

Every verdict must carry a verbatim quote from the answer. This is the
anti-hallucination device for the grader itself: a quote can be checked by
string search, and `--verify-quotes` does exactly that, flagging any verdict
whose supporting quote is not actually present in the answer being graded.

    coverage = (covered + 0.5 * partial) / len(key_facts)
    success  = coverage >= 0.8 and citation_hit

Usage:
    python grade.py                     # grade every ungraded run
    python grade.py --passes 2          # stability check: two judge passes
    python grade.py --redo
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from config import CITATION_RE, RUNS_DIR, load_arms, load_golden

SUCCESS_COVERAGE = 0.8

JUDGE_PROMPT = """You are grading one answer about a codebase against a list of
key facts.

For each key fact, decide whether the ANSWER states it:

- "covered": the answer states the fact, or an unambiguous paraphrase of it.
- "partial": the answer gestures at the fact but omits the specific part
  that makes it a fact (a number, a name, a condition), or states it with a
  hedge that leaves it unclear.
- "missing": the answer does not state it, or states something contradicting it.

Rules:
- Judge only against the ANSWER text given. Do not use your own knowledge of
  this codebase, and do not reward an answer for being plausible.
- Wording may differ; meaning must not. A different identifier name is a
  different fact.
- Every verdict needs "quote": a verbatim substring copied from the ANSWER
  that justifies it. For "missing", use an empty string.
- Extra correct detail is neither rewarded nor penalised. Only the listed
  facts count.

Return ONLY a JSON object, no prose and no code fences:

{"verdicts": [{"fact_index": 0, "verdict": "covered", "quote": "..."}]}

QUESTION:
{question}

KEY FACTS:
{facts}

ANSWER:
{answer}
"""


def judge_once(
    model: str, question: str, facts: list[str], answer: str, timeout: int
) -> dict:
    prompt = (
        JUDGE_PROMPT.replace("{question}", question)
        .replace("{facts}", "\n".join(f"[{i}] {f}" for i, f in enumerate(facts)))
        .replace("{answer}", answer)
    )
    proc = subprocess.run(
        [
            "pi",
            "-p",
            "--mode",
            "json",
            "--no-session",
            # The judge reasons over text it was handed; a judge with tools
            # could go read the repository and grade against what it finds
            # there instead of against the answer.
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-mcp",
            "--model",
            model,
            "--thinking",
            "off",
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
        if event.get("type") == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, list):
                    chunk = "".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    if chunk.strip():
                        text = chunk
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"judge returned no JSON object: {text[:300]!r}")
    return json.loads(match.group())


def quote_is_present(quote: str, answer: str) -> bool:
    """Is the judge's supporting quote really in the answer?

    The point is to catch a judge that invented support, not to police
    punctuation. Judges legitimately elide the middle of a long passage with
    an ellipsis, so each fragment is checked separately and all must appear.
    Whitespace is normalised for the same reason: a quote rewrapped across
    lines is still the answer's own words.

    Markdown emphasis is stripped on both sides for the same reason. Agents
    write **bold** conclusions and judges quote the words inside them; the
    asterisks are formatting, and failing a verdict over them would train us
    to ignore this check \u2014 which is worse than not having it.

    Fragments shorter than eight characters are ignored, because "the" is
    present in every answer and proves nothing.
    """
    quote = (quote or "").strip()
    if not quote:
        return True  # "missing" verdicts carry no quote by design

    def normalise(text: str) -> str:
        return " ".join(text.replace("*", "").replace("_", "").replace("`", "").split())

    haystack = normalise(answer)
    fragments = [f for f in re.split(r"\.\.\.|\u2026", quote) if len(f.strip()) >= 8]
    if not fragments:
        return normalise(quote) in haystack
    return all(normalise(f) in haystack for f in fragments)


def score(verdicts: list[dict], total: int) -> tuple[float, dict]:
    tally = {"covered": 0, "partial": 0, "missing": 0}
    seen = set()
    for verdict in verdicts:
        index = verdict.get("fact_index")
        name = verdict.get("verdict")
        if index in seen or name not in tally:
            continue
        seen.add(index)
        tally[name] += 1
    # Facts the judge silently dropped count as missing. Omission must never
    # be cheaper than an honest "missing".
    tally["missing"] += total - len(seen)
    coverage = (tally["covered"] + 0.5 * tally["partial"]) / total if total else 0.0
    return coverage, tally


def citation_hit(answer: str, gold: list[dict]) -> tuple[bool, list[str]]:
    """True when a cited range overlaps a gold range in the same file.

    Overlap, not containment: an answer citing lines 70-120 for a gold range
    of 76-107 has pointed the reader at the right code, and demanding exact
    bounds would grade formatting rather than retrieval.
    """
    matched = []
    for match in CITATION_RE.finditer(answer or ""):
        path = match.group("path")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        for want in gold:
            if not str(want["path"]).endswith(path) and not path.endswith(
                str(want["path"])
            ):
                continue
            if start <= int(want["end"]) and int(want["start"]) <= end:
                matched.append(f"{path}:{start}-{end}")
                break
    return bool(matched), matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="golden")
    parser.add_argument(
        "--model",
        help="judge model (overrides EVAL_JUDGE_MODEL and arms.yaml)",
    )
    parser.add_argument("--passes", type=int, default=1, help="judge passes per answer")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--only", metavar="RUN_ID", help="grade exactly one run, e.g. q08__bare__r1"
    )
    parser.add_argument("--redo", action="store_true")
    parser.add_argument(
        "--verify-quotes",
        action="store_true",
        default=True,
        help="flag verdicts whose quote is absent from the answer",
    )
    args = parser.parse_args()

    golden = load_golden(args.golden)
    _, defaults = load_arms()
    model = args.model or defaults.get("judge_model")
    questions = {q.id: q for q in golden.questions}

    records = (
        [RUNS_DIR / args.only / "record.json"]
        if args.only
        else sorted(RUNS_DIR.glob("*/record.json"))
    )
    records = [p for p in records if p.exists()]
    if not records:
        raise SystemExit(f"no records under {RUNS_DIR} — run run.py first")

    graded = 0
    for record_path in records:
        run_dir = record_path.parent
        grade_path = run_dir / "grade.json"
        if grade_path.exists() and not args.redo:
            continue

        record = json.loads(record_path.read_text())
        question = questions.get(record.get("question_id"))
        if question is None:
            print(f"skip {run_dir.name}: unknown question")
            continue
        # An answerless run is a zero only if the agent genuinely produced
        # nothing. If the provider errored, grading it zero would blame
        # retrieval for a billing problem, so leave it ungraded and let the
        # report list it as a caveat.
        if record.get("provider_error"):
            print(f"skip {run_dir.name}: provider error, re-run this cell")
            continue
        if not record.get("answer"):
            grade_path.write_text(
                json.dumps(
                    {
                        "run_id": record["run_id"],
                        "coverage": 0.0,
                        "citation_hit": False,
                        "success": False,
                        "error": "empty answer",
                    },
                    indent=2,
                )
            )
            continue

        passes = []
        for _ in range(args.passes):
            try:
                result = judge_once(
                    model,
                    question.question,
                    question.key_facts,
                    record["answer"],
                    args.timeout,
                )
            except Exception as exc:  # noqa: BLE001 — one bad pass must not stop the sweep
                print(f"  judge error on {run_dir.name}: {exc}")
                continue
            verdicts = result.get("verdicts", [])
            if args.verify_quotes:
                for verdict in verdicts:
                    verdict["quote_ok"] = quote_is_present(
                        verdict.get("quote", ""), record["answer"]
                    )
            coverage, tally = score(verdicts, len(question.key_facts))
            passes.append({"coverage": coverage, "tally": tally, "verdicts": verdicts})

        if not passes:
            continue

        coverage = sum(p["coverage"] for p in passes) / len(passes)
        # Disagreement between passes is reported, not averaged away: an
        # unstable grade is a reason to distrust that row.
        spread = max(p["coverage"] for p in passes) - min(p["coverage"] for p in passes)
        hit, matched = citation_hit(record["answer"], question.gold_citations)
        bad_quotes = sum(
            1 for p in passes for v in p["verdicts"] if v.get("quote_ok") is False
        )

        grade = {
            "run_id": record["run_id"],
            "question_id": question.id,
            "question": question.question,
            # Carried alongside the verdicts so a grade can be audited
            # without cross-referencing the golden file by hand.
            "key_facts": question.key_facts,
            "arm": record.get("arm"),
            "judge_model": model,
            "passes": len(passes),
            "coverage": round(coverage, 4),
            "coverage_spread": round(spread, 4),
            "citation_hit": hit,
            "citations_matched": matched,
            "success": coverage >= SUCCESS_COVERAGE and hit,
            "unverified_quotes": bad_quotes,
            "tally": passes[0]["tally"],
            "verdicts": passes[0]["verdicts"],
        }
        grade_path.write_text(json.dumps(grade, indent=2))
        graded += 1
        print(
            f"{record['run_id']:<34} coverage={coverage:.2f} "
            f"cite={'Y' if hit else 'n'} "
            f"{'success' if grade['success'] else '-':<7}"
            f"{f' spread={spread:.2f}' if spread else ''}"
            f"{f' BADQUOTES={bad_quotes}' if bad_quotes else ''}"
        )

    print(f"\ngraded {graded} run(s) with {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
