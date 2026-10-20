"""Turn one pi run into one `record.json`.

Everything downstream — grading, reporting, gap analysis — reads records,
never raw sessions. That boundary matters: a run costs money, parsing is
free, so parsing must be re-runnable without re-running the agent.

Two files are read per run, for different reasons:

* `session/*.jsonl` is authoritative for usage and cost. Pi writes usage on
  assistant messages, on standalone `usage` entries (cache warming), and on
  compaction entries; a total that ignores the latter two under-reports what
  the run actually cost.
* `stream.jsonl` (`--mode json`) is authoritative for tool calls, because
  `tool_execution_end` carries `durationMs` and the full result, neither of
  which survives into the session file in a usable form.

Usage:
    python extract.py                 # every run under evals/runs
    python extract.py q01__bare__r1   # one run
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import shlex
import sys

from config import CITATION_RE, RUNS_DIR

# Tool-call taxonomy. The question these counts answer is "how did the agent
# find things", so bash is not one bucket: a bash call running `smritikosh
# explore search` is retrieval, and one running `rg` is agentic grep. Lumping
# them together would make the central comparison unmeasurable.
# Split by where the search text sits in the argument list. For grep and
# friends the first positional argument is the pattern; for find and ls it
# is the directory to look in, and the pattern (if any) hides behind -name.
# Recording a directory as if it were a query would make the gap analyst
# think the agent searched for "smritikosh/indexing/".
PATTERN_FIRST_CMDS = {"grep", "rg", "ag", "ack"}
PATH_FIRST_CMDS = {"find", "fd", "ls", "tree", "glob"}
GREP_CMDS = PATTERN_FIRST_CMDS | PATH_FIRST_CMDS
READ_CMDS = {"cat", "head", "tail", "sed", "awk", "less", "more", "bat", "nl"}
SEPARATORS = {"&&", "||", "|", ";", "(", ")", "{", "}"}


def _blocks(content) -> list[dict]:
    """Normalise message content to a list of blocks.

    Pi stores user content as a plain string and assistant content as a list
    of blocks; callers should not have to care which.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)]


def _content_chars(content) -> int:
    """Size of a message's content as the model receives it."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(json.dumps(b)) for b in content)
    return 0


def _text_of(message: dict) -> str:
    return "\n".join(
        b.get("text", "")
        for b in _blocks(message.get("content"))
        if b.get("type") == "text"
    ).strip()


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed run can leave a half-written final line. One truncated
            # record should not cost us the rest of the run.
            continue
    return out


def _session_path(run_dir: pathlib.Path) -> pathlib.Path | None:
    candidates = sorted((run_dir / "session").glob("*.jsonl"))
    return candidates[-1] if candidates else None


def _parse_ts(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) / 1000.0
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _segments(command: str) -> list[list[str]]:
    """Split a bash command into its pipeline segments, as token lists.

    Classification must look at *command position* only. The first version
    of this matched the substring "smritikosh" anywhere in the command, and
    promptly scored `find smritikosh -name '*.py'` in the bare arm as two
    retrieval calls - the repository is named after the tool, so its name
    appears in half the paths in the tree. A metric that counts greps as
    retrieval would make the headline comparison meaningless.

    Splitting on separators also means `cd repo && rg foo | head` reports
    rg rather than cd.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    segments: list[list[str]] = []
    current: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in SEPARATORS:
            if current:
                segments.append(current)
            current = []
            continue
        if not current and token in {"cd", "pushd", "export"}:
            skip_next = True  # `cd somewhere && real_command`
            continue
        if (
            not current
            and "=" in token
            and not token.startswith("-")
            and "/" not in token.split("=")[0]
        ):
            continue  # FOO=bar prefix; the program is the next token
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _programs(command: str) -> list[str]:
    """Just the program names, in order."""
    return [pathlib.Path(s[0]).name for s in _segments(command) if s]


def _first_command(command: str) -> str:
    programs = _programs(command)
    return programs[0] if programs else ""


def classify_tool(name: str, args: dict) -> str:
    if name.startswith("smritikosh"):
        return "smritikosh"
    if name.startswith("graphify"):
        return "graphify"
    if name == "read":
        return "read"
    if name != "bash":
        return "other"

    programs = _programs(str(args.get("command", "")))
    # Retrieval wins when a command pipes one into the other:
    # `smritikosh explore search x | head` is a retrieval call.
    if "smritikosh" in programs:
        return "smritikosh"
    if "graphify" in programs:
        return "graphify"
    if any(p in GREP_CMDS for p in programs):
        return "grep_find"
    if any(p in READ_CMDS for p in programs):
        return "read"
    return "other"


SUBCOMMANDS = {"explore", "search", "query", "path", "explain", "chunks", "index"}
# Flags whose *value* is a filter rather than a query term, so the value
# following them is the search text we want.
PATTERN_FLAGS = {"-name", "-iname", "-e", "--regexp", "--include"}


def extract_queries(name: str, args: dict) -> list[str]:
    """What the agent actually asked for, in its own words.

    The single most valuable field for gap analysis: a loss is only
    actionable once you can see the query that failed to surface the answer.

    Worth being fussy here. A first pass took every bare token of a bash
    command, which filled the dossier with `|`, `head` and `-f` - noise that
    makes the gap analyst guess. Only the search text counts: the facets
    handed to a retrieval tool, or the pattern handed to grep.
    """
    if name.startswith(("smritikosh", "graphify")) and isinstance(
        args.get("queries"), list
    ):
        return [str(q) for q in args["queries"]]
    if name != "bash":
        return []

    out: list[str] = []
    for segment in _segments(str(args.get("command", ""))):
        if not segment:
            continue
        program = pathlib.Path(segment[0]).name
        rest = segment[1:]

        if program in {"smritikosh", "graphify"}:
            # Every positional argument that is not a subcommand, a path or
            # a line number is a facet the agent chose to search for.
            for token in rest:
                if token.startswith("-") or token in SUBCOMMANDS:
                    continue
                if token.isdigit() or "/" in token or "." in token:
                    continue
                out.append(token)
        elif program in GREP_CMDS:
            take_next = False
            for token in rest:
                if take_next:
                    out.append(token)
                    break
                if token in PATTERN_FLAGS:
                    take_next = True  # the pattern is the next token
                    continue
                if token.startswith("-"):
                    continue
                # A bare positional is the pattern for grep, and the
                # directory to walk for find. Stop at it only for grep;
                # find's pattern still lies ahead, behind -name.
                if program in PATTERN_FIRST_CMDS:
                    out.append(token)
                    break
    return out[:8]


def result_text(result) -> str:
    if isinstance(result, dict):
        return "\n".join(
            b.get("text", "")
            for b in _blocks(result.get("content"))
            if b.get("type") == "text"
        )
    if isinstance(result, str):
        return result
    return ""


def extract_ranges(text: str) -> list[str]:
    """`PATH:START-END` locations appearing in a tool result.

    What retrieval actually handed back, as opposed to what the agent then
    chose to cite. The difference between the two is itself a finding: the
    right range returned but not cited is a presentation problem, not a
    retrieval one.
    """
    seen, out = set(), []
    for m in CITATION_RE.finditer(text or ""):
        token = (
            f"{m.group('path')}:{m.group('start')}-{m.group('end') or m.group('start')}"
        )
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out[:60]


def extract_run(run_dir: pathlib.Path) -> dict:
    meta = (
        json.loads((run_dir / "meta.json").read_text())
        if (run_dir / "meta.json").exists()
        else {}
    )
    sub = (
        json.loads((run_dir / "subprocess.json").read_text())
        if (run_dir / "subprocess.json").exists()
        else {}
    )

    session_entries = (
        _read_jsonl(_session_path(run_dir)) if _session_path(run_dir) else []
    )
    stream = _read_jsonl(run_dir / "stream.jsonl")

    # ---- usage and cost -----------------------------------------------
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    # Cost is kept broken down, not just totalled. On a real run cacheWrite
    # was 46 % of the bill and fresh input was 0 %, which is the opposite of
    # what everyone assumes, and a single total hides it.
    costs = {
        "total": 0.0,
        "input": 0.0,
        "output": 0.0,
        "cacheRead": 0.0,
        "cacheWrite": 0.0,
    }
    # Context composition. An agent re-reads its whole conversation every
    # turn, so what the context is *made of* is what the run costs. Measured
    # in characters here and converted once, at the end.
    ctx_chars = {"system": 0, "tool_result": 0, "assistant": 0, "user": 0}
    turns = 0
    answer = ""
    provider_error: str | None = None
    timestamps: list[float] = []
    model = meta.get("model")

    for entry in session_entries:
        ts = _parse_ts(entry.get("timestamp"))
        if ts:
            timestamps.append(ts)

        usage = None
        if entry.get("type") == "message":
            message = entry.get("message") or {}
            if message.get("role") == "assistant":
                turns += 1
                usage = message.get("usage")
                model = message.get("model") or model
                text = _text_of(message)
                if text:
                    answer = text  # last non-empty assistant text wins
                # A provider error (exhausted credit, rate limit, overload)
                # arrives as an assistant message with stopReason "error"
                # and no text. Left undetected it reads as an agent that
                # answered nothing, which a grader scores as zero and a
                # report prints as a loss. Record it so a billing problem
                # can never masquerade as a retrieval result.
                if message.get("stopReason") == "error":
                    provider_error = (
                        message.get("errorMessage") or "provider returned an error"
                    )
        elif entry.get("type") in {"usage", "compaction", "branch_summary"}:
            usage = entry.get("usage")

        # Context composition, by message role. Sizes are of the text the
        # model actually receives, so tool results count as what was handed
        # back rather than what the tool printed to a terminal.
        if entry.get("type") == "message":
            message = entry.get("message") or {}
            role = message.get("role")
            if role == "system":
                ctx_chars["system"] += sum(
                    len(str(v)) for v in (message.get("sections") or {}).values()
                )
                # Tool schemas are part of the fixed overhead an arm pays.
                ctx_chars["system"] += sum(
                    len(json.dumps(t)) for t in (message.get("toolsAdded") or [])
                )
            elif role == "toolResult":
                ctx_chars["tool_result"] += _content_chars(message.get("content"))
            elif role == "assistant":
                ctx_chars["assistant"] += _content_chars(message.get("content"))
            elif role == "user":
                ctx_chars["user"] += _content_chars(message.get("content"))

        if isinstance(usage, dict):
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
            entry_cost = usage.get("cost") or {}
            for key in costs:
                costs[key] += float(entry_cost.get(key) or 0.0)

    if not answer:
        # Fall back to the stream if the session file is unusable.
        for event in reversed(stream):
            if event.get("type") == "message_end":
                message = event.get("message") or {}
                if message.get("role") == "assistant" and _text_of(message):
                    answer = _text_of(message)
                    break

    # ---- tool calls ----------------------------------------------------
    tool_calls: list[dict] = []
    pending: dict[str, dict] = {}
    for event in stream:
        etype = event.get("type")
        if etype == "tool_execution_start":
            pending[event.get("toolCallId")] = {
                "name": event.get("toolName"),
                "args": event.get("args") or {},
            }
        elif etype == "tool_execution_end":
            call_id = event.get("toolCallId")
            started = pending.pop(call_id, {})
            name = event.get("toolName") or started.get("name") or "unknown"
            args = started.get("args") or event.get("args") or {}
            text = result_text(event.get("result"))
            tool_calls.append(
                {
                    "name": name,
                    "kind": classify_tool(name, args),
                    "args": args,
                    "queries": extract_queries(name, args),
                    "ranges_returned": extract_ranges(text),
                    "result_chars": len(text),
                    "duration_ms": event.get("durationMs"),
                    "is_error": bool(event.get("isError")),
                }
            )

    counts = {"smritikosh": 0, "graphify": 0, "read": 0, "grep_find": 0, "other": 0}
    for call in tool_calls:
        counts[call["kind"]] = counts.get(call["kind"], 0) + 1

    files_read = []
    for call in tool_calls:
        if call["kind"] != "read":
            continue
        path = call["args"].get("path") or call["args"].get("file_path")
        if path:
            files_read.append(str(path))
        elif call["name"] == "bash":
            files_read.extend(
                t
                for t in str(call["args"].get("command", "")).split()
                if "/" in t and "." in t
            )

    retrieval_queries = [q for call in tool_calls for q in call["queries"]]
    ranges_returned = [r for call in tool_calls for r in call["ranges_returned"]]

    # The single largest thing any one tool call pushed into the context.
    # grep is cheap; the `read` that follows it is not, and a 27 000-character
    # file read to find a ten-line answer is the mechanism this whole
    # experiment is about. A median hides that; a maximum does not.
    max_result_chars = max((c["result_chars"] for c in tool_calls), default=0)

    # ---- timing ---------------------------------------------------------
    # Two clocks: the subprocess wall and the span of session timestamps.
    # They agree within a few hundred ms on a healthy run, so the second is
    # not reported \u2014 a number that is only interesting when it is wrong
    # belongs in a warning, not a column.
    session_wall_ms = (
        (max(timestamps) - min(timestamps)) * 1000 if len(timestamps) > 1 else None
    )
    wall_ms = sub.get("wall_ms")
    clock_skew_ms = (
        abs(wall_ms - session_wall_ms)
        if wall_ms is not None and session_wall_ms is not None
        else None
    )
    if clock_skew_ms is not None and clock_skew_ms > 5000:
        print(
            f"  warning: {run_dir.name} clocks disagree by "
            f"{clock_skew_ms / 1000:.1f}s; something stalled outside "
            "the model loop"
        )

    context = {f"{k}_tok": v // 4 for k, v in ctx_chars.items()}
    context["peak_tok"] = sum(ctx_chars.values()) // 4

    # Four groups, and nothing derivable. `answer_chars`, `tokens_fresh`,
    # `tokens_total` and `tool_calls_total` used to live here and were all
    # arithmetic on neighbouring fields: a stored derivation is a definition
    # that can drift out of step with the one in the report. report.py
    # computes them now, and re-runs for free.
    return {
        # -- identity: enough to reproduce this exact cell
        "run_id": run_dir.name,
        "question_id": meta.get("question_id"),
        "question": meta.get("question"),
        "category": meta.get("category"),
        "arm": meta.get("arm"),
        "repeat": meta.get("repeat"),
        "model": model,
        "commit": meta.get("commit"),
        # -- outcome: did it work, and what did it claim
        "exit_code": sub.get("exit_code", meta.get("exit_code")),
        "provider_error": provider_error,
        "answer": answer,
        "citations": extract_ranges(answer),
        # -- economics: what it cost, and where
        "turns": turns,
        "wall_ms": wall_ms,
        "tokens": totals,
        "cost": {k: round(v, 6) for k, v in costs.items()},
        "context": context,
        # -- behaviour: how it got there
        "tool_counts": counts,
        "max_result_chars": max_result_chars,
        "retrieval_queries": retrieval_queries,
        "ranges_returned": ranges_returned,
        "files_read": sorted(set(files_read)),
        "tool_calls": tool_calls,
    }


def main(argv: list[str]) -> int:
    targets = (
        [RUNS_DIR / name for name in argv]
        if argv
        else sorted(p for p in RUNS_DIR.glob("*") if p.is_dir())
    )
    if not targets:
        print(f"no runs found under {RUNS_DIR}", file=sys.stderr)
        return 1

    for run_dir in targets:
        if not (run_dir / "stream.jsonl").exists():
            print(f"skip {run_dir.name}: no stream.jsonl")
            continue
        record = extract_run(run_dir)
        (run_dir / "record.json").write_text(json.dumps(record, indent=2))
        used = "/".join(f"{k}:{v}" for k, v in record["tool_counts"].items() if v)
        print(
            f"{record['run_id']:<34} "
            f"tools={len(record['tool_calls']):>3} ({used}) "
            f"cost=${record['cost']['total']:.4f} "
            f"ctx={record['context']['peak_tok']:>6} "
            f"results={record['context']['tool_result_tok']:>6} "
            f"wall={(record['wall_ms'] or 0) / 1000:.1f}s "
            f"cites={len(record['citations'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
