# The harness

Six scripts that turn questions into a report. The experiment they serve —
which arms exist and why — is described in [`../README.md`](../README.md);
this file is about the code.

## The one rule

```
run.py costs money. Everything downstream of it is free.
```

That is why the stages are separate files communicating through JSON on disk
rather than one script holding everything in memory. A grading bug, a judge
outage, or a change to how the report is laid out costs you nothing: re-run
that stage against the existing run tree. Only `run.py` talks to an agent.

## Who calls what

Nothing calls `run.py` automatically. **You do**, through `make`:

```
you
 └─ make eval-pilot                      (Makefile, repo root)
     ├─ prepare.py      worktrees + indexes
     ├─ run.py          the matrix
     │   └─ arms/<arm>.sh                 one subprocess per cell
     │       └─ _common.sh  → run_pi()
     │           └─ pi -p --mode json     the agent being measured
     ├─ grade.py        → pi --no-tools   the judge
     ├─ report.py       pure Python, no model
     ├─ analyze.py      → pi --no-tools   the gap analyst
     └─ visualise.py    pure Python, no model, no network
```

There is no scheduler, daemon, or CI hook. Each stage is an ordinary script
you can also run by hand, and `make` only chains them in the right order.
That is deliberate: a benchmark that spends money should start when someone
decides it should, not when a timer fires.

Three processes reach a model: the agent (once per cell), the judge (once per
answer), the analyst (once per loss). Everything else is arithmetic.

## The pipeline

```
golden/golden.yaml ─┐
arms/arms.yaml ─────┴──▶ config.py  (read by every stage)

prepare.py ──▶ .worktrees/<profile>/        code at the pinned commit + index
run.py     ──▶ runs/<run_id>/               stream.jsonl, session/, meta.json
extract.py ──▶ runs/<run_id>/record.json    cost, time, tools, queries, answer
grade.py   ──▶ runs/<run_id>/grade.json     coverage, citation hit, verdicts
report.py  ──▶ results/<date>/report.md     medians, contrasts, loss list
analyze.py ──▶ results/<date>/gaps.md       why each loss happened
visualise.py ▶ results/<date>/report.html   the same data, legible
```

`run_id` is `<question>__<arm>__r<repeat>`, e.g. `q03__smritikosh_tuned__r1`.
It is the directory name, the resume key, and the argument to `--only`.

## What each script does

### `config.py` — the shared address book

Loads the two YAML files and hands out `Arm`, `Question` and `Golden`
objects. Exists so nothing else knows where files live or how the YAML is
shaped.

Its one piece of real logic: `docs` may be a single filename or a list, and
`Arm.docs_paths` always returns a list. That is what lets `smritikosh_tuned`
stack `general.md + smritikosh.md` rather than substituting one for the
other — without it, that arm would differ from `smritikosh_general` in two
ways at once and the contrast between them would mean nothing.

### `arms/*.sh` — the desks

Four scripts of about three lines each. Every one sets `ARM_FLAGS` (always
`--no-extensions`) and optionally `ARM_CTX`, then calls `run_pi` from
`_common.sh`, which holds every flag that must be identical across arms.

An arm therefore cannot accidentally differ in a dimension the experiment is
not testing — it has no way to change the model, the tool loadout or the
timeout. Both retrieval tools are CLIs reached through bash, so `smritikosh.sh`
and `graphify.sh` are the same script with a different context file.

### `prepare.py` — set up the rooms

For each index profile: create a detached `git worktree` at the golden
commit, then build the indexes that profile needs, recording build time and
size as setup data.

Two guards matter. It refuses to proceed if an existing worktree sits at the
wrong commit, because the failure mode there is silent answer leakage rather
than an error. And arms sharing an index share a worktree, so three arms
search byte-identical data instead of three separately built indexes.

### `run.py` — the invigilator

```
build the cell list (question x arm x repeat)
shuffle it under a fixed seed
for each cell:
    record.json already there?  -> skip, add its cost to the total
    over --max-spend?           -> stop
    write meta.json             (before running, so a crash still says what it tried)
    exec arms/<arm>.sh with the question in the environment
    extract immediately         -> record.json
```

The ordering is the design. `meta` before, `extract` after: kill it at any
moment and you lose only the cell in flight.

Shuffling is not cosmetic. Running an arm's cells consecutively lets
provider-side prompt caching and time-of-day latency land on whichever arm
went last; a fixed seed keeps that unbiased and still reproducible.

A provider error — exhausted credit, rate limit, overload — is treated as a
failure rather than a result. It arrives as an empty answer, and an empty
answer scored as zero would blame retrieval for a billing problem.

### `extract.py` — the counter

Reads two files per run, because each is authoritative for different things:

- **`session/*.jsonl`** for money. Usage also hides in standalone `usage`
  and `compaction` entries, so a total built only from assistant messages
  under-reports what the run cost.
- **`stream.jsonl`** for tools, because only `tool_execution_end` carries
  `durationMs` and the full result.

Then the fiddly part. `_segments()` splits a bash command at `&& | ;`, drops
`cd` and `FOO=bar` prefixes, and looks at what sits in *command position*.
From that come two fields:

- `tool_counts` — which bucket each call belongs in. Both rules here exist
  because both were wrong first: this repository is *named* `smritikosh`, so
  matching the word anywhere scored `find smritikosh` in the grep-only arm
  as retrieval.
- `retrieval_queries` — the search text, and only that. `grep`'s pattern is
  its first positional argument; `find`'s first positional is a *directory*,
  with the pattern behind `-name`. Recording the directory would tell the gap
  analyst the agent searched for `smritikosh/indexing/`.

`ranges_returned` and `citations` are a deliberate pair: the right range
retrieved but not cited is a presentation problem, not a retrieval one.

### `grade.py` — the blindfolded marker

Two halves, different in kind:

- **`citation_hit`** is interval arithmetic. Does a cited range overlap a
  gold range in the same file? No model, identical every run. Overlap rather
  than containment, because citing 70–120 for gold 76–107 points the reader
  at the right code, and demanding exact bounds would grade formatting.
- **`coverage`** is judged, per fact, by a model that is told the question,
  the facts and the answer — and *not* the arm, the cost or the tool calls. A
  judge that knows which answer came from the tool under test will find
  reasons to like it.

Three hardenings:

| Risk | Guard |
| --- | --- |
| Judge invents support | every verdict needs a verbatim quote, which `quote_is_present()` then string-checks |
| Judge skips a hard fact | facts with no verdict count as missing, so omission is never cheaper than honesty |
| Judge is inconsistent | `--passes 2` records `coverage_spread`; the report flags anything over 0.2 |

`quote_is_present()` is forgiving about elisions, whitespace and markdown
emphasis, and unforgiving about words that are not there. That balance is
the point: a check that fails on `**bold**` teaches you to ignore it, which
is worse than not having it.

### `report.py` — the results sheet

Flattens every `record.json` + `grade.json` into rows, writes `runs.csv`,
and renders four sections: per-arm medians, **the contrasts**, a
per-question grid, and the loss list.

Medians and IQR rather than means, because with few repeats one timeout
drags a mean somewhere misleading.

Its one opinion lives in `losses()`: you have lost if another arm beat your
coverage, **or** matched it for under 80% of your cost. Equal answers at
half the price is a loss worth looking at.

### `analyze.py` — the marker who sees everything

Two passes, at two scales.

**Per loss:** assembles a dossier — queries issued, ranges returned, files
the winner read, facts missed — and asks a model for one of six causes plus
one concrete fix.

**Per sweep:** a second pass sees every question, every arm's score, and all
the per-loss diagnoses at once, and writes the standing backlog. This is the
only pass that can find a *pattern* — "every cross-file question loses, and
all of them needed a caller lookup" is a sentence no single dossier contains.
It gets the strongest model configured (`summary_model`), because it runs
once per sweep and is the artefact a human acts on.

It is explicitly allowed to name the eval itself as the problem, and on the
first two-run smoke test it did: it ranked *"the eval has only 1 question"*
as the highest-severity item and refused to let the sample drive
prioritisation. A backlog that cannot say that is a backlog that will
cheerfully send you off to tune a ranker on n=1.

Deliberately the inverse of `grade.py`. The grader is blind because knowing
the arm would bias a score; the analyst must see everything because the
diagnosis *is* the relationship between those numbers. Its output is prose
to argue with, never a number that feeds back into the tables.

### `test_extract.py` — the alarm

39 tests, aimed only where **a bug yields a plausible number instead of a
crash**: the bash classifier, the query extractor, citation overlap, the
coverage arithmetic, quote verification and provider-error detection. Those
are the failures nobody notices.

```bash
cd evals/harness && ../../.venv/bin/python -m pytest test_extract.py -q --no-cov
```

## Choosing models

Three models, each overridable at three levels. Precedence, loosest last:

```
arms.yaml  <  EVAL_* environment  <  --model on the command line
```

| Role | `arms.yaml` key | Environment | Default |
| --- | --- | --- | --- |
| agent under test | `model` | `EVAL_MODEL` | `openrouter/anthropic/claude-sonnet-4.5` |
| coverage judge | `judge_model` | `EVAL_JUDGE_MODEL` | `openrouter/google/gemini-2.5-flash` |
| gap analyst | `analyst_model` | `EVAL_ANALYST_MODEL` | falls back to `model` |

```bash
make eval-pilot MODEL=openrouter/anthropic/claude-sonnet-4.5
make eval-grade JUDGE_MODEL=openrouter/google/gemini-2.5-flash
EVAL_MODEL=anthropic/claude-sonnet-4-5 python run.py --only q03__bare__r1
```

The environment layer exists because provider access changes for reasons that
have nothing to do with the experiment — an exhausted plan, a key that only
works for one vendor — and none of those are a reason to commit a diff.

`report.py` names the models that **actually ran**, read back from the
records, not whatever the config says now. A sweep re-pointed at another
provider between running and reporting would otherwise be misattributed.

## The record schema

One `record.json` per run, in four groups. **Nothing in it is derivable from
anything else in it** — `answer_chars`, `tokens_fresh`, `tokens_total` and
`tool_calls_total` all used to live here and were cut, because a stored
derivation is a second definition that drifts out of step with the one in the
report. `report.py` computes them, and re-runs for free.

```jsonc
{
  // identity: enough to reproduce this exact cell
  "run_id", "question_id", "question", "category",
  "arm", "repeat", "model", "commit",

  // outcome: did it work, and what did it claim
  "exit_code", "provider_error", "answer", "citations",

  // economics: what it cost, and where
  "turns", "wall_ms",
  "tokens":  { "input", "output", "cacheRead", "cacheWrite" },
  "cost":    { "total", "input", "output", "cacheRead", "cacheWrite" },
  "context": { "system_tok", "tool_result_tok", "assistant_tok",
               "user_tok", "peak_tok" },

  // behaviour: how it got there
  "tool_counts": { "smritikosh", "graphify", "read", "grep_find", "other" },
  "max_result_chars",
  "retrieval_queries", "ranges_returned", "files_read",
  "tool_calls": [ ... ]
}
```

### Why the less obvious fields exist

| field | the question it answers |
| --- | --- |
| `cost.*` broken down | on a real run `cacheWrite` was 46 % of the bill and fresh `input` was 0 %. A single total hides the only thing worth knowing. |
| `context.system_tok` | the arm's fixed entry fee — tool schemas plus appended docs. An arm with a bigger system prompt must earn it back. |
| `context.tool_result_tok` | 11,636 for grep against 3,950 for smritikosh on the same question. The clearest number in the experiment. |
| `context.peak_tok` | how close the agent came to the dumb zone. |
| `max_result_chars` | 27,157 against 2,618. Proves the mechanism is *bounded vs unbounded tool output*, not "semantic search is smarter". |
| `ranges_returned` vs `citations` | the right range retrieved but not cited is a presentation problem, not a retrieval one. |

`grade.json` keeps a copy of `key_facts` even though the golden file has
them. The duplication is deliberate: a grade that needs two files open to
audit will not get audited, and if the golden set is later edited, the stored
copy still shows what was actually graded.

### What the report shows, and what it hides

The rendered table is eight columns. `runs.csv` carries all 41, including the
cache split, wall time and raw token counts.

That split is a judgement call. Those columns are real, but on the page they
mislead: fresh input tokens are typically 0 % of the bill, so a column of
them reads as "this run was almost free". The CSV is for auditing; the report
is for deciding.

Two derived figures appear only in the report:

- **`$/fact`** — cost divided by covered facts. The cheap arm that answers
  nothing and the correct arm that burns a dollar both look bad, which is
  right.
- **Break-even** — an index arm pays a larger system prompt up front and
  recovers it in smaller tool results, so "36 % cheaper" is true at the
  observed conversation length and not at every length. The section says so
  out loud.

## The page

`visualise.py` renders the whole run tree as one self-contained HTML file.
It computes nothing new — it draws the numbers `extract.py` and `grade.py`
already produced — because the tables are correct and still hard to absorb.
"Tool results were 70 % of context" is a fact you assemble in your head; the
same thing as a stacked bar is a fact you see.

```bash
make eval-view                  # build and open
python visualise.py --out /tmp/eval.html
```

Six sections, in the order you want them:

| section | shows |
| --- | --- |
| **Arms** | headline medians per arm, best value in each row marked green, direction-aware so lower-is-better and higher-is-better both read correctly |
| **Where the context went** | stacked bars on a **shared scale**, so a longer bar is a bigger context. Normalising each to 100 % would hide that one arm's context is twice another's, which is the finding |
| **Where the money went** | the same story in dollars: cost split by input / output / cache read / cache write, per arm, with the dollar figure and token count under each bar |
| **Per question** | coverage grid, coloured, `✓` for a citation hit, question text on hover |
| **Summary and backlog** | the sweep-wide verdict and ranked gaps from `gaps.json`, with severity, evidence and a proposed fix |
| **Runs** | filterable by question and arm; each row opens to show queries issued, **every tool call as a bar sized by what it returned**, **where the money went**, graded facts, citations and the full answer |

Two pictures in there are worth more than the tables they replace.

The **tool timeline** sizes each call by what it returned. A run that greps
five times and then reads one enormous file looks nothing like a run of
evenly sized retrieval calls, and no table conveys that difference.

The **cost breakdown** surprises everyone the first time, and appears twice:
once per arm under "where the money went", once per run inside each row.
Fresh input is a rounding error and cache *writes* are usually the largest
line. The agent is paying to store and re-read its own accumulated reading,
not to think about the question. cacheWrite is ordered first so that is not
buried, and the dollar figures sit in a table under the bars because a hover
tooltip is not something a reader can scan or copy into a note.

The largest line is marked, not left to be worked out — amber text plus a
`largest` tag per run, a highlighted cell per arm, and a sentence underneath
naming it in words. Borrowed from the pi-tropos footer, which highlights
whichever counter holds the largest share of session spend: the biggest
number should not have to be found.

The two bar sections are deliberate twins: context composition in tokens,
then cost composition in dollars. Cache reads move by far the most tokens and
cost the least per token, which is the whole reason a smaller context is
worth more than a cheaper model.

Three constraints it holds to:

- **One file, no network.** No CDN, no framework, no build step. Mail it to
  someone, open it on a plane. A visualiser that breaks when a CDN moves is
  worse than a table.
- **JavaScript only as an enhancement.** Collapsing is native `<details>`.
  The question/arm filter is the one script on the page, and every run is
  already in the DOM — with scripting off the page still shows everything,
  just unfiltered, and Ctrl-F still finds every run.
- **Every pixel traces to a field.** Bar widths are proportions of real
  measurements; nothing is smoothed or rounded for looks. The backlog is
  *read* from `gaps.json`, never recomputed: this page draws, `analyze.py`
  judges, and a second opinion rendered here would be a third to reconcile.

## Running it

```bash
make eval-pilot                          # the lot: prepare, run, grade, report, analyze
make eval-one RUN=q03__smritikosh__r1    # one cell, end to end
make eval-run RUN_ARGS="--questions q03" # one question, every arm
make eval-run RUN_ARGS="--dry-run"       # print the matrix, spend nothing
make eval-report                         # rebuild reports from existing runs
make eval-clean                          # remove worktrees, runs, results
```

Every stage also runs directly, from this directory:

```bash
python run.py --only q08__bare__r1          # one cell
python run.py --questions q03,q05 --arms bare,smritikosh_tuned
python run.py --max-spend 2 --timeout 600
python grade.py --only q08__bare__r1 --redo
python grade.py --passes 2                  # stability check
python analyze.py --only q03
python report.py --focus smritikosh
```

`--only` takes a full run id and is the debugging loop: one question, one
arm, one repeat. It exists as its own flag because spelling the same cell out
as three separate options is how you accidentally run forty.

## Adding things

**A question** — append to `golden/golden.yaml`, verifying every gold range
against the pinned commit. Moving `commit` means re-verifying all of them.

**An arm** — one entry in `arms/arms.yaml`. A new *tool* also needs an arm
script; copy `bare.sh`, which is three lines. A new *docs variant* needs no
script at all.

**A metric** — add it in `extract.py`, then to `CSV_COLUMNS` in `report.py`.
Re-run `extract.py` over the existing tree; no agent runs needed.
