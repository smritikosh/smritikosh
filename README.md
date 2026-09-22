<div align="center">

<img src="https://raw.githubusercontent.com/smritikosh/smritikosh/main/assets/hero-light.svg" alt="Git repositories, whether one monorepo or many, plus documents and Slack threads, stay live as they change, and the list keeps growing, with more sources you can ask for on the way. Smritikosh keeps them in sync incrementally, re-reading only the delta and leaving everything else untouched, then hands a coding agent a handful of exact source ranges that never point at a stale line number, whether they come from either repository, from a document, or from a Slack thread." width="100%" draggable="false">

# Your agents deserve _exact evidence._

**Star us →** [Smritikosh on GitHub](https://github.com/smritikosh/smritikosh) ·
[PyPI](https://pypi.org/project/smritikosh/) ·
[Benchmarks](#benchmarks) ·
[Issues](https://github.com/smritikosh/smritikosh/issues) ·
[Contributing](https://github.com/smritikosh/smritikosh/blob/main/CONTRIBUTING.md)

Smritikosh is a local-first indexing and retrieval tool for code, documents, and conversations.
It incrementally indexes changes and returns precise, line-cited passages for agents, without
API keys, hosted vector databases, or sending source data off-device.

**Precise** · line-level citations &nbsp;·&nbsp; **Incremental** · indexes only changes &nbsp;·&nbsp; **Local** · no API keys or cloud

[![CI](https://github.com/smritikosh/smritikosh/actions/workflows/ci.yml/badge.svg)](https://github.com/smritikosh/smritikosh/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/smritikosh?color=4C8DFF)](https://pypi.org/project/smritikosh/)
[![Downloads](https://img.shields.io/pypi/dm/smritikosh?color=34D399)](https://pypi.org/project/smritikosh/)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://www.python.org/)
[![Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/smritikosh/smritikosh/blob/main/LICENSE)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)

</div>

## Get started

No Python on the machine? `uv` brings its own.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows: docs.astral.sh/uv
uv tool install smritikosh
```

Or `pipx install smritikosh`. Where the command is missing from PATH, `python -m smritikosh`
is the same thing.

**Index whatever the agent should know.** It all lands in one `./smritikosh.duckdb` and is
searched as a single corpus.

```bash
smritikosh index ./auth-svc        # one repository, or fifty (repeat per repo)
smritikosh index ./docs            # Markdown and MDX beside the code
smritikosh index ./slack-export    # Slack threads and .docx (soon)
```

**Then let the agent explore.** `explore tools` hands it the workflow and the citation
contract, `search` returns locations, `chunks` reads only those ranges.

```bash
smritikosh explore tools
smritikosh explore search "authorization decision flow" "authorization tests"
smritikosh explore chunks --range src/auth/policy.py 176 193
```

Re-run `index` anytime: unchanged files are skipped, so only the Δ costs anything.
Add `--watch` to keep it live while you work.

## Retrieval: _evidence you can point at_

<img src="https://raw.githubusercontent.com/smritikosh/smritikosh/main/assets/hybrid-retrieval-light.svg" alt="An agent asks one plain question, why can't users log in, and Smritikosh turns it into four angles: where login is decided, what it checks, where it says no, and the tests that cover it. Each angle runs down three retrieval channels: semantic recall over dense vectors, lexical precision over BM25, and a call-graph channel that is not yet implemented. The rankings are combined by Reciprocal Rank Fusion, Facet Reservation, and Maximal Marginal Relevance, and what survives comes back as a short list of exact line ranges such as auth/policy.py 176-193: whole definitions plus one hop further, with line numbers that stay true." width="100%" draggable="false">

Ask _why can't users log in?_ and Smritikosh looks at it from four sides: where the decision
is made, what it checks, where it says no, and the tests that cover it.

It matches the idea _and_ the exact names, keeps an answer from every side, drops
near-copies, and widens each one to the whole thought plus one place that refers to it,
so what comes back is the exact lines, not everything around them.

## Why _incremental?_

Your agent is only as good as the lines it can trust. Code moves all day, and an index that
doesn't move with it quietly points at the wrong ones. Smritikosh keeps up, and only ever
re-reads what changed.

<img src="https://raw.githubusercontent.com/smritikosh/smritikosh/main/assets/incremental-light.svg" alt="Why an index has to keep up. Before: in the file policy.py, the login check sits at lines 176 to 193. Then someone adds two new lines near the top of that file, and everything below them moves down, so the login check now sits at lines 178 to 195. An index that was not updated still answers lines 176 to 193: that answer is instant, it is two lines too high, and it quotes the wrong code. Smritikosh notices the file changed, reads that one file again in 3 seconds, and answers lines 178 to 195. Leave smritikosh index with the watch flag running and it keeps up on every save; re-reading all 59 files in the project instead would take 40 seconds." width="100%" draggable="false">

## Benchmarks

<img src="https://raw.githubusercontent.com/smritikosh/smritikosh/main/assets/benchmarks-light.svg" alt="Recorded session cost by benchmark task: multi-repo analysis 0.381 dollars with Smritikosh versus 1.634 reading the repository directly, single-repo analysis 0.230 versus 0.330, architecture discovery 0.361 versus 1.259. Lower is better." width="100%" draggable="false">

Paired agent sessions, same question and same effective model per pair:

| Task | Smritikosh | Baseline | Effect |
| --- | --- | --- | --- |
| Multi-repo analysis | 105.3 s · $0.381 | 193.4 s · $1.634 | 45.6% faster · 76.7% cheaper · 90.0% fewer tokens |
| Single-repo analysis | 68.4 s · $0.230 | 55.8 s · $0.330 | 30.4% cheaper · 32.5% fewer tokens · 40% fewer tool calls |
| Curated architecture discovery | 56.6 s · $0.361 | 113.9 s · $1.259 | 50.3% faster · 71.3% cheaper · 80.4% fewer tokens |

These are individual exported sessions, not statistically controlled measurements, and
cumulative token counts include repeated cache reads. The table keeps the cases that went the
other way (the single-repo run was 12.6 seconds slower, and the multi-repo baseline covered more
repositories), so the efficiency numbers can be read honestly.

## How it extends: _ports and adapters_

The pipeline talks to contracts, never to what sits behind them:
[`ports`](https://github.com/smritikosh/smritikosh/tree/main/smritikosh/ports) defines them,
[`adapters`](https://github.com/smritikosh/smritikosh/tree/main/smritikosh/adapters) is what exists
today.

| Plug point | Shipping today | Same contract, not yet written |
| --- | --- | --- |
| **Source** | Local filesystem, `.gitignore`-aware | Slack, Google Drive, S3, Confluence, meeting notes |
| **Structure** | Python, TypeScript, JavaScript, Java, Kotlin, Markdown, MDX, JSON, TOML | Go, Rust, C#; transcripts by speaker turn, tickets by field |
| **Store** | DuckDB, one local file | Postgres with pgvector, Qdrant, Neo4j |
| **Embedder** | FastEmbed on local ONNX, no API key; Apple GPU via `smritikosh[mps]` | OpenAI, Voyage, or any hosted model |
| **Retrieval** | Dense vectors and BM25 | Call graph: callers and callees |

## How it works

Indexing walks the tree once and writes everything into one DuckDB file. Search reads that
same file back as a handful of line ranges.

<img src="https://raw.githubusercontent.com/smritikosh/smritikosh/main/assets/how-it-works-light.svg" alt="How Smritikosh works, end to end, for documents, code and config alike. Step 1, walk the tree: it discovers the files in a project, here a handbook, a source file and a settings file, and skips what you ignore, shown struck through. Step 2, keep whole thoughts: it parses structure rather than plain text, extracts the definitions and sections it finds, and chunks them so a range is always one whole idea. Step 3, index three ways, each drawn as a small picture: a cloud of scattered dots for semantic recall, two literal tokens for lexical precision, and a five-node graph of connected callers and callees for the call graph, which is still to come. Step 4, re-read only the delta: the changed file is read again while the unchanged ones are skipped. Step 5, persist: chunks and metadata, dense vectors, BM25 postings, and the hash and memo state all live in one local database file, smritikosh.duckdb. Search reads that same file back: a question is fused across the rankings, coverage is kept and near-duplicates dropped, survivors expand to whole passages, and the answer comes back as an exact path with a start and end line, never the file body." width="100%" draggable="false">

## What can you build?

The recipe never changes: `index` whatever the answer could live in, then let the agent
`search` and read only the ranges it picks. Only what you point it at is different.

<img src="https://raw.githubusercontent.com/smritikosh/smritikosh/main/assets/what-you-build-light.svg" alt="What you can build on Smritikosh, as six cards over two rows. The top row ships today over code, Markdown and config, and every card ends in a citation. Five git repositories: gateway, service and worker in one index, so one question returns lines from whichever repository owns the answer, citing plt-notification-svc/handler.py lines 88 to 102. Code plus Markdown: onboarding from docs and code together, where the handbook explains and the source proves, citing docs/billing-handbook.md lines 42 to 58. One repository: business rules you can check, the formula, the branch that changes it and the test that pins the number, citing src/billing/charges.py lines 176 to 193. The bottom row is the same ports with adapters not yet written, each marked soon: Slack threads for why a decision was made, PDF and DOCX for the clause rather than the whole file, meeting notes for who committed to what by speaker turn, and CSV tables for the row that explains a number. Everything lands in one local smritikosh.duckdb and is searched as a single corpus." width="100%" draggable="false">

Every card in the top row is measured under [Benchmarks](#benchmarks): the same question asked
with the index and without it, time and cost recorded for both runs.

The bottom row is the honest half. Those sources have a
[port](https://github.com/smritikosh/smritikosh/tree/main/smritikosh/ports) and no adapter
behind it yet, so the citations on those cards are the shape they will take, not something you
can run this week. [Open an issue](https://github.com/smritikosh/smritikosh/issues) for the
one you need first, or build it against the same contract.

Built something on top of Smritikosh?
[Tell us](https://github.com/smritikosh/smritikosh/issues). We want to see it.

## Prior art: _the papers behind these choices_

Three papers shaped decisions you can point at in the code. Citing one credits an idea we
used; it does not mean its authors reviewed or endorsed Smritikosh, and where our
implementation diverges from theirs, that is said out loud.

**Chunk on structure, not on line counts.** Zhang, Zhao, Wang, Yang, Wei and Wu,
[_cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract
Syntax Tree_](https://arxiv.org/abs/2506.15655) (arXiv:2506.15655, 2025), argues that
line-based chunking breaks functions apart and merges unrelated code, and that recursively
splitting oversized AST nodes while merging siblings under a size limit yields self-contained
units. That is the shape of
[`strategies/ast.py`](https://github.com/smritikosh/smritikosh/blob/main/smritikosh/indexing/strategies/ast.py) —
siblings grouped, a class carried together with its initializer — and of `split_oversized` in
[`strategies/_helpers.py`](https://github.com/smritikosh/smritikosh/blob/main/smritikosh/indexing/strategies/_helpers.py).
We diverge on the oversized case: cAST recurses into the node, we cut it into overlapping line
windows.

**Fuse independent channels by rank.** Choudhary, Kandoi and Patel,
[_Hybrid GraphRAG for Cross-Lingual Legal Citation Retrieval: A Multi-Signal Fusion Approach
for Swiss Legal Information Systems_](https://ijecs.in/index.php/ijecs/article/view/5461)
(IJECS, 2026), runs lexical, dense, and graph retrieval as separate families and combines them
with weighted Reciprocal Rank Fusion, reporting `k = 60` as where top-rank emphasis balances
against rank dilution.
[`retrieval/hybrid.py`](https://github.com/smritikosh/smritikosh/blob/main/smritikosh/retrieval/hybrid.py)
keeps that channel separation,
[`retrieval/fusion.py`](https://github.com/smritikosh/smritikosh/blob/main/smritikosh/retrieval/fusion.py)
defaults to that `k`, and the call-graph channel listed above as not yet written is their third
family. We weigh sources after fusion in
[`retrieval/priors.py`](https://github.com/smritikosh/smritikosh/blob/main/smritikosh/retrieval/priors.py)
rather than weighting each signal inside RRF, and we run no cross-encoder or LLM verification
stage.

**Measure an index by toggling only the index.** Bhola, Krishnan, Kurmala and NS,
[_Code Isn't Memory: A Structural Codebase Index Inside a Coding
Agent_](https://arxiv.org/abs/2606.22417) (arXiv:2606.22417, 2026), holds the agent harness and
the model fixed, switches only the index on and off, and reports cost per solved task against
an agentic-grep comparator. [Benchmarks](#benchmarks) borrows that framing. It does not borrow
the rigour: their protocol adds multiple seeds, a leak audit, and statistical separation, which
is why our table is offered as recorded sessions rather than a controlled result.

Reciprocal Rank Fusion, Okapi BM25, and Maximal Marginal Relevance all predate these three and
belong to their own authors; the papers above are where we met them in this shape.

**Built by** [Shantanu Vashishtha](https://github.com/learncoder4848) and
[Sarvesh Sawant](https://github.com/devsarvesh92).

<div align="center">

Apache 2.0 · © Smritikosh contributors

</div>
