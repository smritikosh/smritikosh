<!--
The general-purpose CLAUDE.md. Appended to the system prompt for the
claude_md and smritikosh_general arms.

Written from README.md and the directory layout alone, deliberately before
reading the golden set, and reviewed against it only to confirm that no key
fact appears here. A docs arm that quietly contains the answers is not a
comparator, it is a leak: it would beat every retrieval arm and teach us
nothing. If a question is ever added whose answer is stated below, this file
is what must change, not the question.

It is written the way a maintainer would write one: orientation, not
reference. Where things live and what the parts are called, so a reader knows
where to start looking. It does not document behaviour, defaults, or
constants, because a real CLAUDE.md that tried to would be stale within a
month.
-->

# Smritikosh

A local-first indexing and retrieval tool for code, documents, and
conversations. It indexes a repository incrementally and returns precise,
line-cited passages for coding agents. Everything runs on-device: no API
keys, no hosted vector database, no source leaving the machine.

Python 3.10–3.13. Embeddings come from fastembed (ONNX Runtime, with a
PyTorch/MPS path on Apple Silicon). Storage is a single DuckDB file,
`smritikosh.duckdb`, in the repository root. Parsing is tree-sitter.

## Layout

```
smritikosh/
  cli.py              `index` / `search` / `explore` entry points
  exploration_cli.py  the agent-facing `explore` subcommands
  exploration.py      exploration logic behind that CLI
  constants.py        shared constants
  indexing/
    pipeline/         orchestration: file routing and the indexing run
    strategies/       how a file becomes chunks (AST, section, regex)
  retrieval/          search: hybrid channels, fusion, priors, selection
  queries/            tree-sitter tags.scm query files, one dir per language
  engine/             memoization and incremental-rebuild machinery
  models/             dataclasses and option objects
  ports/              interfaces for pluggable sources and backends
  adapters/           implementations of those ports
tests/                mirrors the package layout
```

## How it fits together

Indexing walks the repository, routes each file to a chunking strategy by
extension, embeds the resulting chunks, and writes vectors plus lexical
postings into DuckDB. Re-running it is incremental — unchanged work is not
repeated.

Retrieval splits a question into several focused facets, searches a dense
vector channel and a lexical BM25 channel, combines the rankings, adjusts
them by file metadata, and selects a spread of results to return. Output is
locations — path plus line range — rather than file contents.

## Conventions

- Ports define contracts; adapters implement them. New backends are added as
  adapters, not by editing call sites.
- Private modules are underscore-prefixed (`_pipeline.py`, `_memo.py`) and
  not part of the public surface.
- Tests mirror the package path of the code they cover.
- Ruff for lint and format, line length 88.
