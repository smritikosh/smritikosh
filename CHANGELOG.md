# Changelog

Notable changes to Smritikosh. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is alpha (`0.x`), a minor bump may carry a breaking change; the entry
will say so.

Land your entry under `Unreleased` in the same pull request as the change. At release the
heading is renamed to the version and a fresh `Unreleased` opens above it.

## [Unreleased]

### Added

- `python -m smritikosh` as a second way to reach the CLI, for installs where the console
  script's directory is not on `PATH` — `pip install --user` on macOS does not put it
  there, and PEP 668 pushes people toward exactly that flag.
- A `SourceReader` port, with the read-only explorer behind it as `DuckDBSourceReader`, so
  code that only reads an index depends on the contract instead of on DuckDB.
  `smritikosh.exploration` still exports `ReadOnlyExplorer`, now an alias of the adapter.
- `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and pull request templates, and a contributor
  guide covering environment setup, the architecture boundary, and what CI enforces.
- `explore chunks --range PATH START END`, repeatable, so several ranges are read in one
  process instead of one process per range.

### Changed

- **Breaking for adapter authors.** `StorageAdapter` gains `clear_nodes` and
  `clear_caches`, and `VectorStore` gains `clear`. A third-party adapter that does not
  implement them will no longer instantiate. `VectorStore.delete_many` is new as well but
  defaults to looping over `delete`, so it needs nothing from existing implementations.
- **Breaking.** `smritikosh explore` is now three subcommands — `search`, `chunks`, and
  `tools` — and `info`, `paths`, and `text` are gone. One subcommand per reader method
  pushed the choice of how to explore onto the agent and cost a round trip per decision;
  `tools` states the workflow and its limits so an agent can start without a prompt that
  describes them.
- **Breaking.** `explore search` fuses dense and BM25 candidates for a question's facets
  and returns locations without any source, leaving the agent to read only the ranges it
  judges worth reading through `explore chunks --range`. `--top-k`, `--include-path`,
  `--exclude-path`, and `--full` are gone from `search`; `--max-results` bounds the page
  instead, at up to 24.

### Fixed

- Chunk ids now fold in the file path and line span. They were `sha256(text)[:16]`, so
  identical source in two files shared one id, and the second chunk silently replaced the
  first and vanished from search results. Every stored id changes, so the first
  `smritikosh index` after upgrading re-embeds the whole repository; it happens
  automatically and `--full` is not needed.
- `smritikosh index --full` now empties the indexes before rebuilding. It cleared the memo
  cache and the file hashes but left the previous run's nodes and vectors in place, so a
  rebuild layered new rows over stale ones instead of starting clean.
- Deleting a source file now removes its vectors too. The nodes and the file hash were
  removed, but the rows keyed by the chunk ids that had just been dropped were left
  behind. An index built before this fix may still be carrying them; one
  `smritikosh index --full` clears them out.

[Unreleased]: https://github.com/smritikosh/smritikosh/compare/v0.1.0...HEAD
