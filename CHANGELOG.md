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

- Automatic Apple GPU acceleration for CodeRankEmbed. Indexing selects the best runtime
  available on the machine, stores its vector-space identity, and search restores the
  compatible runtime without exposing backend choices in the CLI.

- `CODE_QUERY_INSTRUCTION` in `constants`, the required CodeRankEmbed query instruction.
  Both embedder adapters now read it from there rather than keeping their own copy, which
  could drift — a changed instruction silently invalidates every cached embedding.

### Changed

- Apple-silicon installations include the PyTorch runtime through platform-marked
  dependencies. Other platforms skip those packages and keep the portable ONNX runtime.

- Vectors and lexical postings are written in one batch per file through Arrow rather
  than a row at a time. Binding a 768-float vector as a Python list made DuckDB convert
  it element by element at ~40 us each, so storing one vector cost ~30 ms and the write
  dominated indexing: on a 140-file repository, 36.9 s of a 49.1 s run. Per-element cost
  held flat from 96 to 768 dims, so the work is per element rather than per statement and
  batching the SQL alone changed nothing — the fix is to hand DuckDB an Arrow buffer it
  can ingest as-is. The same treatment applies to BM25 postings, which went in through
  `executemany` at ~470 us per row. That repository now indexes in ~3.1 s, and the stored
  bytes are unchanged, so search results do not move.

- `VectorStore` gains `upsert_many`, defaulting to one `upsert` per item so existing
  adapters keep working. `process_chunk` now returns its vector instead of storing it,
  leaving `process_file` to write a whole file's worth at once.

### Fixed

- Changing between equal-width embedding vector spaces now clears stale vectors and
  incremental caches before rebuilding. Search and exploration reject legacy or
  incompatible index metadata instead of silently comparing unrelated vectors.

- `upsert_many` takes the vector width from the stored column rather than from the first
  item in the batch, and rejects a batch that is not uniformly that wide. Binding one
  vector per statement made DuckDB's `FLOAT[dims]` cast the guard; Arrow cuts the batch
  into fixed-width rows first, so `np.fromiter` dropped the overflow past `count` and one
  over-long vector shifted every vector after it into the wrong slots, stored as
  well-formed rows the cast had no reason to reject.

- Arrow handoffs are serialised on a reentrant lock, spanning the lexical store's
  transaction rather than only the inserts inside it. `register`/`unregister` mutate
  catalog state owned by the connection: two threads doing it at once deadlock inside
  DuckDB, and doing it during another thread's open transaction invalidates the pending
  result. Parameter binding tolerated both. This does not make a shared connection
  thread-safe — readers take no lock — so use one connection per thread; separate
  connections to the same file are unaffected.

## [0.2.0] - 2026-09-21

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
- Markdown is split at heading boundaries and sized against the embedding model's own
  tokenizer instead of a character estimate. A nested section used to be stored twice,
  once inside its parent heading's chunk and again as its own, and a chunk that looked
  small enough in characters could still overrun the model's window, which encodes the
  prefix and drops the rest. Each chunk now carries its heading breadcrumb, so a passage
  read from the middle of a document still says which section it came from. Boundaries
  change for every Markdown file, so run `smritikosh index --full` once to rebuild them.

### Fixed

- `smritikosh index --watch` now shows the same progress bar as the first index run.
  Re-index used to print “re-indexing …” and “Done.” with no bar, because the watch
  loop called `build_index` without `on_file_indexed`.
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
- Separate chunks of one Markdown section no longer collapse into a single result. Every
  chunk of a section carries that section's heading as its symbol, and deduplication keyed
  on the symbol, so a long section returned only its first chunk however many matched.

[Unreleased]: https://github.com/smritikosh/smritikosh/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/smritikosh/smritikosh/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smritikosh/smritikosh/releases/tag/v0.1.0
