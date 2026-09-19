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
- `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and pull request templates, and a contributor
  guide covering environment setup, the architecture boundary, and what CI enforces.

### Fixed

- Chunk ids now fold in the file path and line span. They were `sha256(text)[:16]`, so
  identical source in two files shared one id, and the second chunk silently replaced the
  first and vanished from search results. Every stored id changes, so the first
  `smritikosh index` after upgrading re-embeds the whole repository; it happens
  automatically and `--full` is not needed.

[Unreleased]: https://github.com/smritikosh/smritikosh/compare/v0.1.0...HEAD
