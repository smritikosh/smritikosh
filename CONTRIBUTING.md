# Contributing

Bugs, adapters, and languages are all welcome. Open an issue before a large change so we
can agree on the shape; small fixes can go straight to a pull request.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Vulnerabilities go
through a [private advisory](SECURITY.md), never a public issue.

## Getting set up

```bash
make reset-env            # builds .venv on the native CPU and installs every group
source .venv/bin/activate
make pre-commit-install
```

`reset-env` always uses `uv`, because choosing the interpreter is the point of it. The
plain `make install` target defaults to Poetry; pass `TOOL=uv` if you want it to match CI.

On macOS, run `make verify-arch` if anything feels slow. `uv` caches both builds of
CPython and will build the venv from whichever it already has, so an Apple Silicon machine
can quietly end up on an x86_64 interpreter under Rosetta, pinned to an older ONNX Runtime.
`CoreMLExecutionProvider` in the provider list means the embedder can reach the GPU.

## The loop

```bash
make test     # fast suites only, no coverage
make cov      # the same suites with the coverage gate
make check    # lint + cov, which is what you want before pushing
make fmt      # rewrites files; CI runs the read-only --check form
```

`make test` and `make cov` run the `FAST_TESTS` list from the Makefile, which skips
everything that loads the real embedding model. Tests marked `slow` do load it, so the
first run downloads weights: `pytest -m slow` when you specifically want them. Add new
fast directories to `FAST_TESTS` or they never run in the loop.

## What CI enforces

Every pull request has to pass seven checks before it can merge, and nobody can bypass
them:

- `lint` — `ruff check`, plus `ruff format --check`
- `test (3.10)` through `test (3.13)` — the full matrix, because `pyproject.toml`
  advertises all four
- `coverage` — total below 90% fails the build; the threshold lives in
  `[tool.coverage.report]`
- `package` — builds the wheel, confirms it carries the tag queries, and runs
  `twine check`

Ruff is pinned to an exact version in the `dev` group and must stay in step with the `rev`
in `.pre-commit-config.yaml`. A formatter that drifts between the hook and CI reformats the
tree on whichever one runs second.

No lock file is committed, so CI resolves dependencies fresh on every run. If a build goes
red on something you did not touch, check whether a dependency released that morning before
assuming it is your change.

## Where code goes

The pipeline talks to contracts, never to what sits behind them.
[`ports`](smritikosh/ports) defines them and [`adapters`](smritikosh/adapters) is what
exists today. A new source, store, or embedder implements the existing port and gets
selected at wiring time; it does not reach into the pipeline, and the pipeline does not
learn its name.

If your change needs the pipeline to know which adapter it is talking to, that is the
signal to discuss it in an issue first.

## Adding a language

1. Add `smritikosh/queries/<lang>/tags.scm` with the captures the chunker expects.
2. Register the language where the router picks a strategy.
3. Add a fixture under `tests/queries/` and list the directory in `FAST_TESTS`.

The `.scm` files are data, not packages, so they reach the wheel only through the
`[tool.setuptools.package-data]` glob in `pyproject.toml`. Nothing in the test suite
notices if that glob stops matching, which is why `make check-wheel` counts the files in a
built wheel and compares against what is on disk.

## Commits

Format: `type | short title`

Types: `feat` `fix` `docs` `refactor` `chore` `test`

- Imperative, lowercase title after the pipe
- One logical change per commit
- No filler body text; add a body only when the title is not enough

```
feat | add DuckDB storage adapter
fix | skip excluded JSON lockfiles in FileRouter
docs | add v1 indexing plan
```

## Attribution

Every commit is authored by the person accountable for the change.

- `Author`, `Committer`, and `Co-authored-by` name people only
- Never credit an assistant, bot, or tool: no `Co-authored-by: Cursor`,
  `Claude`, `Copilot`, or similar, and no "generated with" trailers
- Tools may help you write a change; they do not sign it
- Strip any trailer a tool adds before pushing:

  ```
  git commit --amend        # last commit
  git rebase -i main        # a range, then reword each
  ```

## Pull requests

- Title uses the same `type | title` format as the commit
- Body: 1–3 sentences on what changed and why
- One concern per PR
- Do not paste design essays, test-plan checklists, or generated-by notes
- Rebase onto `main` before merge; do not leave the PR based on a rewritten ancestor
- Add an entry to `CHANGELOG.md` under `Unreleased` when the change is user-visible

`main` takes squash merges only, so the pull request title becomes the commit subject and
your individual commit messages become its body. One approval is required and reviews are
dismissed when you push again.
