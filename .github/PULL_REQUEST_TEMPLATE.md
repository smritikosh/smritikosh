<!--
Title format: `type | short title`
Types: feat  fix  docs  refactor  chore  test

One concern per pull request. See CONTRIBUTING.md.
-->

## What and why

<!-- One to three sentences. No design essays, no test-plan checklists. -->

## Checks

- [ ] `make check` passes locally (lint and coverage, the gates CI runs)
- [ ] One concern, and the title follows `type | short title`
- [ ] Rebased onto `main`
- [ ] No tool or assistant attribution in any commit trailer
- [ ] `CHANGELOG.md` updated under `Unreleased`, or this change is not user-visible

<!--
Adding a language? `smritikosh/queries/<lang>/tags.scm` is the parser side; the
wheel check in CI counts those files, so confirm the packaged wheel picked it up.
-->
