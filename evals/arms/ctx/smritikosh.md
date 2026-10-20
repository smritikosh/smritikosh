<!--
Arm context for smritikosh. Appended to the system prompt for the
smritikosh, smritikosh_general, smritikosh_tuned and both arms.

Deliberately the same shape and length as ctx/graphify.md: what the commands
are, what they return, and nothing about this repository. Any hint here
about where the answers live would be coaching, and a coached arm proves
nothing about retrieval.

Note what this file does *not* contain: the facet-count guidance, the
search -> read -> cite discipline, the "verify before asserting" rule. Those
live in docs/smritikosh.md and belong to the smritikosh_tuned arm alone.
Keeping them out of here is what makes `smritikosh_tuned` minus
`smritikosh_general` a measurement of the workflow prompt rather than of two
differently worded tool descriptions.
-->

## smritikosh

A semantic index of this repository is available through the `smritikosh`
command, runnable with bash:

- `smritikosh explore search QUERY [QUERY ...]` — search one question's
  facets; pass several focused queries in one call. Returns locations only.
- `smritikosh explore chunks --range PATH START END` — print those source
  lines, numbered. Repeat `--range` to read several spans in one process.
- `smritikosh explore chunks PATH` — list the indexed definitions in a file.
- `smritikosh explore tools` — the exact syntax and options.

Search returns locations, not file contents. Read what matters, and cite
evidence as `PATH:START-END`.
