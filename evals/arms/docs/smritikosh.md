<!--
The smritikosh-tuned CLAUDE.md. Appended to the system prompt for the
smritikosh_tuned arm, in addition to general.md.

This is the delta against general.md and nothing else: the same project
overview plus the retrieval workflow. Keeping it additive is what makes
smritikosh_tuned - smritikosh_general a clean measurement of "what does
teaching the workflow add", rather than a measurement of two differently
written documents.

Same leak rule as general.md: workflow guidance only, never an answer. Note
that it says nothing about this repository's internals — if it did, the arm
would win by being briefed, not by retrieving.
-->

## Using the code index

A semantic index of this repository is available through the
`smritikosh_search` and `smritikosh_chunks` tools. Prefer them over grep for
any "where is X" or "how does Y work" question; grep finds spellings, the
index finds meaning.

**Search, then read, then cite.**

1. `smritikosh_search` with 4–8 *focused facets* of the one question. A facet
   is a distinct angle, not a synonym — flow, state, configuration, failure
   path, tests. Repeating the same idea in different words wastes a slot and
   returns the same chunk.
2. `smritikosh_chunks` with `--range PATH START END` to read the ranges that
   look relevant. Batch independent reads into one call. Search returns
   locations only; it is not evidence until you have read it.
3. Cite every material claim as `PATH:START-END`, using ranges you actually
   read back.

Treat search results as candidates, not findings. Verify anything you assert
by reading the source, including the tests that pin the behaviour. Keep
findings, assumptions, and unknowns separate — an honest "not present in this
codebase" is a correct answer, and a plausible invention is not.
