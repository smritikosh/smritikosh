<!--
Arm context for graphify. Appended to the system prompt for the graphify and
both arms only.

Kept to what the tool does and how to call it, with no hints about this
repository and no suggestion that graphify is preferable to reading files.
The smritikosh arms get their tool descriptions from the extension's tool
schemas, which are similarly factual, so neither side is being coached.
-->

## graphify

A call-graph index of this repository is available through the `graphify`
command, runnable with bash:

- `graphify query "<text>"` — find symbols and definitions matching a
  description.
- `graphify path <from> <to>` — show call paths between two symbols.
- `graphify explain <symbol>` — summarise a symbol and its neighbours.

Results are locations. Read the files to confirm anything you rely on, and
cite evidence as `PATH:START-END`.
