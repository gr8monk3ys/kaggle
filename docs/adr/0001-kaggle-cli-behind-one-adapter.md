# The Kaggle CLI sits behind one adapter, not fifteen

Fifteen modules each invoked the Kaggle CLI directly, because the shared helper
returned an argv prefix rather than a result — so every caller still owned
`subprocess.run`, its own CSV parse, and its own idea of what failure meant.
We replaced that with a single `KaggleClient` interface whose methods are named
after Kaggle nouns and whose results are typed, with a live adapter and a fake
that ships in the package.

## Considered Options

- **Keep `kaggle_command()` and tidy the callers.** Rejected: an argv prefix is a
  fragment of a seam, not a seam. Tidying fifteen callers leaves fifteen policies.
- **Return `list[dict]` instead of dataclasses.** Rejected: the CSV field names
  *are* the bug surface. A `Warning:` banner parsed as a header row yields dicts
  with garbage keys and no error — eight of ten parse sites had that bug. A typed
  result fails loudly instead.
- **Deprecated shims during a multi-PR migration.** Rejected: that leaves three
  ways to call Kaggle where there were two, and a shim with no deadline is
  permanent.

## Consequences

- Version drift is handled once. The `--page-size` capability is probed and
  cached rather than being handled three incompatible ways; the warning-line
  defence stops being reinvented per call site.
- Two behaviour changes fall out and are intended: `leaderboard_tracker` begins
  paginating entered competitions (it had only ever read page 1), and roughly
  eight call sites that swallowed failures now raise. Callers that genuinely want
  to tolerate a failure must now write the `except` — which is the point.
- The Kaggle *SDK* path (the blob-upload auth probe) sits behind the same
  interface. That moves an import guard which catches `SystemExit` — because the
  SDK can terminate the interpreter on import — off a module's top level, where
  it was a hazard for anything importing it.
- Tests that monkeypatched the old internals were deleted rather than kept. Tests
  that patch past an interface break on every refactor, which is the cost the
  adapter exists to remove.
