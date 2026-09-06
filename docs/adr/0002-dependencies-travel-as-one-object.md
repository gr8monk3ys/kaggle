# Ambient dependencies travel as one `Deps` object, constructed at the CLI edge

The repo root, the clock, the Kaggle client and report emission were all reached
through import-time module globals — seventeen independent derivations of the
repo root, twenty-plus bare `datetime.now()` calls, and two repo-wide directory
scans that ran on import. We now build one `Deps` object in `cli.py` and pass it
down, so what a command can touch is visible in its signature.

## Considered Options

- **Four explicit parameters** (`client`, `layout`, `clock`, `emitter`) on every
  handler. Rejected: fifty handlers times four parameters, and adding a fifth
  dependency later means touching all fifty again.
- **Default-argument injection** with a module-level factory when the argument is
  `None`. Rejected: the default is a hidden global, which is the thing this
  decision exists to remove, and a test that forgets to pass a fake silently
  reaches production behaviour.

## Consequences

- Each `main()` takes `deps=None` and constructs a default when none is given, so
  `python -m kaggle_portfolio.<module>` keeps working. That is not a concession
  to convenience: the container's cron jobs and health checks invoke modules that
  way, and dropping it would break them.
- `KAGGLE_DIR` and `--today` begin working uniformly across all commands rather
  than the handful that happened to thread them.
- Path defaults that were relative (`Path("medal_ops")`, and the tracker path)
  are anchored to the repo root through the layout. They previously resolved
  correctly only because subprocess delegation forced the working directory; once
  commands run in-process, an unanchored default would silently relocate every
  report. Anchoring them lands *before* in-process dispatch, not with it.
- Effects are a property of `Deps`, so `--dry-run` is enforced at the seam rather
  than by a conditional each command remembers to write. Two known bugs disappear
  by construction: a dry-run sync that still wrote two reports, and a flywheel
  tick that ignored its kill switch.
