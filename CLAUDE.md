# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A monorepo of Kaggle artifacts (competition entries, educational notebooks, published datasets) **plus** `kaggle_portfolio/` — a tested Python package that automates the whole Kaggle workflow: validating/pushing notebooks & datasets, scoring quality/usability, and tracking medal progress. Everything is driven through `./manage.sh` from the repo root.

**Goal**: Kaggle Grandmaster across all 4 categories (Competitions, Notebooks, Datasets, Discussion). Live status lives in `docs/reports/grandmaster-tracker.md` — refresh with `./manage.sh sync` rather than hardcoding counts anywhere (they go stale).

## Repository layout

Run `ls` — the tree is self-describing. Two things it does not tell you: `medal_ops/`
is generated output (gitignored except its README), and dataset CSVs are build
output: `./manage.sh build-datasets` regenerates them from each `create_dataset.py`.

Each `projects/*` and `datasets/*` subfolder holds one `.ipynb` plus a
`kernel-metadata.json` or `dataset-metadata.json`.

## The `kaggle_portfolio` package (engine behind `manage.sh`)

Dispatch chain: `manage.sh` → `kaggle_portfolio/cli.py` → `manage_commands.main()`.

- **Command registry**: `manage_commands.py` holds a `COMMANDS` list. Each entry
  names a `handler` (a function here) or a `module` (a dotted path whose
  `main(argv, deps=...)` is called **in-process**). Modules are imported at dispatch,
  not when the table is built, so `help` does not pay for sklearn.
- **Failure**: commands raise `CommandError`, which the dispatcher turns into an
  exit code. `SystemExit` from inside a command would kill the interpreter they
  now share, so it belongs only in `__main__` guards.
- **Effects**: `--dry-run` is read once, at the dispatcher, and sets
  `deps.effects`. Mutating Kaggle calls and report writes are gated there rather
  than by a conditional each command remembers.
- **Subpackages**: `ops/` `datasets/` `notebooks/` `shared/` —
  `ls kaggle_portfolio/*` for the modules. **Reuse `shared/` rather than
  re-implementing**: `kaggle_client` (the only thing that talks to Kaggle),
  `layout` (the only thing that derives a repo path), `clock`, `reports` (report
  names and emission), `deps`, `errors`, `build_utils`.
  `notebooks/competition_lab/` is one module per competition behind an unchanged
  `BENCHMARKS` registry.
- **Medal-ops data flow**: `docs/reports/grandmaster-tracker.md` is the hand-maintained baseline → `ops/medal_ops.py` reads it, syncs live Kaggle CLI counts, and writes reports into `medal_ops/reports/` (gitignored) via `shared/reports.py`. `--dry-run` previews without writing state (convention across `sync`, `push`, `publish-datasets`).

## Common commands

### Develop / test / lint (run from repo root)

`pytest` and `pre-commit run --all-files` work as usual; `.coveragerc` is the
coverage config. The suite is fully offline — Kaggle is mocked, so no test needs
credentials or a network.

There is **no** `pyproject.toml` / `setup.py` / `requirements.txt` at the root: the package is run via `PYTHONPATH` (set by `manage.sh` and `tests/conftest.py`), not pip-installed. Test fixtures (`repo_root`, `md_cell`, `code_cell`, `write_notebook`, `write_kernel_bundle`, `write_queue_json`) live in `tests/conftest.py`.

### Publishing & ops (`./manage.sh`, run from repo root — `./manage.sh help` lists all subcommands)

```bash
./manage.sh validate [dir]            # Validate metadata JSON + scan for leaked credentials (no Kaggle CLI needed)
./manage.sh push <dir>                # Push one notebook/dataset dir (auto-validates first)
./manage.sh push-nb | push-ds         # Push all notebooks / all datasets
./manage.sh build-datasets            # Regenerate dataset CSVs (not committed)
./manage.sh preflight [--no-pytest]   # Core gate: validate + doctor + dataset usability + pytest
./manage.sh doctor                    # Preflight checks (tracker age, env, credentials)
./manage.sh sync --dry-run            # Preview tracker metric sync from live Kaggle
./manage.sh sync                      # Live sync: updates the tracker + writes medal_ops/history/ snapshot
./manage.sh digest                    # One-message summary from the snapshot history
./manage.sh scout --update            # Rank boards for medals (Featured/Research) and notebook votes
./manage.sh create-competition-entry <slug> [--gpu]   # Scaffold a new competition dir
```

`requires_kaggle=True` commands need credentials; `validate`/`digest`/`build-datasets` run offline.

## Conventions & enforced guardrails

These are checked by `tests/test_repo_guardrails.py` — a violation fails CI:

- **Never commit `kaggle.json`** at the repo root. Store credentials at `~/.kaggle/kaggle.json` (`chmod 600`); copy `kaggle.json.example` as a starting point. Credentials are resolved from env tokens → env vars → `~/.kaggle/kaggle.json` → `./kaggle.json`.
- **No hardcoded `/Users/...` paths** in scripts (cross-platform portability).
- **No `trust_remote_code=True`** anywhere (HuggingFace security gate).
- **No top-level `*.py` scripts at the repo root** — new automation belongs in `kaggle_portfolio/<subpackage>/`, project code under `projects/`/`datasets/`.

Other conventions:
- One `.ipynb` per `projects/*` / `datasets/*` subfolder; each notebook declares its own deps (common: PyTorch/Transformers, scikit-learn, pandas/numpy, XGBoost/LightGBM, plotly/matplotlib/seaborn). GPU notebooks set `enable_gpu: true` in their `kernel-metadata.json`.
- The `.ipynb` is the source. Edit notebooks directly (run them on Kaggle before pushing a claim about results); there are no generator scripts.
- Don't hardcode medal/vote counts in docs; regenerate via `manage.sh`.

## CI / automation

Workflows live in `.github/workflows/`. The one that matters when a PR goes red:
`ci.yml` runs `preflight --no-pytest` plus the full pytest suite, so a preflight
gate failure and a test failure look the same in the check name.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `gr8monk3ys/kaggle`, driven through the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name
(`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`).
See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root, both created
lazily. See `docs/agents/domain.md`.
