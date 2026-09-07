# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A monorepo of Kaggle artifacts (competition entries, educational notebooks, published datasets, discussion drafts) **plus** `kaggle_portfolio/` — a tested Python package that automates the whole Kaggle workflow: validating/pushing notebooks & datasets, scoring quality/usability, tracking medal progress, and running promotion campaigns. Everything is driven through `./manage.sh` from the repo root.

**Goal**: Kaggle Grandmaster across all 4 categories (Competitions, Notebooks, Datasets, Discussion). Live status lives in `docs/reports/grandmaster-tracker.md` — refresh with `./manage.sh sync` rather than hardcoding counts anywhere (they go stale).

## Repository layout

Run `ls` — the tree is self-describing. Two things it does not tell you: `medal_ops/`
is generated output (gitignored except its README), and `pi-automation/` is a separate
Docker + Playwright + cron stack with its own dependencies.

Each `projects/*` and `datasets/*` subfolder holds one `.ipynb` plus a
`kernel-metadata.json` or `dataset-metadata.json`.

## The `kaggle_portfolio` package (engine behind `manage.sh`)

Dispatch chain: `manage.sh` → `kaggle_portfolio/cli.py` → `manage_commands.main()`.

- **Command registry**: `manage_commands.py` holds a `COMMANDS` list. Each entry
  names a `handler` (a function here), a `module` (a dotted path whose
  `main(argv, deps=...)` is called **in-process**), or a `script` (a subprocess —
  reserved for `pi-automation`, whose Playwright dependency must not become
  reachable from a `kaggle_portfolio` import). Modules are imported at dispatch,
  not when the table is built, so `help` does not pay for sklearn.
- **Failure**: commands raise `CommandError`, which the dispatcher turns into an
  exit code. `SystemExit` from inside a command would kill the interpreter they
  now share, so it belongs only in `__main__` guards.
- **Effects**: `--dry-run` is read once, at the dispatcher, and sets
  `deps.effects`. Mutating Kaggle calls and report writes are gated there rather
  than by a conditional each command remembers.
- **Subpackages**: `ops/` `quality/` `datasets/` `notebooks/` `campaigns/` `shared/` —
  `ls kaggle_portfolio/*` for the modules. **Reuse `shared/` rather than
  re-implementing**: `kaggle_client` (the only thing that talks to Kaggle),
  `layout` (the only thing that derives a repo path), `clock`, `reports` (report
  names and emission), `deps`, `errors`, `build_utils`. `discussions/draft_queue`
  is the single Draft Queue model, shared with the `pi-automation` poster.
  `notebooks/competition_lab/` is one module per competition behind an unchanged
  `BENCHMARKS` registry.
- **Medal-ops data flow**: `docs/reports/grandmaster-tracker.md` is the hand-maintained baseline → `ops/medal_ops.py` reads it, syncs live Kaggle CLI counts, and writes reports into `medal_ops/reports/` (gitignored) via `shared/reports.py`. `--dry-run` previews without writing state (convention across `sync`, `campaign-execute`, `post-discussion`).

## Common commands

### Develop / test / lint (run from repo root)

`pytest` and `pre-commit run --all-files` work as usual; `.coveragerc` is the
coverage config. The suite is fully offline — Kaggle is mocked, so no test needs
credentials or a network.

There is **no** `pyproject.toml` / `setup.py` / `requirements.txt` at the root: the package is run via `PYTHONPATH` (set by `manage.sh` and `tests/conftest.py`), not pip-installed. Test fixtures (`repo_root`, `md_cell`, `code_cell`, `write_notebook`, `write_kernel_bundle`, `write_queue_json`) live in `tests/conftest.py`. `pi-automation/` has its own `scripts/requirements.txt` and `tests/`.

### Publishing & ops (`./manage.sh`, run from repo root — `./manage.sh help` lists all ~48 subcommands)

```bash
./manage.sh validate [dir]            # Validate metadata JSON + scan for leaked credentials (no Kaggle CLI needed)
./manage.sh push <dir>                # Push one notebook/dataset dir (auto-validates first)
./manage.sh push-nb | push-ds         # Push all notebooks / all datasets
./manage.sh preflight [--no-pytest]   # Core gate: validate + doctor + quality + usability + draft SLA + pytest
./manage.sh doctor                    # Preflight checks (tracker age, sync inputs, env, credentials)
./manage.sh sync --dry-run            # Preview tracker metric sync from live Kaggle
./manage.sh scorecard | weekly-plan | pace      # Medal-ops reports → medal_ops/reports/
./manage.sh quality --min-score 70 --scope all  # Notebook quality rubric
./manage.sh scout --update            # Regenerate competition-scout-report.md
./manage.sh create-competition-entry <slug> [--gpu]   # Scaffold a new competition dir
```

`requires_kaggle=True` commands need credentials; `validate`/`quality`/`scorecard` run offline.

## Conventions & enforced guardrails

These are checked by `tests/test_repo_guardrails.py` — a violation fails CI:

- **Never commit `kaggle.json`** at the repo root. Store credentials at `~/.kaggle/kaggle.json` (`chmod 600`); copy `kaggle.json.example` as a starting point. Credentials are resolved from env tokens → env vars → `~/.kaggle/kaggle.json` → `./kaggle.json`.
- **No hardcoded `/Users/...` paths** in scripts (cross-platform portability).
- **No `trust_remote_code=True`** anywhere (HuggingFace security gate).
- **No top-level `*.py` scripts at the repo root** — new automation belongs in `kaggle_portfolio/<subpackage>/`, project code under `projects/`/`datasets/`.

Other conventions:
- One `.ipynb` per `projects/*` / `datasets/*` subfolder; each notebook declares its own deps (common: PyTorch/Transformers, scikit-learn, pandas/numpy, XGBoost/LightGBM, plotly/matplotlib/seaborn). GPU notebooks set `enable_gpu: true` in their `kernel-metadata.json`.
- When a competition ships a `build_notebook.py`, the `.ipynb` is generated from it — edit the builder, not the notebook, and keep `model.py`/`baseline.py` logic in sync to avoid drift.
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
