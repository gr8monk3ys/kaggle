# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A monorepo of Kaggle artifacts (competition entries, educational notebooks, published datasets, discussion drafts) **plus** `kaggle_portfolio/` — a tested Python package that automates the whole Kaggle workflow: validating/pushing notebooks & datasets, scoring quality/usability, tracking medal progress, and running promotion campaigns. Everything is driven through `./manage.sh` from the repo root.

**Goal**: Kaggle Grandmaster across all 4 categories (Competitions, Notebooks, Datasets, Discussion). Live status lives in `docs/reports/grandmaster-tracker.md` — refresh with `./manage.sh sync` rather than hardcoding counts anywhere (they go stale).

## Repository layout

```
manage.sh              # Entry point → python3 -m kaggle_portfolio.cli (run ./manage.sh help)
kaggle_portfolio/      # The automation engine (see architecture below)
tests/                 # pytest suite for kaggle_portfolio (28 files, fully offline/mocked)
medal_ops/             # Generated scorecards/plans/reports (gitignored except README)
pi-automation/         # Dockerized Playwright/cron automation (own scripts/ + tests/ + requirements.txt)
docs/reports/          # grandmaster-tracker.md (source of truth), competition-scout-report.md
docs/discussions/      # engagement-strategy.md (12-week plan), discussion-drafts.md
projects/competitions/ # Per-competition dirs; each has a build_notebook.py and/or baseline.py
projects/educational/  # Per-topic teaching notebooks (feature-engineering, attention-guide, …)
datasets/              # Custom Kaggle datasets (CSV/Parquet + dataset-metadata.json)
```

Each `projects/*` and `datasets/*` subfolder is self-describing by name; the primary artifact is a single `.ipynb` (Kaggle kernel) plus a `kernel-metadata.json` or `dataset-metadata.json`. Competition starters may ship a `baseline.py` + `build_notebook.py` that emits the notebook.

## The `kaggle_portfolio` package (engine behind `manage.sh`)

Dispatch chain: `manage.sh` → `kaggle_portfolio/cli.py` → `manage_commands.main()`.

- **Command registry**: `kaggle_portfolio/manage_commands.py` defines a `COMMANDS` list of `Command(name, description, handler, args, requires_kaggle)` dataclasses, indexed into `COMMAND_INDEX`. `main()` looks up argv[0] and calls the handler. Two handler styles:
  - **Local handlers** (`cmd_push`, `cmd_status`, `cmd_validate`, `cmd_votes`, …) — push/validate/status logic that lives directly in `manage_commands.py`.
  - **Module delegation** — `lambda a: run_module("kaggle_portfolio.<sub>.<mod>", [...a])`, which runs the submodule as `python -m` with subcommand-style args. Every delegated module has its own `main()` / argparse.
- **Subpackages**:

  | Package | Role | Key modules |
  |---|---|---|
  | `ops/` | Medal tracking, repo health, scheduling, telemetry | `medal_ops.py` (scorecard/weekly-plan/pace/sync/doctor), `repo_ops.py` (preflight/smoke-live), `kaggle_auth_doctor.py`, `discussion_scheduler.py`, `leaderboard_tracker.py`, `stale_content_detector.py`, `metadata_tracker.py` |
  | `quality/` | Notebook quality rubric scoring + fix selection | `notebook_quality.py` |
  | `datasets/` | Dataset lifecycle: score → publish gate → live usability | `dataset_usability.py`, `dataset_publish_pipeline.py`, `dataset_optimizer.py`, `dataset_explore_generator.py` |
  | `notebooks/` | Build automation, scouting, entry scaffolding | `notebook_pipeline.py`, `competition_scout.py`, `competition_entry.py`, `local_competition_lab.py`, `notebook_promoter.py` |
  | `campaigns/` | Multi-channel promotion queue | `campaign_pack.py`, `campaign_dispatcher.py`, `campaign_execute.py` |
  | `shared/` | **Reuse these — do not re-implement** | `kaggle_utils.py`, `build_utils.py` |

- **Shared helpers to reuse** (`kaggle_portfolio/shared/`):
  - `kaggle_utils.py`: `kaggle_command()` / `run_kaggle()` (invoke the Kaggle CLI with error summarization), `has_kaggle_cli()`, `parse_iso_date()` / `resolve_today()` (date handling with override), `@retry(...)`, `configure_logging()`.
  - `build_utils.py`: `md()` / `code()` (Jupyter cell factories) + `write_notebook()` — use these when generating `.ipynb` files instead of hand-building cell dicts.
  - In `manage_commands.py`: `discover_notebook_dirs()` / `discover_dataset_dirs()` (rglob for `*-metadata.json`, skipping `.git`/`.venv`), `has_kaggle_credentials()`, `validate_kernel()` / `validate_dataset()`.
- **Medal-ops data flow**: `docs/reports/grandmaster-tracker.md` is the hand-maintained baseline → `ops/medal_ops.py` reads it, syncs live Kaggle CLI counts, and writes reports into `medal_ops/reports/` (gitignored). `--dry-run` previews without writing state (convention across `sync`, `campaign-execute`, `post-discussion`).

## Common commands

### Develop / test / lint (run from repo root)

```bash
pytest -q                                   # Full suite (offline; mocked via monkeypatch — no real Kaggle calls)
pytest tests/test_medal_ops.py -v           # Single file
pytest tests/test_medal_ops.py::test_name   # Single test
pytest -q --cov=. --cov-config=.coveragerc --cov-report=term-missing  # With coverage
pre-commit run --all-files                  # Lint + format (ruff + ruff-format) and hygiene hooks
```

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

GitHub Actions in `.github/workflows/`: `ci.yml` (PR/push: `preflight --no-pytest` + full pytest coverage), `medal-ops-health.yml` (daily 09:10 UTC health checks, opens/updates an issue on failure), `live-smoke.yml` (manual, non-mutating live checks), and security scanning (`codeql.yml`, `semgrep.yml`, `security-baseline.yml`). `pi-automation/` is a separate Docker + Playwright + cron stack for scheduled UI engagement (see its `DEPLOY.md`).
