# Repo Freshness + Health Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the repo current from its March 2026 snapshot — refresh tracker/scout data via the repo's own tooling, fix the small bugs blocking that tooling, update CLAUDE.md to match reality, and verify with the test suite plus a code review.

**Architecture:** Tooling-first refresh on branch `chore/freshness-health-pass`. A project `.venv` provides the `kaggle` CLI (PEP 668 blocks system pip). All medal_ops/scout/quality commands run from the repo root as `.venv/bin/python -m kaggle_portfolio.<module>`, with flags placed AFTER the subcommand (argparse subparser defaults silently clobber flags placed before it). Generated reports under `medal_ops/` are gitignored; the committed diff is: two code fixes, the tracker, the scout report, CLAUDE.md, the health workflow flag fix, and the spec/plan docs.

**Tech Stack:** Python 3.14 venv, pytest 9, kaggle CLI, existing `kaggle_portfolio` package.

**Spec:** `docs/superpowers/specs/2026-06-10-freshness-health-pass-design.md`

**Pre-verified facts (from tooling exploration, 2026-06-11):**
- `pip install kaggle` into system Python is blocked (PEP 668, Homebrew 3.14). `.venv/` is already gitignored.
- `pytest --collect-only`: 742 tests + 2 collection errors — `tests/test_kaggle_auth_doctor.py` (needs `kaggle` module), `tests/test_local_competition_lab.py` (needs `pandas`). Both resolve once the venv installs those packages.
- Confirmed bug: `kaggle_portfolio/shared/kaggle_utils.py` lines 104 and 112 — `importlib.util.find_spec("kaggle.cli")` raises `ModuleNotFoundError` (instead of returning None) when the `kaggle` package is absent, making the `["kaggle"]` fallback unreachable.
- `manage.sh` is committed mode 100644 → documented `./manage.sh` fails with permission denied. Line 5 hardcodes `MODULE_ROOT="/workspaces/kaggle"` (stale devcontainer path).
- `medal_ops sync` (live mode) surgically rewrites `docs/reports/grandmaster-tracker.md` in place: only the `**Last Updated:**` line and `Current` cells of the Notebooks/Datasets/Competitions tables. It does NOT touch the "Active competitions to enter" table — that needs a hand-edit from fresh scout data.
- `competition_scout --update` rewrites `docs/reports/competition-scout-report.md`; needs live Kaggle API; `--update` is its only flag.
- `weekly-plan` prefers the latest `medal_ops/history/` snapshot over the tracker — run `scorecard`/`pace` first.
- `.github/workflows/medal-ops-health.yml` passes `--output-root` BEFORE the medal_ops subcommand, so argparse resets it to the default — reports go to the repo's `medal_ops/` in CI instead of `/tmp/medal_ops_health`.
- Recent Medal Ops Health cron failures are GitHub **billing** failures (job never starts). Only the account owner can fix that; out of repo scope.
- CI guardrail: any literal `/Users/` in committed `*.py`/`*.sh` outside `tests/` fails CI. Keep absolute paths out of committed files.

---

### Task 1: Create project venv with all dependencies

**Files:** none committed (`.venv/` is gitignored)

- [ ] **Step 1: Create venv and install packages**

```bash
cd /Users/natalyscaturchio/code/kaggle
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install kaggle pandas pytest pytest-mock pytest-cov requests
```

Expected: clean install, no errors.

- [ ] **Step 2: Verify the CLI and modules resolve**

```bash
.venv/bin/kaggle --version
.venv/bin/python -c "import pandas, kaggle; print('imports OK')"
```

Expected: a kaggle version string (creds at `~/.kaggle/kaggle.json` are picked up automatically), then `imports OK`.

### Task 2: Test baseline

- [ ] **Step 1: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ~744+ tests collected (the 2 previously-erroring files now collect), **0 failures**. CI was green on the last push (2026-03-13), so failures here are environment-related (Python 3.14 vs CI's 3.11) or real regressions — either way, STOP and diagnose with superpowers:systematic-debugging before proceeding. Record the exact pass count for the final report.

### Task 3: Fix `kaggle_utils` crash when `kaggle` package is absent (TDD)

**Files:**
- Modify: `kaggle_portfolio/shared/kaggle_utils.py:102-114`
- Test: `tests/test_kaggle_utils.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_kaggle_utils.py`; match existing monkeypatch style at lines 8-32)

```python
def _raise_module_not_found(name):
    raise ModuleNotFoundError("No module named 'kaggle'")


def test_kaggle_command_survives_missing_kaggle_package(monkeypatch):
    monkeypatch.setattr(kaggle_utils, "kaggle_cli_path", lambda: None)
    monkeypatch.setattr(kaggle_utils.importlib.util, "find_spec", _raise_module_not_found)

    assert kaggle_utils.kaggle_command() == ["kaggle"]


def test_has_kaggle_cli_survives_missing_kaggle_package(monkeypatch):
    monkeypatch.setattr(kaggle_utils, "kaggle_cli_path", lambda: None)
    monkeypatch.setattr(kaggle_utils.importlib.util, "find_spec", _raise_module_not_found)

    assert kaggle_utils.has_kaggle_cli() is False
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_kaggle_utils.py -q
```

Expected: 2 FAILED with `ModuleNotFoundError: No module named 'kaggle'` raised from `has_kaggle_cli`/`kaggle_command`.

- [ ] **Step 3: Implement the fix** — in `kaggle_portfolio/shared/kaggle_utils.py`, replace lines 102-114 with:

```python
def _kaggle_cli_module_available() -> bool:
    """Return whether the ``kaggle.cli`` module can be imported."""
    try:
        return importlib.util.find_spec("kaggle.cli") is not None
    except ModuleNotFoundError:
        return False


def has_kaggle_cli() -> bool:
    """Return whether a usable Kaggle CLI is available."""
    return kaggle_cli_path() is not None or _kaggle_cli_module_available()


def kaggle_command() -> list[str]:
    """Return a runnable Kaggle CLI command prefix."""
    binary = kaggle_cli_path()
    if binary:
        return [binary]
    if _kaggle_cli_module_available():
        return [sys.executable, "-m", "kaggle.cli"]
    return ["kaggle"]
```

- [ ] **Step 4: Run the test file, then the full suite**

```bash
.venv/bin/python -m pytest tests/test_kaggle_utils.py -q && .venv/bin/python -m pytest -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/shared/kaggle_utils.py tests/test_kaggle_utils.py
git commit -m "Fix kaggle CLI detection crash when kaggle package is absent"
```

### Task 4: Fix `manage.sh` exec bit and stale devcontainer path

**Files:**
- Modify: `manage.sh:5` (and file mode)

- [ ] **Step 1: Edit line 5** — replace `MODULE_ROOT="/workspaces/kaggle"` with:

```bash
MODULE_ROOT="${KAGGLE_DIR}"
```

(`KAGGLE_DIR` resolves to the script's directory on line 4, so devcontainer behavior is unchanged and the wrapper now also works outside the repo root.)

- [ ] **Step 2: Restore the executable bit**

```bash
chmod +x manage.sh
```

- [ ] **Step 3: Verify** — `./manage.sh help` from repo root must print the 44-command usage; from another cwd it must now import cleanly:

```bash
./manage.sh help | head -3
(cd /tmp && "$OLDPWD/manage.sh" help | head -3)
```

Expected: both print `Usage: ./manage.sh <command> [options]`.

- [ ] **Step 4: Run guardrail tests, then commit** (mode change must show as 100644 → 100755)

```bash
.venv/bin/python -m pytest tests/test_repo_guardrails.py tests/test_manage_commands.py -q
git add manage.sh
git status --short && git diff --cached --summary
git commit -m "Make manage.sh executable and drop hardcoded devcontainer path"
```

Expected: tests pass; `git diff --cached --summary` shows `mode change 100644 => 100755 manage.sh`.

### Task 5: Doctor preflight

- [ ] **Step 1: Run doctor** (flags AFTER subcommand)

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops doctor
```

Expected: exit 0 with WARNINGS — tracker stale (>7 days; it's 3 months old; the live sync in Task 6 fixes this), everything else READY (CLI now resolves via `.venv/bin/kaggle` sibling of `sys.executable` when run via `.venv/bin/python`; creds present). Read `medal_ops/reports/latest-doctor.md`; if it reports ERRORS (BLOCKED), fix those before continuing.

### Task 6: Live sync

- [ ] **Step 1: Dry-run preview**

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops sync --dry-run
```

Expected: exit 0; read `medal_ops/reports/latest-sync.md` — it lists proposed tracker changes (vote totals, medal counts, entered competitions) from live data. If it exits non-zero with `Command failed:` the creds are invalid → STOP, report to user (never fabricate numbers; spec's error-handling rule).

- [ ] **Step 2: Real sync**

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops sync
```

- [ ] **Step 3: Verify the rewrite is surgical**

```bash
git diff docs/reports/grandmaster-tracker.md
```

Expected: changes ONLY to the `**Last Updated:**` line and `Current` cells in the Notebooks/Datasets/Competitions progress tables. Any other change is a bug — STOP and investigate. Do not commit yet (Task 8 also edits this file).

### Task 7: Scout refresh

- [ ] **Step 1: Regenerate the scout report**

```bash
.venv/bin/python -m kaggle_portfolio.notebooks.competition_scout --update
```

Expected: prints a ranked top-15 list; rewrites `docs/reports/competition-scout-report.md` with a fresh `*Generated: 2026-06-…*` timestamp and competitions actually open now. If all four category fetches fail it prints `Failed to fetch competitions. Check kaggle CLI credentials.` and exits 1 → STOP and report.

### Task 8: Refresh the tracker's active-competitions table (the one part sync doesn't touch)

**Files:**
- Modify: `docs/reports/grandmaster-tracker.md` ("Active competitions to enter" table + the `**Priority: …**` line under it)

- [ ] **Step 1: Rewrite the table from fresh scout data.** Open the new `docs/reports/competition-scout-report.md`; take the top entries (aim for 5-8) and rewrite the tracker's second Competitions table, keeping its exact column structure (`Competition | Teams | Deadline | Medal Difficulty | Strategy`) and date format `Mon DD, YYYY` — the medal_ops parser requires both. Drop every competition whose deadline has passed. Update the `**Priority: …**` line to the new best-odds competition (smallest field still open). Strategy cells: one short phrase per row (e.g. "Tabular ensemble baseline"); for competitions matching existing project dirs (e.g. a running Hull Tactical entry), keep the existing strategy text.

- [ ] **Step 2: Verify the tracker still parses**

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops doctor
```

Expected: exit 0; the staleness warning is GONE (Last Updated is now today); no parse errors on the active-competitions table.

- [ ] **Step 3: Commit the data refresh**

```bash
git add docs/reports/grandmaster-tracker.md docs/reports/competition-scout-report.md
git commit -m "Refresh tracker metrics and competition scout data from live Kaggle"
```

### Task 9: Regenerate medal_ops reports (order matters)

- [ ] **Step 1: Snapshot writers first, then consumers**

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops scorecard
.venv/bin/python -m kaggle_portfolio.ops.medal_ops pace
.venv/bin/python -m kaggle_portfolio.ops.medal_ops weekly-plan
.venv/bin/python -m kaggle_portfolio.ops.medal_ops badge-plan
```

Expected: each exits 0 and writes `medal_ops/reports/latest-<cmd>.md` (all gitignored — nothing to commit). `pace` will report velocity `n/a` (needs ≥2 snapshots ≥1 day apart — expected with a cold history). Skim `latest-scorecard.md` and `latest-weekly-plan.md`; their content feeds the final summary to the user.

### Task 10: Update CLAUDE.md to match reality

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add missing directories to the structure tree.** In the `kaggle/` tree block, add these entries (keep existing style/comments):

```
├── kaggle_portfolio/                  # Python package behind manage.sh (CLI, ops, quality, campaigns)
├── tests/                             # pytest suite for the kaggle_portfolio package
├── medal_ops/                         # Generated scorecards/plans/reports (gitignored except README)
├── pi-automation/                     # Dockerized Playwright/cron automation for Kaggle engagement
```

and under `docs/`, add `└── superpowers/` with `specs/` and `plans/`.

- [ ] **Step 2: Fix the Management Script section.** Note that `manage.sh` exposes 44 subcommands via `kaggle_portfolio/manage_commands.py`; show `./manage.sh help` as the discovery command and add the medal-ops family examples (`doctor`, `sync --dry-run`, `scorecard`, `weekly-plan`, `quality`, `scout --update`). Keep the existing push examples.

- [ ] **Step 3: Fix Key Context numbers and priorities.** Replace the stale "Current status" line with the values from the fresh tracker (read them from `docs/reports/grandmaster-tracker.md` after Task 6 — notebooks live, datasets published, competitions entered, current tier per category). Replace the "Priority competitions" line (Med-Gemma/Vesuvius/Akkadian — all closed) with the new top picks from Task 8. Convert "21 notebooks live" style claims to the synced values.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md to document package, tests, and current status"
```

### Task 11: Fix health-workflow flag order and replicate the health checks locally

**Files:**
- Modify: `.github/workflows/medal-ops-health.yml` (only the medal_ops invocations that place `--output-root`/`--today` before the subcommand)

- [ ] **Step 1: Fix flag placement.** In `medal-ops-health.yml`, move `--output-root /tmp/medal_ops_health` (and any `--tracker`/`--today`) to AFTER the medal_ops subcommand in each `python -m kaggle_portfolio.ops.medal_ops …` invocation (argparse subparser defaults clobber pre-subcommand flags, so today those reports silently land in the repo's `medal_ops/`). Leave non-medal_ops checks (notebook_quality, dataset_usability, discussion_scheduler) untouched — their flags are plain argparse without subparsers.

- [ ] **Step 2: Replicate the suite locally** (live creds exist here, so this approximates the workflow's live mode):

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops doctor --strict --max-stale-days 30 --output-root /tmp/medal_ops_health
.venv/bin/python -m kaggle_portfolio.quality.notebook_quality --output-root /tmp/medal_ops_health --scope all --min-score 95 --fail-under-threshold; echo "quality exit: $?"
.venv/bin/python -m kaggle_portfolio.datasets.dataset_usability --output-root /tmp/medal_ops_health --strict --fail-under 85; echo "usability exit: $?"
.venv/bin/python -m kaggle_portfolio.ops.discussion_scheduler --health-check; echo "scheduler exit: $?"
.venv/bin/python -m kaggle_portfolio.ops.medal_ops sync --dry-run --output-root /tmp/medal_ops_health
```

Expected: doctor and sync pass post-refresh. Quality at min-score 95 may legitimately fail (local avg was ~89) — that is notebook-content work, explicitly out of scope: record which notebooks are below 95 from `/tmp/medal_ops_health/reports/latest-notebook-quality-fixes.md` for the follow-up report, do NOT fix them here. Same for usability/scheduler: fix only cheap config/data issues; record the rest.

- [ ] **Step 3: Commit the workflow fix**

```bash
git add .github/workflows/medal-ops-health.yml
git commit -m "Pass medal_ops output-root after subcommand in health workflow"
```

### Task 12: Quality snapshot (repo-local, report-only)

- [ ] **Step 1: Standard report-only run into the repo's medal_ops/**

```bash
.venv/bin/python -m kaggle_portfolio.quality.notebook_quality --min-score 70 --scope all
```

Expected: exit 0, pass/improve summary on stdout, 8 report files under `medal_ops/reports/` (gitignored). Skim `latest-notebook-quality-fixes.md`; carry the top findings into the final summary.

### Task 13: Final verification and code review

- [ ] **Step 1: Full suite re-run**

```bash
.venv/bin/python -m pytest -q
```

Expected: same green result as Task 3 step 4.

- [ ] **Step 2: Review the branch diff** — `git log --oneline main..HEAD` and `git diff main --stat`. Expected commits: spec, plan, kaggle_utils fix, manage.sh fix, data refresh, CLAUDE.md, workflow fix. No gitignored artifacts, no `/Users/` literals in committed `*.py`/`*.sh` (CI guardrail):

```bash
git diff main --name-only | xargs grep -l "/Users/" 2>/dev/null | grep -vE "^tests/|\.md$" ; echo "guardrail exit: $? (1 = clean)"
```

- [ ] **Step 3: Run /code-review on the branch** (user-requested gate) and address findings with the superpowers:receiving-code-review discipline.

- [ ] **Step 4: Report.** Final summary must include: test counts before/after, what sync changed (old → new metrics), new top competitions, health-check replication results (incl. the quality-gate status at 95), the GitHub billing issue (user action required — Settings → Billing & plans; the daily cron cannot run until then), and scoped follow-ups (notebooks below 95, anything usability/scheduler flagged).
