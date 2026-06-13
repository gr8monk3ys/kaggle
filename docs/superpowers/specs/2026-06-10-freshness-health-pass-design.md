# Repo Freshness + Health Pass — Design

**Date:** 2026-06-10
**Status:** Approved
**Branch:** `chore/freshness-health-pass`

## Problem

The repo's last commit is 2026-03-12. The grandmaster tracker and competition
scout report were generated 2026-03-10, and every competition they recommend
except Hull Tactical (deadline 2026-06-16) has already closed. CLAUDE.md does
not document the `kaggle_portfolio/` package, `tests/` suite, `medal_ops/`
reports, or `pi-automation/` directory, and its "Key Context" numbers (1
competition entered, 21 notebooks) contradict the tracker (12 entered, 65
notebooks live). Whether the automation still works after 3 months is unknown.

## Goal

Bring the repo current, verify the automation end-to-end, and document the
real structure — finishing with a code review of the branch diff.

## Approach

Tooling-first refresh: let the repo's own machinery (`medal_ops` doctor/sync,
scout, quality scorecard) do the data refresh, so the pass validates the
tooling and the content at the same time. Hand-editing generated artifacts is
avoided; only CLAUDE.md (not tool-generated) is edited by hand.

## Constraints

- Kaggle credentials exist at `~/.kaggle/kaggle.json`; the `kaggle` CLI is not
  installed. Live sync was approved, so the CLI gets installed (user-level
  `pip install kaggle`; fall back to a project `.venv` if the system Python is
  externally managed).
- Read-only against the Kaggle API: sync/scout pulls only. No notebook,
  dataset, or kernel pushes.
- Python 3.14.5 / pytest 9.0.3 is the local environment.

## Steps

1. **Test baseline.** Run the full pytest suite before any changes. Diagnose
   and fix failures first (distinguishing real regressions from environment
   issues), so later steps start from green.
2. **Environment.** Install the `kaggle` package so the CLI works with the
   existing credentials.
3. **Doctor.** Run `python3 -m kaggle_portfolio.ops.medal_ops doctor`; fix
   what it flags (tracker health, environment readiness).
4. **Live sync.** Run `medal_ops sync` to refresh tracker metrics, then
   regenerate scorecard, weekly-plan, and pace reports.
5. **Scout refresh.** Regenerate the competition scout report so it reflects
   competitions open as of 2026-06-10.
6. **CLAUDE.md update.** Add the undocumented directories to the structure
   section; correct the Key Context numbers from the fresh sync; note the
   medal_ops commands alongside `manage.sh`.
7. **Quality snapshot.** Run the notebook quality scorecard report-only, as
   input for a future improvement pass.
8. **Verify + review.** Re-run the full test suite, then run `/code-review`
   on the branch diff and address findings.

## Error handling

- Live API failure (expired creds, network): fall back to `sync --dry-run` /
  CSV mode and report exactly which data could not refresh. Never fabricate
  numbers.
- Test failures: report honestly with output; fix real regressions; document
  environment-only failures if they cannot be fixed in scope.

## Out of scope

- Notebook content fixes (the quality snapshot scopes that follow-up).
- New competition entries or submissions.
- pi-automation changes.
- Any push/publish to Kaggle.

## Success criteria

- Full test suite green.
- `doctor` passes.
- Tracker, scorecard, weekly plan, pace, and scout report dated 2026-06-10.
- CLAUDE.md structure and numbers match the repo and fresh tracker.
- Code review of the branch diff comes back clean (findings addressed).
