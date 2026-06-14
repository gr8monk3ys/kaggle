# Grandmaster Program — Phases 0–1 Design (Safety Rails + Measurement Loop)

**Date:** 2026-06-14
**Status:** Approved
**Branch:** `feat/grandmaster-phase-0-1`

## Problem

A multi-agent review of `kaggle_portfolio` (2026-06-13/14) found a repo with
substantial automation that is built but idle or mis-aimed, plus a handful of
latent bugs that would corrupt or leak if the automation were switched on:

- **Telemetry is a black hole.** `medal_ops sync` already writes
  `medal_ops/history/snapshot-{date}.json` and updates the tracker
  (medal_ops.py:1734, 1760), but both runners sabotage persistence: the pi
  `sync.sh` passes `--output-root /tmp/health` and the CI job passes
  `--output-root /tmp/... --dry-run`. The snapshot is written to `/tmp` and
  discarded every day, so `pace` is permanently stuck at "1 snapshot, velocity
  n/a" and ETA-to-Grandmaster is non-functional.
- **The assumed host does not exist.** The `pi-automation` stack that was
  presumed to run the daily cadence is not actually deployed (owner unsure), so
  nothing executes the engine on a schedule.
- **Gating bugs.** `FORUM_MAP` misroutes `"NLP Getting Started"` drafts to the
  wrong board and silently drops their priority; `metadata_tracker` records
  `votes=0` for every notebook on any Kaggle CLI failure and writes that
  corrupt snapshot unconditionally; `notify.py` echoes the Telegram bot token
  in error strings; a Kaggle session-cookie path defaults into a git-tracked
  directory; and `kaggle_auth_doctor --timeout` is wired to a parameter it
  never uses.

Baseline is green (781 tests pass, 0 fail). The fix is not to build more — it
is to **make progress measurable and install the rails that gate everything
else**, using only infrastructure that already exists.

## Goal

Make Grandmaster progress measurable and steerable with **no new
infrastructure**, and land the review fixes that must precede safe automation.
Concretely: a durable daily telemetry loop running on GitHub Actions and
committing its state into the repo, preceded by the small set of gating fixes.

## Program roadmap (all four initiatives)

Sequence (approved): **rails → measurement → engagement → lab**, with
code-health woven in rather than done big-bang.

| Phase | Initiative | Status |
|-------|------------|--------|
| 0 | **Safety rails** — review fixes that gate automation | This doc |
| 1 | **Measurement** — durable daily snapshot + pace + Telegram digest on GitHub Actions, committed to the repo | This doc |
| 2 | **Engagement** — wire the scheduler + 120 drafts to post on a cadence and track discussion votes | Future spec |
| 3 | **Competition lab** — add `benchmark_*` for the scout's real medal boards + leaderboard-rank telemetry | Future spec |
| — | **Code-health** — `pyproject`/installable early; split each god-module as it is touched; lint+typecheck CI gate; architecture docs | Woven in |

**Substrate decisions:** telemetry runs on **GitHub Actions** and commits state
to the repo — there is no always-on host today. The always-on-host question
(needed for browser posting) is deferred to **Phase 2**, where it is actually
forced. Each later phase gets its own spec → plan → implementation cycle.

## Phase 0 — Safety rails

Only the fixes that *gate* the automation. Every fix is test-first (a failing
test, then the change). The broader cleanup (dead code, time-bomb literals,
`--status` choices, etc.) is **out of scope** here — see Deferred.

| # | Fix | Location | Approach |
|---|-----|----------|----------|
| 1 | **Cookie-leak** | `.gitignore` (affects `pi-automation/scripts/kaggle_browser.py:23`, `dataset_metadata_sync.py:27`) | Add `**/kaggle_storage_state.json` so live Kaggle session cookies can never be staged. No code change. |
| 2 | **FORUM_MAP routing** | `kaggle_portfolio/ops/discussion_scheduler.py:143-146` | Resolve the forum by **longest matching key first** (sort `FORUM_MAP` keys by descending length before the substring match) so `"NLP Getting Started"` resolves to the competition board, not the generic one. Add a parametrized regression test pinning each ambiguous label. |
| 3 | **Zero-vote guard** | `kaggle_portfolio/ops/metadata_tracker.py:97-127,135,165-167,215` | Distinguish "fetch failed" from "zero votes": on non-zero returncode or exception, **log the captured error** (not bare `except: pass`) and signal failure; record votes as **absent/`None`** for affected notebooks, and have `cmd_report` skip delta computation when votes is `None` — never write `0`. |
| 4 | **auth_doctor `--timeout`** | `kaggle_portfolio/ops/kaggle_auth_doctor.py:107-150,209` | Pass `timeout=` to `subprocess.run` in `probe_public_listing`, and apply the configured timeout to the blob-upload probe so the flag is honored (the preflight gate must not hang a scheduled job). |
| 5 | **notify.py token redaction** | `pi-automation/scripts/notify.py:18,27,31` | Never log `str(exc)` / response text that contains the bot token; redact the token (`***`) in any message written to stderr. **Promoted into Phase 0** because the digest pipes through `notify.py` and Actions logs persist. |

## Phase 1 — Measurement loop (GitHub Actions + repo commits)

### Components

- **Telemetry workflow** — a **new, separate** daily scheduled workflow
  `.github/workflows/telemetry.yml` with `schedule:` + `workflow_dispatch:` and
  `permissions: contents: write`. Kept separate from `medal-ops-health.yml`
  (which is a read-only alerting gate): the recorder needs write access, and we
  do not want to broaden the health gate's permissions. The Kaggle-creds setup
  steps are duplicated from the health workflow — acceptable for clean
  separation. Steps:
  1. Checkout.
  2. Setup Python; install the `kaggle` CLI and configure credentials (reuse
     the existing live-mode steps from `medal-ops-health.yml`).
  3. `python -m kaggle_portfolio.ops.medal_ops sync` with the **default
     `--output-root` (repo `medal_ops/`) and no `--dry-run`** → refreshes
     `grandmaster-tracker.md` and appends `medal_ops/history/snapshot-{date}.json`.
  4. `medal_ops pace` + regenerate scorecard / weekly-plan.
  5. `scout --update` → refresh `competition-scout-report.md`; flag any board
     newly crossing the "ENTER NOW" threshold.
  6. Compose and send the digest via `notify.send()` (Telegram secrets in Actions).
  7. Commit changed tracked files: `chore(telemetry): daily snapshot {date} [skip ci]`,
     pushed with the default `GITHUB_TOKEN`.
- **Snapshot persistence** — un-gitignore `medal_ops/history/` so snapshots
  accumulate in git (Decision A). `reports/` and `sync_inputs/` stay ignored.
- **Digest composer** — a new `medal_ops digest` subcommand (consistent with
  the existing `scorecard` / `pace` subcommands) that reads the latest two
  snapshots, the tracker, draft-queue health, and the top badge-plan action,
  and returns a Telegram-ready Markdown string for `notify.send()`.

### Decisions

- **A — Un-gitignore `medal_ops/history/`.** Snapshots persist in git (one small
  JSON/day). *Alternative considered:* a dedicated `telemetry` data branch to
  keep `main`'s history clean — rejected for v1 in favor of in-repo visibility.
- **B — The tracker becomes auto-updated.** The daily live `sync` auto-commits
  live counts to `grandmaster-tracker.md` (no longer `--dry-run` / hand-curated).
  The snapshot history is the audit trail. This is *why* the Phase 0 fixes gate
  Phase 1: bad data would otherwise auto-commit.
- **C — Zero-vote: record `None` + log error** (not skip-write), so the metadata
  row survives while votes are honestly marked unknown.
- **D — FORUM_MAP: longest-key-first** (keeps the existing substring flexibility
  while fixing precedence).

### Data flow

```
kaggle CLI
  └─ medal_ops sync (default output-root, no --dry-run)
       ├─ grandmaster-tracker.md            (updated)
       └─ medal_ops/history/snapshot-DATE.json (appended)
            └─ medal_ops pace / digest ──▶ notify.send() (Telegram)
git commit { tracker, snapshot, scout report }  [skip ci]
```

### Digest v1 content

One Telegram message/day: **tracker deltas since last snapshot** +
**pace/ETA summary** + **nearest deadline** + **draft-queue health** +
**the single top badge-plan action for today.** Richer alerts
(leaderboard-near-bronze, new-board ENTER-NOW pings) are deferred to later
phases.

## Testing

- **Phase 0** — a failing unit test per fix before the change: FORUM_MAP
  routing (`"NLP Getting Started"` → competition URL), zero-vote → `None` on
  CLI failure (and report skips the delta), `--timeout` honored by the
  subprocess probe, token redacted from `notify.py` error output. Cookie-leak
  verified via `git check-ignore`.
- **Phase 1** — unit tests for the digest composer over fixture snapshots
  (delta math, and the single-snapshot fallback so the first run does not
  crash). The workflow is validated once via `workflow_dispatch` before the
  schedule is trusted. The full suite stays green (currently 781 passing).

## Deferred (explicitly out of scope)

- **Non-gating review findings** → a separate hygiene batch: dead code
  (`_parse_score`, `print_candidate_table`, unused `step` param), time-bomb
  literals (`dataset_optimizer` end-date/citation year, march-mania
  `season >= 2021`), `--status` choices mismatch, `--posts-per-day` slot cap,
  cover-image registry warning, duplicated TE/OOF scaffolding, and test-gaps
  (`medal_ops.main`, `is_stale`, brittle guardrail literals).
- **Metadata A/B tracker cadence** — its #4 fix lands now; turning on a daily
  metadata snapshot is optional and later.
- **Phases 2 (engagement) and 3 (lab)**, and the always-on-host decision they
  force.
- **Merging PRs #32/#33** — blocked on the `gh` token's `workflow` scope; an
  operational step, not code.

## Open questions / risks

- The daily live `sync` needs Kaggle creds + the `kaggle` CLI in Actions; the
  existing `medal-ops-health.yml` live mode already provides both, so reuse it.
  If creds/secrets are absent the job must **fail loudly** (and the digest must
  report the failure) rather than silently committing stale data.
- Auto-committing the tracker changes the curation model. If undesired, gate the
  commit behind a diff-size sanity check (e.g. refuse to commit if more than N
  metrics swing at once).
- **GitHub Actions billing is currently failing org-wide** — the telemetry
  workflow will not actually run on schedule until that account-level issue is
  resolved. Noted as a non-code prerequisite.
