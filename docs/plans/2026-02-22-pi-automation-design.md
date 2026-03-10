# Raspberry Pi Automation Container Design

**Date:** 2026-02-22
**Status:** Approved
**Goal:** Fully automated Kaggle portfolio operations running on the Raspberry Pi — health checks, live sync, deadline alerts, weekly reports, and browser-based discussion posting — with Telegram notifications.

---

## Context

The repo already has:
- `manage.sh` — push, sync, doctor, quality, scorecard, weekly-plan, pace
- `kaggle_portfolio.ops.medal_ops` — parses tracker, generates reports, syncs live data
- `kaggle_portfolio.quality.notebook_quality` — scores notebooks against rubric
- `.github/workflows/medal-ops-health.yml` — daily CI health check (GitHub Actions)

What's missing: a persistent local agent that runs on the Pi, handles live Kaggle API calls (CI runs offline-fixture), posts discussions via browser automation, and notifies via Telegram.

---

## Architecture

Single Docker container (`kaggle-autobot`) added to the existing Pi compose stack. ARM64-compatible image. Cron daemon inside the container drives all jobs. Volumes mount the repo directory and a secrets file so the container sees live tracker changes without rebuild.

```
pi-automation/
├── Dockerfile
├── docker-compose.yml        # additive — references existing network
├── .env.example
├── crontab
└── scripts/
    ├── health_check.sh       # doctor + quality → Telegram digest
    ├── sync.sh               # live kaggle sync → update tracker
    ├── deadline_alert.py     # parses tracker deadlines → Telegram alert if <72h
    ├── weekly_report.sh      # weekly-plan report → Telegram
    ├── discussion_post.py    # Playwright: posts next item from queue
    └── notify.py             # Telegram dispatcher used by all scripts
```

Supporting data files (mounted volume, not in image):
```
pi-automation/data/
├── discussion_queue.json     # ordered list of posts to publish
└── posted_history.json       # audit log of what was posted and when
```

---

## Cron Schedule

```
# health check + quality score
10 9 * * *   /scripts/health_check.sh

# live sync (runs 5 min after health check so doctor output is fresh)
15 9 * * *   /scripts/sync.sh

# deadline alert — every 6 hours
0 */6 * * *  python3 /scripts/deadline_alert.py

# weekly plan report — Monday 8am UTC
0 8 * * 1    /scripts/weekly_report.sh

# discussion poster — Tuesday and Friday 10am UTC (2 posts/week pace)
0 10 * * 2,5 python3 /scripts/discussion_post.py
```

---

## Discussion Queue System

`discussion_queue.json` is a JSON array of pending posts. Each entry has:

```json
[
  {
    "id": "draft-1",
    "title": "5 Feature Engineering Tricks That Won Me Bronze",
    "forum_url": "https://www.kaggle.com/discussions/getting-started",
    "body_file": "docs/discussions/discussion-drafts.md",
    "body_section": "Draft 1",
    "status": "pending",
    "scheduled_after": "2026-02-24T10:00:00Z"
  }
]
```

The `discussion_post.py` script:
1. Reads the queue, finds the next `pending` item past its `scheduled_after` time
2. Launches Playwright (headless Chromium, ARM64 build)
3. Logs into Kaggle using stored credentials
4. Navigates to `forum_url`, opens "New Topic"
5. Fills title + body (extracted from `body_file` by `body_section` header)
6. Submits
7. Captures the live post URL from the redirect
8. Sends Telegram message: "✓ Posted: [title] → [url]"
9. Updates `discussion_queue.json`: `status: "posted"`, records timestamp and URL in `posted_history.json`

If Playwright fails (selector mismatch, login issue), it sends a Telegram alert with the error and leaves the item as `pending` for the next run.

---

## Notification Contract

All scripts call `notify.py` with a message string. `notify.py` sends to the Telegram Bot API:

```
POST https://api.telegram.org/bot{TOKEN}/sendMessage
{ chat_id: CHAT_ID, text: message, parse_mode: "Markdown" }
```

Secrets come from environment variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) injected via `.env`.

### Message formats

**Daily health digest:**
```
📊 Kaggle Health – 2026-02-22
Doctor: ✅ PASS
Quality: ✅ 18/21 notebooks ≥95
Sync: ✅ Updated tracker
Notebooks: 4 votes | Datasets: 0 votes | Discussions: 0 posts
```

**Deadline alert:**
```
⏰ DEADLINE ALERT
Med-Gemma Impact Challenge
Due in 47 hours (Feb 24, 2026)
Action: submit predictions
```

**Discussion posted:**
```
✅ Discussion posted
"5 Feature Engineering Tricks That Won Me Bronze"
https://www.kaggle.com/discussions/getting-started/12345
Next post scheduled: Fri Feb 28
```

---

## Docker Image

Base: `python:3.11-slim` (multi-arch, runs on ARM64 Pi 4/5).
Playwright installs ARM64 Chromium via `playwright install chromium`.
Kaggle CLI installed via pip.
Cron via `cron` package, runs as `cron -f` in foreground (so Docker sees the PID).

Image is built on the Pi (`docker buildx build --platform linux/arm64`), not pulled from a registry, so no registry credentials needed.

---

## Volumes and Secrets

```yaml
volumes:
  - /path/to/kaggle-repo:/repo          # live repo mount
  - /path/to/pi-automation/data:/data   # queue + history
  - /path/to/.env:/run/secrets/.env     # credentials (read-only)
```

Credentials in `.env`:
```
KAGGLE_USERNAME=lorenzoscaturchio
KAGGLE_KEY=xxxx
TELEGRAM_BOT_TOKEN=xxxx
TELEGRAM_CHAT_ID=xxxx
KAGGLE_EMAIL=xxxx
KAGGLE_PASSWORD=xxxx        # for Playwright login
```

`kaggle.json` is generated at container start from `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars.

---

## Med-Gemma (Immediate — Separate from Pi)

This needs to happen today, not via Pi automation. Steps:
1. Check the competition's `sample_submission.csv` to understand the expected output format
2. Review the local `projects/competitions/med-gemma-challenge/` notebook to see what it currently outputs
3. Push the notebook to Kaggle and run it on Kaggle GPUs
4. Download the output `submission.csv`
5. Submit via `kaggle competitions submit`

This is manual and urgent (deadline: Feb 24). The Pi automation handles future competitions.

---

## What This Does NOT Do

- Push notebooks to Kaggle on a schedule (only on manual `manage.sh push`)
- Automatically train models or generate competition submissions
- Auto-merge tracker changes to git (sync updates tracker file but git commit is manual)
- Replace the GitHub Actions CI (that continues to run on push/PR as before)

---

## Success Criteria

- Pi container survives reboots (`restart: unless-stopped`)
- Daily Telegram digest arrives every morning without manual intervention
- Deadline alerts fire reliably when a competition is <72 hours from close
- Discussion posts at the 2x/week cadence with Telegram confirmation per post
- Zero interaction needed to keep the portfolio operational week-to-week
