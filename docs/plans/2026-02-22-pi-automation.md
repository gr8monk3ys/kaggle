# Pi Automation Container Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy a `kaggle-autobot` Docker container on Raspberry Pi that runs cron-driven health checks, live Kaggle sync, deadline alerts, weekly reports, and Playwright-based discussion posting with Telegram notifications — plus immediate Med-Gemma competition submission.

**Architecture:** Single ARM64 Docker container (Option A from design) with a cron daemon driving five scripts. A `discussion_queue.json` file controls post scheduling. All scripts share a `notify.py` Telegram dispatcher. Container mounts the live repo directory read-only so tracker/draft changes are picked up without rebuild. Med-Gemma submission is a separate urgent task handled outside the container.

**Tech Stack:** Python 3.11-slim (ARM64), Playwright (headless Chromium), Kaggle CLI, requests, cron inside container, Docker Compose.

---

### Task 1: Scaffold directory structure and entrypoint

**Files:**
- Create: `pi-automation/Dockerfile`
- Create: `pi-automation/entrypoint.sh`
- Create: `pi-automation/crontab`
- Create: `pi-automation/.env.example`
- Create: `pi-automation/scripts/__init__.py`
- Create: `pi-automation/scripts/requirements.txt`
- Create: `pi-automation/data/.gitkeep`
- Create: `pi-automation/tests/__init__.py`

**Step 1: Create directory layout**

```bash
mkdir -p pi-automation/scripts pi-automation/data pi-automation/tests
touch pi-automation/scripts/__init__.py pi-automation/tests/__init__.py pi-automation/data/.gitkeep
```

**Step 2: Write requirements.txt**

```
# pi-automation/scripts/requirements.txt
requests==2.31.0
playwright==1.44.0
kaggle==1.6.14
pytest==8.2.0
pytest-mock==3.14.0
```

**Step 3: Write Dockerfile**

```dockerfile
# pi-automation/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /scripts
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install chromium --with-deps

COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh 2>/dev/null || true

COPY crontab /etc/cron.d/kaggle-autobot
RUN chmod 0644 /etc/cron.d/kaggle-autobot && crontab /etc/cron.d/kaggle-autobot

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
```

**Step 4: Write entrypoint.sh**

```bash
#!/bin/bash
set -euo pipefail

# Source .env if mounted
if [ -f /run/secrets/.env ]; then
    set -a; source /run/secrets/.env; set +a
fi

# Write kaggle.json from env vars
mkdir -p /root/.kaggle
cat > /root/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USERNAME:-}","key":"${KAGGLE_KEY:-}"}
EOF
chmod 600 /root/.kaggle/kaggle.json

echo "kaggle-autobot starting cron..."
cron -f
```

**Step 5: Write crontab**

```
# pi-automation/crontab
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Daily health check + quality — 9:10 UTC
10 9 * * *   /scripts/health_check.sh >> /var/log/cron.log 2>&1

# Live sync — 9:15 UTC
15 9 * * *   /scripts/sync.sh >> /var/log/cron.log 2>&1

# Deadline alert — every 6 hours
0 */6 * * *  python3 /scripts/deadline_alert.py >> /var/log/cron.log 2>&1

# Weekly plan — Monday 8:00 UTC
0 8 * * 1    /scripts/weekly_report.sh >> /var/log/cron.log 2>&1

# Discussion poster — Tuesday and Friday 10:00 UTC
0 10 * * 2,5 python3 /scripts/discussion_post.py >> /var/log/cron.log 2>&1
```

**Step 6: Write .env.example**

```bash
# pi-automation/.env.example
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
KAGGLE_EMAIL=your_kaggle_email
KAGGLE_PASSWORD=your_kaggle_password
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
REPO_PATH=/repo
QUEUE_PATH=/data/discussion_queue.json
```

**Step 7: Validate shell syntax**

```bash
bash -n pi-automation/entrypoint.sh
```
Expected: no output

**Step 8: Commit**

```bash
git add pi-automation/
git commit -m "feat: scaffold pi-automation container structure"
```

---

### Task 2: notify.py — Telegram dispatcher

**Files:**
- Create: `pi-automation/scripts/notify.py`
- Create: `pi-automation/tests/test_notify.py`

**Step 1: Write the failing tests**

```python
# pi-automation/tests/test_notify.py
import importlib
import sys
from unittest.mock import patch

sys.path.insert(0, "pi-automation/scripts")


def _reload_notify():
    if "notify" in sys.modules:
        del sys.modules["notify"]
    import notify
    return notify


def test_send_calls_telegram_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    n = _reload_notify()
    with patch("requests.post") as mock_post:
        mock_post.return_value.ok = True
        n.send("hello world")
    mock_post.assert_called_once()
    args = mock_post.call_args
    assert "test-token" in args[0][0]
    assert args[1]["json"]["text"] == "hello world"
    assert args[1]["json"]["chat_id"] == "12345"


def test_send_raises_on_missing_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    import pytest
    n = _reload_notify()
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        n.send("test")


def test_send_prints_stderr_on_api_failure(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    n = _reload_notify()
    with patch("requests.post") as mock_post:
        mock_post.return_value.ok = False
        mock_post.return_value.text = "Forbidden"
        n.send("hello")
    captured = capsys.readouterr()
    assert "Telegram send failed" in captured.err
```

**Step 2: Run tests — expect failure**

```bash
python -m pytest pi-automation/tests/test_notify.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'notify'`

**Step 3: Implement notify.py**

```python
# pi-automation/scripts/notify.py
"""Telegram notification dispatcher for kaggle-autobot."""
from __future__ import annotations

import os
import sys
import requests


def send(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=10,
    )
    if not response.ok:
        print(f"Telegram send failed: {response.text}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()
    send(args.message)
```

**Step 4: Run tests — expect pass**

```bash
python -m pytest pi-automation/tests/test_notify.py -v
```
Expected: 3 PASS

**Step 5: Commit**

```bash
git add pi-automation/scripts/notify.py pi-automation/tests/test_notify.py
git commit -m "feat: add Telegram notify dispatcher with tests"
```

---

### Task 3: deadline_alert.py — parse tracker deadlines

**Files:**
- Create: `pi-automation/scripts/deadline_alert.py`
- Create: `pi-automation/tests/test_deadline_alert.py`

**Step 1: Write the failing tests**

```python
# pi-automation/tests/test_deadline_alert.py
import sys
from datetime import date
sys.path.insert(0, "pi-automation/scripts")
sys.path.insert(0, ".")   # repo root — for medal_ops import

import deadline_alert

SAMPLE_TRACKER = """
## Current Progress (2026-01-25)

### Competitions
| Status | Target | Current |
|--------|--------|---------|
| Tier | Grandmaster | Novice |

**Active competitions to enter:**
| Competition | Teams | Deadline | Medal Difficulty | Strategy |
|-------------|-------|----------|-----------------|----------|
| Med-Gemma Impact Challenge | 58 | Feb 24, 2026 | Easiest | Fine-tune |
| Akkadian Translation | 1321 | Mar 23, 2026 | Hard | ByT5 |
| Already Expired | 100 | Jan 01, 2026 | Easy | skip |
"""


def test_parse_deadlines_finds_all_rows():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    assert len(deadlines) == 3


def test_filter_urgent_finds_within_72h():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    urgent = deadline_alert.filter_urgent(deadlines, hours=72)
    assert len(urgent) == 1
    assert urgent[0].competition == "Med-Gemma Impact Challenge"


def test_filter_urgent_excludes_past():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    urgent = deadline_alert.filter_urgent(deadlines, hours=72)
    names = [d.competition for d in urgent]
    assert "Already Expired" not in names


def test_format_alert_contains_key_fields():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    urgent = deadline_alert.filter_urgent(deadlines, hours=72)
    msg = deadline_alert.format_alert(urgent[0])
    assert "Med-Gemma" in msg
    assert "Feb 24" in msg
```

**Step 2: Run — expect failure**

```bash
python -m pytest pi-automation/tests/test_deadline_alert.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement deadline_alert.py**

```python
# pi-automation/scripts/deadline_alert.py
"""Alert via Telegram when a competition deadline is within the threshold."""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(os.environ.get("REPO_PATH", "/repo"))))

import notify
from medal_ops import parse_active_competitions, ParsedDeadline

REPO = Path(os.environ.get("REPO_PATH", "/repo"))
TRACKER_PATH = REPO / "grandmaster-tracker.md"
ALERT_HOURS = 72


def parse_deadlines(content: str, today: date) -> list[ParsedDeadline]:
    return parse_active_competitions(content, today)


def filter_urgent(deadlines: list[ParsedDeadline], hours: int = 72) -> list[ParsedDeadline]:
    threshold_days = hours / 24
    return [
        d for d in deadlines
        if d.days_to_deadline is not None and 0 <= d.days_to_deadline <= threshold_days
    ]


def format_alert(d: ParsedDeadline) -> str:
    hours_left = int((d.days_to_deadline or 0) * 24)
    return (
        f"⏰ *DEADLINE ALERT*\n"
        f"{d.competition}\n"
        f"Due in ~{hours_left}h ({d.deadline_raw})\n"
        f"Teams: {d.teams} | {d.difficulty}\n"
        f"Strategy: {d.strategy}"
    )


def main() -> None:
    today = date.today()
    if not TRACKER_PATH.exists():
        print(f"Tracker not found: {TRACKER_PATH}", file=sys.stderr)
        sys.exit(1)
    content = TRACKER_PATH.read_text(encoding="utf-8")
    deadlines = parse_deadlines(content, today)
    urgent = filter_urgent(deadlines, hours=ALERT_HOURS)
    if not urgent:
        print(f"No deadlines within {ALERT_HOURS}h.")
        return
    for d in urgent:
        msg = format_alert(d)
        notify.send(msg)
        print(f"Alert sent: {d.competition}")


if __name__ == "__main__":
    main()
```

**Step 4: Run — expect pass**

```bash
python -m pytest pi-automation/tests/test_deadline_alert.py -v
```
Expected: 4 PASS

**Step 5: Commit**

```bash
git add pi-automation/scripts/deadline_alert.py pi-automation/tests/test_deadline_alert.py
git commit -m "feat: add deadline alert script with tests"
```

---

### Task 4: discussion_queue.py — queue reader/writer

**Files:**
- Create: `pi-automation/scripts/discussion_queue.py`
- Create: `pi-automation/tests/test_discussion_queue.py`
- Create: `pi-automation/data/discussion_queue.json`

**Step 1: Write the failing tests**

```python
# pi-automation/tests/test_discussion_queue.py
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "pi-automation/scripts")

import discussion_queue as dq

SAMPLE_QUEUE = [
    {"id": "d1", "title": "Post One", "forum_url": "https://kaggle.com/discussions/getting-started",
     "body_file": "discussion-drafts.md", "body_section": "Draft 1",
     "status": "pending", "scheduled_after": "2026-02-01T00:00:00Z"},
    {"id": "d2", "title": "Post Two", "forum_url": "https://kaggle.com/discussions/general",
     "body_file": "discussion-drafts.md", "body_section": "Draft 2",
     "status": "posted", "scheduled_after": "2026-01-01T00:00:00Z"},
    {"id": "d3", "title": "Future Post", "forum_url": "https://kaggle.com/discussions/general",
     "body_file": "discussion-drafts.md", "body_section": "Draft 3",
     "status": "pending", "scheduled_after": "2099-01-01T00:00:00Z"},
]
NOW = datetime(2026, 2, 22, 10, 0, tzinfo=timezone.utc)


def test_next_pending_returns_first_eligible():
    item = dq.next_pending(SAMPLE_QUEUE, now=NOW)
    assert item is not None
    assert item["id"] == "d1"


def test_next_pending_skips_posted():
    item = dq.next_pending(SAMPLE_QUEUE, now=NOW)
    assert item["id"] != "d2"


def test_next_pending_skips_future_scheduled():
    item = dq.next_pending(SAMPLE_QUEUE, now=NOW)
    assert item["id"] != "d3"


def test_next_pending_returns_none_when_nothing_ready():
    assert dq.next_pending([], now=NOW) is None


def test_mark_posted_updates_queue_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_QUEUE, f)
        path = Path(f.name)
    dq.mark_posted(path, "d1", post_url="https://kaggle.com/discussion/42")
    updated = json.loads(path.read_text())
    item = next(i for i in updated if i["id"] == "d1")
    assert item["status"] == "posted"
    assert item["post_url"] == "https://kaggle.com/discussion/42"
    assert "posted_at" in item
    path.unlink()


def test_extract_body_finds_section():
    content = (
        "## Draft 1: Feature Engineering\n\n"
        "**Target forum:** Getting Started\n\n"
        "### Feature Engineering\n\nBody content here.\n\n---\n\n"
        "## Draft 2: Other\n\nOther content.\n"
    )
    body = dq.extract_body(content, "Draft 1")
    assert "Body content here" in body
    assert "Draft 2" not in body
    assert "Target forum" not in body
```

**Step 2: Run — expect failure**

```bash
python -m pytest pi-automation/tests/test_discussion_queue.py -v
```
Expected: FAIL

**Step 3: Implement discussion_queue.py**

```python
# pi-automation/scripts/discussion_queue.py
"""Queue management for scheduled Kaggle discussion posts."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def next_pending(queue: list[dict], now: datetime | None = None) -> dict | None:
    if now is None:
        now = datetime.now(tz=timezone.utc)
    for item in queue:
        if item.get("status") != "pending":
            continue
        try:
            scheduled = datetime.fromisoformat(
                item.get("scheduled_after", "").replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if now >= scheduled:
            return item
    return None


def mark_posted(queue_path: Path, item_id: str, post_url: str) -> None:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    for item in queue:
        if item["id"] == item_id:
            item["status"] = "posted"
            item["post_url"] = post_url
            item["posted_at"] = datetime.now(tz=timezone.utc).isoformat()
            break
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def extract_body(drafts_content: str, section_name: str) -> str:
    pattern = re.compile(
        rf"## {re.escape(section_name)}:.*?\n(.*?)(?=\n## |\Z)", re.DOTALL
    )
    match = pattern.search(drafts_content)
    if not match:
        raise ValueError(f"Section not found in drafts: {section_name}")
    body = match.group(1).strip()
    # Strip metadata lines
    for meta in (r"\*\*Target forum:\*\*.*\n?", r"\*\*Category:\*\*.*\n?",
                 r"\*\*Expected medal:\*\*.*\n?"):
        body = re.sub(meta, "", body)
    # Strip the ### heading (it becomes the post title, not the body)
    body = re.sub(r"^###.+\n\n?", "", body.lstrip())
    return body.strip()
```

**Step 4: Create pi-automation/data/discussion_queue.json**

```json
[
  {"id":"draft-1","title":"5 Feature Engineering Tricks That Won Me Bronze","forum_url":"https://www.kaggle.com/discussions/getting-started","body_file":"discussion-drafts.md","body_section":"Draft 1","status":"pending","scheduled_after":"2026-02-24T10:00:00Z"},
  {"id":"draft-2","title":"Med-Gemma Challenge: Initial EDA Findings","forum_url":"https://www.kaggle.com/competitions/med-gemma-challenge/discussion","body_file":"discussion-drafts.md","body_section":"Draft 2","status":"pending","scheduled_after":"2026-02-27T10:00:00Z"},
  {"id":"draft-3","title":"Akkadian Translation: Understanding the Data","forum_url":"https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion","body_file":"discussion-drafts.md","body_section":"Draft 3","status":"pending","scheduled_after":"2026-03-03T10:00:00Z"},
  {"id":"draft-4","title":"Complete Guide to Ensemble Methods for Kaggle Competitions","forum_url":"https://www.kaggle.com/discussions/getting-started","body_file":"discussion-drafts.md","body_section":"Draft 4","status":"pending","scheduled_after":"2026-03-06T10:00:00Z"},
  {"id":"draft-5","title":"RAG Systems: What I Learned Building One From Scratch","forum_url":"https://www.kaggle.com/discussions/general","body_file":"discussion-drafts.md","body_section":"Draft 5","status":"pending","scheduled_after":"2026-03-10T10:00:00Z"},
  {"id":"draft-6","title":"Attention Mechanisms Visualized: A Practical Guide","forum_url":"https://www.kaggle.com/discussions/getting-started","body_file":"discussion-drafts.md","body_section":"Draft 6","status":"pending","scheduled_after":"2026-03-13T10:00:00Z"},
  {"id":"draft-7","title":"Time Series Pitfalls: Don't Random Split Your Data!","forum_url":"https://www.kaggle.com/discussions/getting-started","body_file":"discussion-drafts.md","body_section":"Draft 7","status":"pending","scheduled_after":"2026-03-17T10:00:00Z"},
  {"id":"draft-8","title":"My End-to-End ML Competition Pipeline","forum_url":"https://www.kaggle.com/discussions/getting-started","body_file":"discussion-drafts.md","body_section":"Draft 8","status":"pending","scheduled_after":"2026-03-20T10:00:00Z"},
  {"id":"draft-9","title":"Vesuvius Challenge: 3D Segmentation Approaches","forum_url":"https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion","body_file":"discussion-drafts.md","body_section":"Draft 9","status":"pending","scheduled_after":"2026-03-24T10:00:00Z"},
  {"id":"draft-10","title":"Top 10 Kaggle Notebooks Every Beginner Should Read","forum_url":"https://www.kaggle.com/discussions/getting-started","body_file":"discussion-drafts.md","body_section":"Draft 10","status":"pending","scheduled_after":"2026-03-27T10:00:00Z"}
]
```

**Step 5: Run — expect pass**

```bash
python -m pytest pi-automation/tests/test_discussion_queue.py -v
```
Expected: 6 PASS

**Step 6: Commit**

```bash
git add pi-automation/scripts/discussion_queue.py pi-automation/tests/test_discussion_queue.py pi-automation/data/discussion_queue.json
git commit -m "feat: add discussion queue manager with tests and initial 10-post schedule"
```

---

### Task 5: discussion_post.py — Playwright browser automation

**Files:**
- Create: `pi-automation/scripts/discussion_post.py`

Note: The browser interaction cannot be unit tested without a real browser. The queue logic (`next_pending`, `mark_posted`, `extract_body`) is covered by Task 4 tests. This task implements the Playwright orchestration layer.

**Step 1: Implement discussion_post.py**

```python
# pi-automation/scripts/discussion_post.py
"""Post the next queued discussion to Kaggle using Playwright."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

sys.path.insert(0, str(Path(__file__).parent))
import notify
import discussion_queue as dq

REPO = Path(os.environ.get("REPO_PATH", "/repo"))
QUEUE_PATH = Path(os.environ.get("QUEUE_PATH", "/data/discussion_queue.json"))
EMAIL = os.environ.get("KAGGLE_EMAIL", "")
PASSWORD = os.environ.get("KAGGLE_PASSWORD", "")


def login(page) -> None:
    page.goto("https://www.kaggle.com/account/login", wait_until="networkidle")
    page.fill('input[name="email"]', EMAIL)
    page.fill('input[name="password"]', PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_url("https://www.kaggle.com/", timeout=20000)


def post_discussion(page, forum_url: str, title: str, body: str) -> str:
    page.goto(forum_url, wait_until="networkidle")
    page.click("text=New Topic", timeout=10000)
    page.wait_for_selector('input[name="title"]', timeout=10000)
    page.fill('input[name="title"]', title)
    editor = page.locator('[contenteditable="true"]').first
    editor.click()
    editor.fill(body)
    page.click('button:has-text("Post")', timeout=10000)
    page.wait_for_load_state("networkidle", timeout=20000)
    return page.url


def main() -> None:
    if not QUEUE_PATH.exists():
        print(f"Queue not found: {QUEUE_PATH}", file=sys.stderr)
        sys.exit(1)

    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    now = datetime.now(tz=timezone.utc)
    item = dq.next_pending(queue, now=now)

    if item is None:
        print("No pending posts ready.")
        return

    drafts_path = REPO / item["body_file"]
    try:
        body = dq.extract_body(drafts_path.read_text(encoding="utf-8"), item["body_section"])
    except (FileNotFoundError, ValueError) as e:
        notify.send(f"❌ Cannot extract draft body: {e}")
        sys.exit(1)

    print(f"Posting: {item['title']}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            login(page)
            post_url = post_discussion(page, item["forum_url"], item["title"], body)
            browser.close()
    except (PlaywrightTimeout, Exception) as e:
        notify.send(f"❌ Post failed: {item['title']}\n{e}")
        sys.exit(1)

    dq.mark_posted(QUEUE_PATH, item["id"], post_url=post_url)

    updated = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    next_item = dq.next_pending(updated, now=now)
    next_info = (
        f"Next: {next_item['title']} ({next_item['scheduled_after'][:10]})"
        if next_item else "Queue empty."
    )

    notify.send(
        f"✅ *Discussion posted*\n"
        f"\"{item['title']}\"\n"
        f"{post_url}\n\n"
        f"{next_info}"
    )
    print(f"Posted: {post_url}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add pi-automation/scripts/discussion_post.py
git commit -m "feat: add Playwright discussion poster"
```

---

### Task 6: Shell scripts — health_check, sync, weekly_report

**Files:**
- Create: `pi-automation/scripts/health_check.sh`
- Create: `pi-automation/scripts/sync.sh`
- Create: `pi-automation/scripts/weekly_report.sh`

**Step 1: Write health_check.sh**

```bash
#!/bin/bash
# pi-automation/scripts/health_check.sh
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

doctor_status="✅ PASS"
quality_status="✅ PASS"

(cd "${REPO}" && python3 -m kaggle_portfolio.ops.medal_ops --output-root /tmp/health --today "${TODAY}" \
    doctor --strict) 2>&1 | tee /tmp/doctor.log || doctor_status="❌ FAIL"

(cd "${REPO}" && python3 -m kaggle_portfolio.quality.notebook_quality --output-root /tmp/health --today "${TODAY}" \
    --scope all --min-score 95 --fail-under-threshold) 2>&1 | tee /tmp/quality.log \
    || quality_status="❌ FAIL"

python3 /scripts/notify.py "📊 *Kaggle Health — ${TODAY}*
Doctor: ${doctor_status}
Quality: ${quality_status}"
```

**Step 2: Write sync.sh**

```bash
#!/bin/bash
# pi-automation/scripts/sync.sh
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

if (cd "${REPO}" && python3 -m kaggle_portfolio.ops.medal_ops --output-root /tmp/health --today "${TODAY}" \
    sync) 2>&1 | tee /tmp/sync.log; then
    python3 /scripts/notify.py "🔄 *Sync complete — ${TODAY}*"
else
    python3 /scripts/notify.py "❌ *Sync failed — ${TODAY}*"
fi
```

**Step 3: Write weekly_report.sh**

```bash
#!/bin/bash
# pi-automation/scripts/weekly_report.sh
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

(cd "${REPO}" && python3 -m kaggle_portfolio.ops.medal_ops --output-root /tmp/health --today "${TODAY}" \
    weekly-plan) 2>&1 | tee /tmp/weekly.log

REPORT=$(head -60 /tmp/health/reports/latest-weekly-plan.md 2>/dev/null || echo "Report not generated.")
python3 /scripts/notify.py "📅 *Weekly Plan — ${TODAY}*

${REPORT}"
```

**Step 4: Validate syntax**

```bash
bash -n pi-automation/scripts/health_check.sh
bash -n pi-automation/scripts/sync.sh
bash -n pi-automation/scripts/weekly_report.sh
```
Expected: no output

**Step 5: Commit**

```bash
git add pi-automation/scripts/health_check.sh pi-automation/scripts/sync.sh pi-automation/scripts/weekly_report.sh
git commit -m "feat: add health check, sync, and weekly report shell scripts"
```

---

### Task 7: docker-compose.yml

**Files:**
- Modify: `pi-automation/Dockerfile` (finalize)
- Create: `pi-automation/docker-compose.yml`

**Step 1: Write docker-compose.yml**

```yaml
# pi-automation/docker-compose.yml
# Extend your existing Pi stack:
#   docker compose -f /path/to/existing.yml -f pi-automation/docker-compose.yml up -d kaggle-autobot

services:
  kaggle-autobot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: kaggle-autobot
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      # Repo: read-only — scripts read tracker and drafts live
      - ..:/repo:ro
      # Queue and posted history — writable
      - ./data:/data
      # Cron log persistence
      - kaggle-autobot-logs:/var/log
    environment:
      - REPO_PATH=/repo
      - QUEUE_PATH=/data/discussion_queue.json
    network_mode: bridge

volumes:
  kaggle-autobot-logs:
```

**Step 2: Add .gitignore entry so .env is never committed**

```bash
echo "pi-automation/.env" >> .gitignore
```

**Step 3: Verify compose file parses (run on any machine with Docker)**

```bash
docker compose -f pi-automation/docker-compose.yml config --quiet
```
Expected: exits 0

**Step 4: Commit**

```bash
git add pi-automation/docker-compose.yml .gitignore
git commit -m "feat: add docker-compose for kaggle-autobot Pi deployment"
```

---

### Task 8: Wire pi-automation tests into CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add pi-automation test step**

In `.github/workflows/ci.yml`, add after the existing `Unit tests` step:

```yaml
      - name: Pi automation unit tests
        run: |
          python -m pip install requests pytest pytest-mock
          pytest pi-automation/tests/ -q
```

**Step 2: Run locally to confirm all tests pass**

```bash
python -m pytest pi-automation/tests/ -v
```
Expected: all PASS

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add pi-automation unit tests"
```

---

### Task 9: Deployment guide and DEPLOY.md

**Files:**
- Create: `pi-automation/DEPLOY.md`

**Step 1: Write DEPLOY.md**

```markdown
# Deploying kaggle-autobot to Raspberry Pi

## Prerequisites

- Pi running Docker + Docker Compose v2
- Repo cloned: `git clone <repo-url> && cd kaggle`
- Kaggle credentials from https://www.kaggle.com/settings → API → Create New Token
- Telegram bot token from @BotFather on Telegram
- Your Telegram chat ID: start a chat with the bot, then visit
  `https://api.telegram.org/bot{TOKEN}/getUpdates` and find `"id"` in the result

## One-time setup

```bash
cd pi-automation
cp .env.example .env
nano .env          # fill in all values
```

## Build and start

```bash
docker compose -f pi-automation/docker-compose.yml up -d --build
```

First build takes 10-15 minutes on Pi 4 (Playwright Chromium install).

## Verify

```bash
# Check it started
docker logs kaggle-autobot

# Test Telegram
docker exec kaggle-autobot python3 /scripts/notify.py "hello from Pi"

# Test deadline check
docker exec kaggle-autobot python3 /scripts/deadline_alert.py

# Force a discussion post (ignore schedule)
docker exec -e FORCE_POST=1 kaggle-autobot python3 /scripts/discussion_post.py
```

## Updating

After `git pull`:
```bash
# Scripts changed → rebuild
docker compose -f pi-automation/docker-compose.yml up -d --build

# Only tracker/drafts changed → restart (repo is mounted live)
docker compose -f pi-automation/docker-compose.yml restart kaggle-autobot
```

## Logs

```bash
docker exec kaggle-autobot tail -f /var/log/cron.log
```
```

**Step 2: Commit**

```bash
git add pi-automation/DEPLOY.md
git commit -m "docs: add Pi deployment guide"
```

---

### Task 10: Med-Gemma submission (urgent — deadline Feb 24)

Do this now. It is independent of the Pi automation.

**Step 1: Check the competition's expected submission format**

```bash
kaggle competitions download -c med-gemma-challenge --path /tmp/medgemma
ls /tmp/medgemma/
```

Look for `sample_submission.csv`. Note the column names and expected values.

**Step 2: Inspect what the local notebook currently outputs**

```bash
python3 -c "
import json
nb = json.load(open('projects/competitions/med-gemma-challenge/med_gemma_eda.ipynb'))
for cell in nb['cells'][-5:]:
    print('---', cell.get('cell_type'))
    print(''.join(cell.get('source', []))[:300])
"
```

If the notebook doesn't yet produce a `submission.csv`, identify the gap and add the inference + output cell.

**Step 3: Push the notebook to Kaggle**

```bash
./manage.sh push med-gemma-challenge
```

**Step 4: On kaggle.com — run the notebook on Kaggle GPUs**

- Open: https://www.kaggle.com/code/lorenzoscaturchio/med-gemma-challenge-eda
- Click "Run All" → wait for completion
- Download output files

**Step 5: Submit predictions**

```bash
kaggle competitions submit \
  -c med-gemma-challenge \
  -f /path/to/submission.csv \
  -m "Baseline MedGemma LoRA — first submission"
```

**Step 6: Update tracker and commit**

In `grandmaster-tracker.md`, mark Med-Gemma as entered in the Competitions section.

```bash
git add grandmaster-tracker.md
git commit -m "track: record Med-Gemma first submission"
```
