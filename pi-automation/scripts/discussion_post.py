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

REPO = Path(os.environ.get("REPO_PATH", str(Path(__file__).parent.parent.parent)))
QUEUE_PATH = Path(os.environ.get("QUEUE_PATH", str(Path(__file__).parent.parent / "data" / "discussion_queue.json")))
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
        if next_item
        else "Queue empty."
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
