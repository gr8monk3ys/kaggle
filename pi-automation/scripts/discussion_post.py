"""Post the next queued discussion to Kaggle using Playwright."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# The repo root has to be importable before the shared Draft Queue model can be.
# Default to /repo, not a __file__-relative walk: inside the container this file
# is /scripts/discussion_post.py, so parents[2] resolves to "/" and every
# repo-relative path below silently points at the filesystem root.
REPO = Path(os.environ.get("REPO_PATH", "/repo"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import notify  # noqa: E402
from kaggle_portfolio.discussions import draft_queue as dq  # noqa: E402
from kaggle_portfolio.shared._browser_session import (  # noqa: E402,F401
    is_browser_challenge,
    require_playwright,
)

QUEUE_PATH = Path(
    os.environ.get(
        "QUEUE_PATH",
        str(Path(__file__).parent.parent / "data" / "discussion_queue.json"),
    )
)
#: Publishing is off unless explicitly enabled. See
#: docs/adr/0004-posting-stays-off-by-default.md — the unified selector changes
#: which drafts are eligible, and publishing is the one irreversible effect here.
POSTING_ENABLED = os.environ.get("DISCUSSION_POSTING_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EMAIL = os.environ.get("KAGGLE_EMAIL", "")
PASSWORD = os.environ.get("KAGGLE_PASSWORD", "")
BROWSER_CHALLENGE_MESSAGE = (
    "Kaggle browser challenge detected. Clear the Cloudflare/reCAPTCHA check in a headed browser "
    "and retry."
)


def notify_safe(message: str) -> None:
    """Send a notification but never let notification failures crash posting."""
    try:
        notify.send(message)
    except Exception as exc:
        print(f"Notification failed: {exc}", file=sys.stderr)


def require_kaggle_login_env() -> None:
    missing = []
    if not EMAIL:
        missing.append("KAGGLE_EMAIL")
    if not PASSWORD:
        missing.append("KAGGLE_PASSWORD")
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s) for Kaggle login: "
            + ", ".join(missing)
        )


def login(page) -> None:
    page.goto("https://www.kaggle.com/account/login", wait_until="networkidle")
    if is_browser_challenge(page):
        raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)
    page.fill('input[name="email"]', EMAIL)
    page.fill('input[name="password"]', PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_url("https://www.kaggle.com/", timeout=20000)


def post_discussion(page, forum_url: str, title: str, body: str) -> str:
    page.goto(forum_url, wait_until="networkidle")
    if is_browser_challenge(page):
        raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)
    page.click("text=New Topic", timeout=10000)
    page.wait_for_selector('input[name="title"]', timeout=10000)
    page.fill('input[name="title"]', title)
    editor = page.locator('[contenteditable="true"]').first
    editor.click()
    editor.fill(body)
    page.click('button:has-text("Post")', timeout=10000)
    page.wait_for_load_state("networkidle", timeout=20000)
    return page.url


def load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Queue not found: {QUEUE_PATH}")
    try:
        payload = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Queue read failed: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Queue payload must be a list: {QUEUE_PATH}")
    return payload


def select_smoke_item(queue: list[dict], now: datetime) -> dict | None:
    item = dq.select_next_post(queue, now=now)
    if item is not None:
        return item
    for candidate in queue:
        if dq.normalize_status(candidate.get("status")) in dq.POSTABLE_STATUSES:
            return candidate
    return None


def load_item_body(item: dict) -> str:
    required_keys = ("id", "title", "forum_url", "body_file", "body_section")
    missing_keys = [key for key in required_keys if not item.get(key)]
    if missing_keys:
        raise ValueError(
            f"Queue item missing required key(s): {', '.join(missing_keys)}"
        )
    body_file = str(item["body_file"])
    candidates = [REPO / body_file]
    # Queues written before the layout reorg store a bare filename, which
    # resolves to the repo root where the drafts file no longer lives.
    if "/" not in body_file:
        candidates.append(REPO / "docs" / "discussions" / body_file)
    drafts_path = next((p for p in candidates if p.is_file()), candidates[0])
    try:
        return dq.extract_body(
            drafts_path.read_text(encoding="utf-8"),
            str(item["body_section"]),
            strict=True,
            strip_heading=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Cannot extract draft body: {exc}") from exc


def smoke_test(*, check_login: bool = False) -> int:
    queue = load_queue()
    now = datetime.now(tz=timezone.utc)
    item = select_smoke_item(queue, now=now)
    if item is None:
        print("No postable discussion items found in queue.")
        return 0

    body = load_item_body(item)
    print(f"Smoke candidate: {item['title']}")
    print(f"Forum: {item['forum_url']}")
    print(f"Body length: {len(body)} characters")

    if not check_login:
        print("Discussion smoke test passed (queue + body validation only).")
        return 0

    require_kaggle_login_env()
    sync_playwright = require_playwright()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            login(page)
        finally:
            browser.close()
    print("Discussion smoke test passed (login verified).")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post the next queued Kaggle discussion or run a smoke test."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate posting prerequisites without creating a post.",
    )
    parser.add_argument(
        "--check-login",
        action="store_true",
        help="With --smoke-test, open Playwright and verify Kaggle login without posting.",
    )
    args = parser.parse_args(argv)
    if args.check_login and not args.smoke_test:
        parser.error("--check-login requires --smoke-test")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args([] if argv is None else argv)

    if args.smoke_test:
        try:
            raise SystemExit(smoke_test(check_login=args.check_login))
        except (
            EnvironmentError,
            FileNotFoundError,
            ValueError,
            RuntimeError,
            Exception,
        ) as exc:
            print(str(exc), file=sys.stderr)
            notify_safe(f"❌ Discussion smoke test failed: {exc}")
            sys.exit(1)

    if not POSTING_ENABLED:
        # Deliberate, not a leftover: unifying the selector changed which drafts
        # are eligible, and the queue holds drafts whose measured claims have not
        # been checked against the tracker. Publishing is the one irreversible
        # effect in this system, so it is enabled by a decision, not by a merge.
        # Set DISCUSSION_POSTING_ENABLED=1 to turn it on.
        print(
            "Posting is disabled (DISCUSSION_POSTING_ENABLED is not set). "
            "See docs/adr/0004-posting-stays-off-by-default.md",
            file=sys.stderr,
        )
        return

    try:
        require_kaggle_login_env()
    except EnvironmentError as exc:
        print(str(exc), file=sys.stderr)
        notify_safe(f"❌ Discussion post skipped: {exc}")
        sys.exit(1)

    try:
        queue = load_queue()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        notify_safe(f"❌ {exc}")
        sys.exit(1)
    now = datetime.now(tz=timezone.utc)
    item = dq.select_next_post(queue, now=now)

    if item is None:
        print("No pending posts ready.")
        return

    try:
        body = load_item_body(item)
    except ValueError as exc:
        notify_safe(f"❌ {exc}")
        sys.exit(1)

    print(f"Posting: {item['title']}")
    try:
        sync_playwright = require_playwright()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            login(page)
            post_url = post_discussion(page, item["forum_url"], item["title"], body)
            browser.close()
    except Exception as e:
        notify_safe(f"❌ Post failed: {item['title']}\n{e}")
        sys.exit(1)

    dq.mark_posted(QUEUE_PATH, item["id"], post_url=post_url)

    updated = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    next_item = dq.select_next_post(updated, now=now)
    next_info = (
        f"Next: {next_item['title']} ({next_item['scheduled_after'][:10]})"
        if next_item
        else "Queue empty."
    )

    notify_safe(f'✅ *Discussion posted*\n"{item["title"]}"\n{post_url}\n\n{next_info}')
    print(f"Posted: {post_url}")


if __name__ == "__main__":
    main(sys.argv[1:])
