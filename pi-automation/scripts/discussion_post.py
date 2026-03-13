"""Post the next queued discussion to Kaggle using Playwright."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import notify
import discussion_queue as dq

REPO = Path(os.environ.get("REPO_PATH", str(Path(__file__).parent.parent.parent)))
QUEUE_PATH = Path(os.environ.get("QUEUE_PATH", str(Path(__file__).parent.parent / "data" / "discussion_queue.json")))
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


def is_browser_challenge(page) -> bool:
    try:
        title = str(page.title() or "").lower()
    except Exception:
        title = ""
    if "checking your browser" in title or "recaptcha" in title:
        return True
    try:
        body = str(page.locator("body").inner_text(timeout=1500) or "").lower()
    except Exception:
        body = ""
    return (
        "checking your browser before accessing" in body
        or "click here if you are not automatically redirected" in body
    )


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
    page.locator('input[name="email"]').fill(EMAIL)
    page.locator('input[name="password"]').fill(PASSWORD)
    page.locator('button[type="submit"]').click()
    page.wait_for_url("https://www.kaggle.com/", timeout=20000)


def post_discussion(page, forum_url: str, title: str, body: str) -> str:
    page.goto(forum_url, wait_until="networkidle")
    if is_browser_challenge(page):
        raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)
    page.get_by_text("New Topic", exact=True).click(timeout=10000)
    page.locator('input[name="title"]').wait_for(timeout=10000)
    page.locator('input[name="title"]').fill(title)
    editor = page.locator('[contenteditable="true"]').first
    editor.click()
    editor.fill(body)
    page.get_by_role("button", name="Post").click(timeout=10000)
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
    item = dq.next_pending(queue, now=now)
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
        raise ValueError(f"Queue item missing required key(s): {', '.join(missing_keys)}")
    drafts_path = REPO / str(item["body_file"])
    try:
        return dq.extract_body(drafts_path.read_text(encoding="utf-8"), str(item["body_section"]))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Cannot extract draft body: {exc}") from exc


def require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed. Run:\n"
            "  pip install -r pi-automation/scripts/requirements.txt\n"
            "  python -m playwright install chromium"
        ) from exc
    return sync_playwright


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
    parser = argparse.ArgumentParser(description="Post the next queued Kaggle discussion or run a smoke test.")
    parser.add_argument("--smoke-test", action="store_true", help="Validate posting prerequisites without creating a post.")
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
        except (EnvironmentError, FileNotFoundError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            notify_safe(f"❌ Discussion smoke test failed: {exc}")
            sys.exit(1)

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
    item = dq.next_pending(queue, now=now)

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
    next_item = dq.next_pending(updated, now=now)
    next_info = (
        f"Next: {next_item['title']} ({next_item['scheduled_after'][:10]})"
        if next_item
        else "Queue empty."
    )

    notify_safe(
        f"✅ *Discussion posted*\n"
        f"\"{item['title']}\"\n"
        f"{post_url}\n\n"
        f"{next_info}"
    )
    print(f"Posted: {post_url}")


if __name__ == "__main__":
    main(sys.argv[1:])
