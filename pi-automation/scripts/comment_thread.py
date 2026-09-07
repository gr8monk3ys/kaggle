#!/usr/bin/env python3
"""Post comments on Kaggle discussion threads via Playwright.

Reads pre-written comments from a queue file or CLI args, navigates to
threads, posts replies, and tracks to prevent duplicate comments.

Rate limit: ~3/min (20s base + jitter between comments).
Comments must be pre-written — this script does not generate content.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_browser as kb


REPO_ROOT = kb.REPO_ROOT
TRACKER_PATH = REPO_ROOT / "pi-automation" / "data" / "comment_tracker.json"
QUEUE_PATH = REPO_ROOT / "pi-automation" / "data" / "comment_queue.json"


def load_comment_queue(queue_path: Path) -> list[dict]:
    """Load comment queue. Format: {"comments": [{"url": "...", "body": "...", "id": "..."}, ...]}"""
    if not queue_path.exists():
        return []
    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        comments = data.get("comments", [])
        if isinstance(comments, list):
            return [c for c in comments if isinstance(c, dict) and c.get("url") and c.get("body")]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def comment_key(item: dict) -> str:
    """Generate a dedup key from comment item."""
    item_id = str(item.get("id", "")).strip()
    if item_id:
        return item_id
    # Fallback: url + first 50 chars of body
    url = str(item.get("url", "")).strip().lower()
    body_prefix = str(item.get("body", ""))[:50].strip().lower()
    return f"{url}|{body_prefix}"


def post_comment(page, url: str, body: str, *, timeout_ms: int) -> str:
    """Navigate to thread and post a comment. Returns result message."""
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(1500)

    if kb.is_browser_challenge(page):
        raise RuntimeError(kb.BROWSER_CHALLENGE_MESSAGE)

    if not kb.is_authenticated(page):
        raise RuntimeError("Not authenticated")

    # Find reply/comment input area
    reply_box = kb.first_available(
        page.get_by_role("textbox", name=re.compile(r"reply|comment|response", re.IGNORECASE)).first,
        page.locator('[contenteditable="true"]').first,
        page.locator('textarea[placeholder*="reply" i]').first,
        page.locator('textarea[placeholder*="comment" i]').first,
    )

    if reply_box is None:
        # Try clicking a "Reply" button first to reveal the input
        reply_trigger = kb.first_available(
            page.get_by_role("button", name=re.compile(r"^reply$", re.IGNORECASE)).first,
            page.get_by_role("button", name=re.compile(r"^add comment$", re.IGNORECASE)).first,
            page.get_by_role("button", name=re.compile(r"^comment$", re.IGNORECASE)).first,
        )
        if reply_trigger is not None:
            reply_trigger.click(timeout=timeout_ms)
            page.wait_for_timeout(800)
            reply_box = kb.first_available(
                page.get_by_role("textbox", name=re.compile(r"reply|comment|response", re.IGNORECASE)).first,
                page.locator('[contenteditable="true"]').first,
                page.locator('textarea[placeholder*="reply" i]').first,
                page.locator('textarea[placeholder*="comment" i]').first,
            )

    if reply_box is None:
        raise RuntimeError(f"Reply box not found on {url}")

    reply_box.click(timeout=timeout_ms)
    reply_box.fill(body, timeout=timeout_ms)
    page.wait_for_timeout(500)

    # Find and click submit/post button
    submit_btn = kb.first_available(
        page.get_by_role("button", name=re.compile(r"^post$", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"^submit$", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"^post comment$", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"^post reply$", re.IGNORECASE)).first,
    )
    if submit_btn is None:
        raise RuntimeError(f"Submit button not found on {url}")

    submit_btn.click(timeout=timeout_ms)
    page.wait_for_timeout(2000)
    return f"commented on {url}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post comments on Kaggle discussion threads.")
    kb.add_common_browser_args(parser)
    parser.add_argument("--url", default=None, help="Thread URL to comment on (with --body).")
    parser.add_argument("--body", default=None, help="Comment text (required with --url).")
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH, help="Comment queue JSON path.")
    parser.add_argument("--limit", type=int, default=3, help="Max comments per session (default 3).")
    parser.add_argument("--tracker", type=Path, default=TRACKER_PATH, help="Tracker JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        print("--limit must be >= 1")
        return 1

    # Collect comments from CLI and queue
    items: list[dict] = []
    if args.url and args.body:
        items.append({"url": args.url, "body": args.body, "id": f"cli-{args.url}"})
    elif args.url and not args.body:
        print("--body is required when using --url")
        return 1

    for item in load_comment_queue(args.queue):
        items.append(item)

    if not items:
        print("No comments to post. Use --url/--body or populate comment_queue.json.")
        return 0

    tracker = kb.TrackerFile(args.tracker)
    pending = [item for item in items if not tracker.has(comment_key(item))]
    pending = pending[:args.limit]

    if not pending:
        print(f"All {len(items)} comments already posted.")
        return 0

    print(f"Will post {len(pending)} comments (of {len(items)} total):")
    for item in pending:
        print(f"  {item['url']} -> {item['body'][:60]}...")

    if args.dry_run:
        print("[dry-run] No comments posted.")
        return 0

    success = 0
    failures = 0
    with kb.open_kaggle_browser(args) as page:
        for idx, item in enumerate(pending):
            try:
                result = post_comment(page, item["url"], item["body"], timeout_ms=args.timeout_ms)
                tracker.mark(comment_key(item), result)
                tracker.save()
                success += 1
                print(f"[done] {result}")
            except Exception as exc:
                failures += 1
                print(f"[failed] {item['url']}: {exc}")

            if idx < len(pending) - 1:
                kb.human_delay(base=20.0, jitter=8.0)

    print(f"Comment summary: success={success} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
