#!/usr/bin/env python3
"""Upvote Kaggle content (notebooks, datasets, discussions) via Playwright.

Accepts URLs or slugs from CLI or a queue file, navigates to each page,
clicks the upvote button, and tracks to prevent double-upvoting.

Rate limit: ~4/min (15s base + jitter between upvotes).
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
TRACKER_PATH = REPO_ROOT / "pi-automation" / "data" / "upvote_tracker.json"
QUEUE_PATH = REPO_ROOT / "pi-automation" / "data" / "upvote_queue.json"


def normalize_url(url_or_slug: str, content_type: str | None = None) -> str:
    """Convert a slug or URL to a full Kaggle URL."""
    url = url_or_slug.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    # Treat as slug: owner/name
    if content_type == "dataset":
        return f"https://www.kaggle.com/datasets/{url}"
    if content_type == "notebook":
        return f"https://www.kaggle.com/code/{url}"
    if content_type == "discussion":
        return f"https://www.kaggle.com/discussions/{url}"
    # Guess from slug structure
    return f"https://www.kaggle.com/{url}"


def tracker_key(url: str) -> str:
    """Extract a stable key from a URL for dedup tracking."""
    # Strip protocol and trailing slash
    return re.sub(r"^https?://", "", url).rstrip("/").lower()


def load_queue(queue_path: Path) -> list[dict]:
    """Load upvote queue. Format: {"items": [{"url": "...", "type": "notebook"}, ...]}"""
    if not queue_path.exists():
        return []
    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict) and i.get("url")]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def upvote_page(page, url: str, *, timeout_ms: int) -> str:
    """Navigate to content page and click upvote. Returns result message."""
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(1500)

    if not kb.is_authenticated(page):
        raise RuntimeError("Not authenticated")

    # Look for upvote button — Kaggle uses various aria labels
    upvote_btn = kb.first_available(
        page.get_by_role("button", name=re.compile(r"upvote", re.IGNORECASE)).first,
        page.locator('button[aria-label*="upvote" i]').first,
        page.locator('button[data-testid="upvote"]').first,
        # Thumbs up / vote icon buttons
        page.locator('button[aria-label*="vote" i]').first,
    )
    if upvote_btn is None:
        raise RuntimeError(f"Upvote button not found on {url}")

    # Check if already upvoted (button often has active/pressed state)
    aria_pressed = upvote_btn.get_attribute("aria-pressed")
    if aria_pressed == "true":
        return f"already upvoted {url}"

    upvote_btn.click(timeout=timeout_ms)
    page.wait_for_timeout(1000)
    return f"upvoted {url}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upvote Kaggle content to build goodwill.")
    kb.add_common_browser_args(parser)
    parser.add_argument("--url", nargs="*", default=[], help="URLs or slugs to upvote.")
    parser.add_argument("--type", choices=["notebook", "dataset", "discussion"], default=None,
                        help="Content type hint for slug resolution.")
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH, help="Upvote queue JSON path.")
    parser.add_argument("--limit", type=int, default=5, help="Max upvotes per session (default 5).")
    parser.add_argument("--tracker", type=Path, default=TRACKER_PATH, help="Tracker JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        print("--limit must be >= 1")
        return 1

    # Collect URLs from CLI and queue
    urls: list[str] = []
    for u in args.url:
        urls.append(normalize_url(u, args.type))
    for item in load_queue(args.queue):
        url = normalize_url(item["url"], item.get("type"))
        if url not in urls:
            urls.append(url)

    if not urls:
        print("No URLs to upvote. Use --url or populate upvote_queue.json.")
        return 0

    tracker = kb.TrackerFile(args.tracker)
    pending = [u for u in urls if not tracker.has(tracker_key(u))]
    pending = pending[:args.limit]

    if not pending:
        print(f"All {len(urls)} items already upvoted.")
        return 0

    print(f"Will upvote {len(pending)} items (of {len(urls)} total):")
    for u in pending:
        print(f"  {u}")

    if args.dry_run:
        print("[dry-run] No upvotes performed.")
        return 0

    success = 0
    failures = 0
    with kb.open_kaggle_browser(args) as page:
        for idx, url in enumerate(pending):
            try:
                result = upvote_page(page, url, timeout_ms=args.timeout_ms)
                tracker.mark(tracker_key(url), result)
                tracker.save()
                success += 1
                print(f"[done] {result}")
            except Exception as exc:
                failures += 1
                print(f"[failed] {url}: {exc}")

            if idx < len(pending) - 1:
                kb.human_delay(base=15.0, jitter=5.0)

    print(f"Upvote summary: success={success} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
