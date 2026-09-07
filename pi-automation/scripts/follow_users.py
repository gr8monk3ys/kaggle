#!/usr/bin/env python3
"""Follow Kaggle users via Playwright to build visibility.

Reads target usernames from follow_targets.json and/or CLI args, visits each
profile, clicks Follow, and tracks completions to avoid double-following.

Rate limit: ~8/min (7.5s base + jitter between follows).
Default limit: 10 per session.
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
TRACKER_PATH = REPO_ROOT / "pi-automation" / "data" / "follow_tracker.json"
TARGETS_PATH = REPO_ROOT / "pi-automation" / "data" / "follow_targets.json"


def load_targets(targets_path: Path) -> list[str]:
    """Load target usernames from JSON file. Format: {"users": ["user1", ...]}"""
    if not targets_path.exists():
        return []
    try:
        data = json.loads(targets_path.read_text(encoding="utf-8"))
        users = data.get("users", [])
        if isinstance(users, list):
            return [str(u).strip() for u in users if str(u).strip()]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def follow_user(page, username: str, *, timeout_ms: int) -> str:
    """Navigate to user profile and click Follow. Returns result message."""
    profile_url = f"https://www.kaggle.com/{username}"
    page.goto(profile_url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(1000)

    if not kb.is_authenticated(page):
        raise RuntimeError("Not authenticated")

    # Check if already following
    following_btn = kb.first_available(
        page.get_by_role("button", name=re.compile(r"^following$", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"^unfollow$", re.IGNORECASE)).first,
    )
    if following_btn is not None:
        return f"already following {username}"

    # Find and click Follow button
    follow_btn = kb.first_available(
        page.get_by_role("button", name=re.compile(r"^follow$", re.IGNORECASE)).first,
    )
    if follow_btn is None:
        raise RuntimeError(f"Follow button not found on {profile_url}")

    follow_btn.click(timeout=timeout_ms)
    page.wait_for_timeout(1000)
    return f"followed {username}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow Kaggle users to build visibility.")
    kb.add_common_browser_args(parser)
    parser.add_argument("--limit", type=int, default=10, help="Max follows per session (default 10).")
    parser.add_argument("--users", nargs="*", default=[], help="Usernames to follow (in addition to targets file).")
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH, help="Follow targets JSON path.")
    parser.add_argument("--tracker", type=Path, default=TRACKER_PATH, help="Tracker JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        print("--limit must be >= 1")
        return 1

    # Merge targets from file and CLI
    targets = load_targets(args.targets)
    for user in args.users:
        if user not in targets:
            targets.append(user)

    if not targets:
        print("No follow targets. Add usernames to follow_targets.json or use --users.")
        return 0

    tracker = kb.TrackerFile(args.tracker)
    pending = [u for u in targets if not tracker.has(u)]
    pending = pending[:args.limit]

    if not pending:
        print(f"All {len(targets)} targets already followed.")
        return 0

    print(f"Will follow {len(pending)} users (of {len(targets)} total targets):")
    for u in pending:
        print(f"  {u}")

    if args.dry_run:
        print("[dry-run] No follows performed.")
        return 0

    success = 0
    failures = 0
    with kb.open_kaggle_browser(args) as page:
        for idx, username in enumerate(pending):
            try:
                result = follow_user(page, username, timeout_ms=args.timeout_ms)
                tracker.mark(username, result)
                tracker.save()
                success += 1
                print(f"[done] {result}")
            except Exception as exc:
                failures += 1
                print(f"[failed] {username}: {exc}")

            if idx < len(pending) - 1:
                kb.human_delay(base=7.5, jitter=3.0)

    print(f"Follow summary: success={success} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
