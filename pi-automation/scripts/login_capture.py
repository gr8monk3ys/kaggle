#!/usr/bin/env python3
"""Capture a Kaggle browser session into the shared Playwright storage state.

Opens Chromium (headed with --headed), authenticates via KAGGLE_EMAIL/
KAGGLE_PASSWORD or an interactive --manual-login, and persists the session
to the storage-state JSON that discussion_post.py, upvote_content.py,
comment_thread.py, and campaign-execute all reuse.

Typical one-time setup:
    python pi-automation/scripts/login_capture.py --manual-login --headed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_browser as kb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Kaggle login into Playwright storage state.")
    kb.add_common_browser_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(f"[dry-run] Would authenticate and save storage state to {args.storage_state}")
        return 0

    try:
        with kb.open_kaggle_browser(args):
            pass
    except RuntimeError as exc:
        print(f"[failed] {exc}")
        return 1

    print(f"[done] Kaggle session saved to {args.storage_state}")
    print("Verify with: ./manage.sh smoke-live --check-discussion-login")
    return 0


if __name__ == "__main__":
    sys.exit(main())
