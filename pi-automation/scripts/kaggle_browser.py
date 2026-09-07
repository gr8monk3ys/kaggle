#!/usr/bin/env python3
"""Shared Kaggle Playwright browser infrastructure.

Extracts common login, locator helpers, anti-bot delays, argparse flags, and
tracker persistence used by dataset metadata sync and campaign execution so all
social-engagement scripts share one code path.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

# The repo root must be importable before the shared session module is. Default
# to /repo, not a __file__-relative walk: in the container this file lives under
# /scripts, so parents[2] resolves to "/".
import os as _os
import sys as _sys
from pathlib import Path as _Path

_repo = _Path(_os.environ.get("REPO_PATH", "/repo"))
if not (_repo / "kaggle_portfolio").is_dir():
    _repo = _Path(__file__).resolve().parents[2]
if str(_repo) not in _sys.path:
    _sys.path.insert(0, str(_repo))

from kaggle_portfolio.shared._browser_session import (  # noqa: E402,F401
    _client_token_is_authenticated,
    _wait_and_check_auth,
    describe_context,
    first_available,
    human_delay,
    is_authenticated,
    is_browser_challenge,
    is_login_prompt_visible,
    locator_count,
    maybe_login,
    require_playwright,
    session_is_signed_in,
    wait_for_challenge_to_clear,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_STATE = (
    REPO_ROOT / "pi-automation" / "data" / "kaggle_storage_state.json"
)
DEFAULT_TIMEOUT_MS = 20_000
# How long --manual-login waits for a human to finish signing in. Generous
# because it covers finding the window, OAuth redirects, and 2FA.
MANUAL_LOGIN_TIMEOUT_S = 900
BROWSER_CHALLENGE_MESSAGE = (
    "Kaggle browser challenge detected. Clear the Cloudflare/reCAPTCHA check in a headed browser "
    "and retry with --manual-login."
)


# ---------------------------------------------------------------------------
# Playwright import guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Locator helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anti-bot delay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def add_common_browser_args(parser: argparse.ArgumentParser) -> None:
    """Add shared --headed, --dry-run, --storage-state, --timeout-ms, creds."""
    parser.add_argument(
        "--headed", action="store_true", help="Run browser headed (visible)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without acting."
    )
    parser.add_argument(
        "--storage-state",
        type=Path,
        default=DEFAULT_STORAGE_STATE,
        help="Playwright storage state JSON path.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="Playwright timeout in ms.",
    )
    parser.add_argument(
        "--manual-login",
        action="store_true",
        default=False,
        help="Allow interactive login if session is unauthenticated.",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("KAGGLE_EMAIL", ""),
        help="Kaggle login email.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("KAGGLE_PASSWORD", ""),
        help="Kaggle login password.",
    )


# ---------------------------------------------------------------------------
# Browser context manager
# ---------------------------------------------------------------------------


@contextmanager
def open_kaggle_browser(
    args: argparse.Namespace,
) -> Generator[Any, None, None]:
    """Launch Chromium, authenticate, yield the page, persist storage state."""
    sync_playwright, _PwTimeout = require_playwright()
    with sync_playwright() as pw:
        storage_path = getattr(args, "storage_state", DEFAULT_STORAGE_STATE)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        state_arg = str(storage_path) if storage_path.exists() else None

        browser = pw.chromium.launch(headless=not getattr(args, "headed", False))
        context = browser.new_context(storage_state=state_arg)
        page = context.new_page()
        try:
            maybe_login(
                page,
                email=getattr(args, "email", ""),
                password=getattr(args, "password", ""),
                manual_login=getattr(args, "manual_login", False),
                timeout_ms=getattr(args, "timeout_ms", DEFAULT_TIMEOUT_MS),
            )
            context.storage_state(path=str(storage_path))
            yield page
        finally:
            browser.close()


# ---------------------------------------------------------------------------
# JSON-backed dedup tracker
# ---------------------------------------------------------------------------


class TrackerFile:
    """Simple JSON-backed tracker with has/mark pattern for dedup."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"completed": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n", encoding="utf-8")

    def has(self, key: str) -> bool:
        return key in self._data.get("completed", {})

    def mark(self, key: str, detail: str = "") -> None:
        if "completed" not in self._data:
            self._data["completed"] = {}
        from datetime import datetime, timezone

        self._data["completed"][key] = {
            "at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "detail": detail,
        }

    @property
    def completed(self) -> dict[str, Any]:
        return self._data.get("completed", {})
