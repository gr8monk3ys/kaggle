#!/usr/bin/env python3
"""Shared Kaggle Playwright browser infrastructure.

Extracts common login, locator helpers, anti-bot delays, argparse flags, and
tracker persistence from dataset_metadata_sync.py / campaign_execute.py so all
social-engagement scripts share one code path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_STATE = REPO_ROOT / "pi-automation" / "data" / "kaggle_storage_state.json"
DEFAULT_TIMEOUT_MS = 20_000


# ---------------------------------------------------------------------------
# Playwright import guard
# ---------------------------------------------------------------------------

def require_playwright():
    """Import and return (sync_playwright, PlaywrightTimeout) or exit."""
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError as exc:
        raise SystemExit(
            "playwright is not installed. Run:\n"
            "  pip install -r pi-automation/scripts/requirements.txt\n"
            "  python -m playwright install chromium"
        ) from exc
    return sync_playwright, PlaywrightTimeout


# ---------------------------------------------------------------------------
# Locator helpers
# ---------------------------------------------------------------------------

def locator_count(locator) -> int:
    """Safe .count() that returns 0 on any exception."""
    try:
        return locator.count()
    except Exception:
        return 0


def first_available(*locators):
    """Return the first locator with count > 0, or None."""
    for locator in locators:
        if locator is not None and locator_count(locator):
            return locator
    return None


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def is_login_prompt_visible(page) -> bool:
    login_markers = (
        page.get_by_role("link", name=re.compile(r"^sign in$", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"^sign in$", re.IGNORECASE)).first,
        page.get_by_role("link", name=re.compile(r"^register$", re.IGNORECASE)).first,
        page.locator('a[href*="/account/login"]').first,
    )
    return any(locator_count(marker) for marker in login_markers)


def is_authenticated(page) -> bool:
    url = str(getattr(page, "url", "") or "").lower()
    if "/account/login" in url:
        return False
    return not is_login_prompt_visible(page)


def _wait_and_check_auth(page, *, timeout_ms: int) -> bool:
    """Wait for the page to settle, then check auth robustly.

    Kaggle is a React SPA — sign-in buttons may not appear immediately.
    We poll up to 5s to let the page fully hydrate before deciding.
    """
    import time
    deadline = time.time() + min(timeout_ms, 5000) / 1000.0
    while time.time() < deadline:
        # If sign-in buttons appear, definitely not authenticated
        if is_login_prompt_visible(page):
            return False
        # If a user avatar/profile button appears, definitely authenticated
        avatar = first_available(
            page.locator('button img[src*="gravatar"], button img[src*="kaggle"]').first,
            page.get_by_role("button", name=re.compile(r"(profile|account)", re.IGNORECASE)).first,
        )
        if avatar is not None:
            return True
        page.wait_for_timeout(500)
    # Fallback to basic check
    return is_authenticated(page)


def maybe_login(
    page,
    *,
    email: str,
    password: str,
    manual_login: bool = False,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> None:
    """Authenticate on Kaggle using credentials or manual browser login."""
    page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(1500)
    if _wait_and_check_auth(page, timeout_ms=timeout_ms):
        return

    page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(1000)

    # Kaggle uses a two-step login: click "Sign in with Email" first to reveal fields
    email_signin_btn = first_available(
        page.get_by_role("button", name=re.compile(r"sign in with email", re.IGNORECASE)).first,
    )
    if email_signin_btn is not None:
        email_signin_btn.click(timeout=timeout_ms)
        page.wait_for_timeout(800)

    email_input = first_available(
        page.locator('input[name="email"]').first,
        page.locator('input[type="email"]').first,
        page.get_by_role("textbox", name=re.compile(r"email", re.IGNORECASE)).first,
    )
    password_input = first_available(
        page.locator('input[name="password"]').first,
        page.locator('input[type="password"]').first,
        page.get_by_role("textbox", name=re.compile(r"password", re.IGNORECASE)).first,
    )

    if email and password and email_input is not None and password_input is not None:
        email_input.fill(email, timeout=timeout_ms)
        password_input.fill(password, timeout=timeout_ms)
        submit_button = first_available(
            page.locator('button[type="submit"]').first,
            page.get_by_role("button", name=re.compile(r"sign in|log in", re.IGNORECASE)).first,
        )
        if submit_button is not None:
            submit_button.click(timeout=timeout_ms)
        else:
            page.keyboard.press("Enter")
        page.wait_for_timeout(2000)
        page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(1500)
        if _wait_and_check_auth(page, timeout_ms=timeout_ms):
            return

    if manual_login:
        page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded", timeout=timeout_ms)
        print("Manual login required: complete Kaggle login in the opened browser window.")
        input("Press Enter after login completes...")
        page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(1500)
        if _wait_and_check_auth(page, timeout_ms=timeout_ms):
            return
        raise RuntimeError("Kaggle login still appears unauthenticated after manual login.")

    raise RuntimeError(
        "Kaggle login required but session appears signed out. "
        "Provide KAGGLE_EMAIL/KAGGLE_PASSWORD or run with --manual-login (headed)."
    )


# ---------------------------------------------------------------------------
# Anti-bot delay
# ---------------------------------------------------------------------------

def human_delay(base: float = 2.0, jitter: float = 1.5) -> None:
    """Sleep for a randomized duration to mimic human pacing."""
    time.sleep(base + random.uniform(0.0, jitter))


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

def add_common_browser_args(parser: argparse.ArgumentParser) -> None:
    """Add shared --headed, --dry-run, --storage-state, --timeout-ms, creds."""
    parser.add_argument("--headed", action="store_true", help="Run browser headed (visible).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without acting.")
    parser.add_argument(
        "--storage-state", type=Path, default=DEFAULT_STORAGE_STATE,
        help="Playwright storage state JSON path.",
    )
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="Playwright timeout in ms.")
    parser.add_argument(
        "--manual-login", action="store_true", default=False,
        help="Allow interactive login if session is unauthenticated.",
    )
    parser.add_argument("--email", default=os.environ.get("KAGGLE_EMAIL", ""), help="Kaggle login email.")
    parser.add_argument("--password", default=os.environ.get("KAGGLE_PASSWORD", ""), help="Kaggle login password.")


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
