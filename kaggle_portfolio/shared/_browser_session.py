#!/usr/bin/env python3
"""Kaggle browser session: sign-in, storage state, and challenge handling.

Moved out of ``pi-automation/scripts/kaggle_browser.py``, which was the fullest
of FOUR copies of these helpers — ``dataset_metadata_sync``, ``campaign_execute``
and ``discussion_post`` each carried their own ``require_playwright``,
``maybe_login``, ``is_authenticated`` and friends, under the same names.

This is the session layer beneath :mod:`kaggle_portfolio.shared.kaggle_site`.
Callers should talk to that; this module exists so there is exactly one
implementation of "are we signed in, and how do we get there".

Playwright is imported lazily inside :func:`require_playwright` — importing this
module must not require a browser.
"""

from __future__ import annotations

import base64
import json
import random
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_STATE = (
    REPO_ROOT / "pi-automation" / "data" / "kaggle_storage_state.json"
)
DEFAULT_TIMEOUT_MS = 20_000
MANUAL_LOGIN_TIMEOUT_S = 900
BROWSER_CHALLENGE_MESSAGE = (
    "Kaggle browser challenge detected. Clear the Cloudflare/reCAPTCHA check in a headed browser "
    "and retry with --manual-login."
)


def require_playwright():
    """Import and return (sync_playwright, PlaywrightTimeout) or exit."""
    try:
        from playwright.sync_api import (
            sync_playwright,
            TimeoutError as PlaywrightTimeout,
        )
    except ImportError as exc:
        raise SystemExit(
            "playwright is not installed. Run:\n"
            "  pip install -r pi-automation/scripts/requirements.txt\n"
            "  python -m playwright install chromium"
        ) from exc
    return sync_playwright, PlaywrightTimeout


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


def wait_for_challenge_to_clear(page, *, timeout_s: int = 180) -> bool:
    """Poll until Kaggle's bot challenge is gone. Returns False on timeout.

    Polled rather than gated on input(): these scripts run from wrappers and
    shells with no TTY, where reading stdin raises EOFError immediately.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(3)
        try:
            if not is_browser_challenge(page):
                return True
        except Exception:
            return False
    return False


def _client_token_is_authenticated(context) -> bool:
    """True when Kaggle's CLIENT-TOKEN cookie carries a signed-in identity.

    Kaggle sets CLIENT-TOKEN for anonymous visitors too, so presence alone means
    nothing; the JWT payload only names a user once signed in.
    """
    try:
        cookies = context.cookies()
    except Exception:
        return False
    for cookie in cookies:
        if cookie.get("name") != "CLIENT-TOKEN":
            continue
        value = str(cookie.get("value") or "")
        parts = value.split(".")
        if len(parts) < 2:
            continue
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)  # restore base64url padding
        try:
            claims = json.loads(
                base64.urlsafe_b64decode(payload).decode("utf-8", "replace")
            )
        except Exception:
            continue
        if not isinstance(claims, dict):
            continue
        for key in ("displayName", "userName", "sub", "userId"):
            candidate = claims.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return True
            if isinstance(candidate, int) and candidate > 0:
                return True
    return False


def session_is_signed_in(context) -> bool:
    """Whether this browser context holds a signed-in Kaggle session.

    Checks every open page, not just the one login started on: Google/Facebook
    sign-in completes in a popup or second tab, so the original page can still
    show the login form while the context is already authenticated.
    """
    if _client_token_is_authenticated(context):
        return True
    try:
        pages = list(context.pages)
    except Exception:
        return False
    for candidate in pages:
        try:
            if "/account/login" in str(candidate.url or "").lower():
                continue
            if _wait_and_check_auth(candidate, timeout_ms=1500):
                return True
        except Exception:
            continue
    return False


def describe_context(context) -> str:
    """Human-readable state summary printed while polling for login.

    Reports which signal is missing so a stalled capture is diagnosable from the
    log alone. Cookie *names* only — values are credentials and are never
    printed.
    """
    parts: list[str] = []
    try:
        urls = [str(p.url or "") for p in context.pages]
        if urls:
            shown = ", ".join(u.split("?")[0][:60] for u in urls[:3])
            parts.append(f"{len(urls)} page(s): {shown}")
        else:
            parts.append("no open pages")
    except Exception:
        parts.append("pages unavailable")

    try:
        cookies = context.cookies()
        kaggle_names = sorted(
            {
                str(c.get("name"))
                for c in cookies
                if "kaggle" in str(c.get("domain", "")).lower()
            }
        )
        if "CLIENT-TOKEN" in kaggle_names:
            state = (
                "identifies a user"
                if _client_token_is_authenticated(context)
                else "anonymous"
            )
            parts.append(f"CLIENT-TOKEN present ({state})")
        else:
            parts.append(
                "no CLIENT-TOKEN; kaggle cookies: "
                + (", ".join(kaggle_names[:6]) or "none")
            )
    except Exception:
        parts.append("cookies unavailable")

    return " | ".join(parts)


def _wait_and_check_auth(page, *, timeout_ms: int) -> bool:
    """Wait for the page to settle, then check auth robustly.

    Kaggle is a React SPA — sign-in buttons may not appear immediately.
    We poll up to 5s to let the page fully hydrate before deciding.
    """
    import time

    deadline = time.time() + min(timeout_ms, 5000) / 1000.0
    while time.time() < deadline:
        if is_browser_challenge(page):
            return False
        # If sign-in buttons appear, definitely not authenticated
        if is_login_prompt_visible(page):
            return False
        # If a user avatar/profile button appears, definitely authenticated
        avatar = first_available(
            page.locator(
                'button img[src*="gravatar"], button img[src*="kaggle"]'
            ).first,
            page.get_by_role(
                "button", name=re.compile(r"(profile|account)", re.IGNORECASE)
            ).first,
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
    page.goto(
        "https://www.kaggle.com/datasets",
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )
    page.wait_for_timeout(1500)
    if is_browser_challenge(page):
        if manual_login:
            print("Kaggle browser challenge detected. Clear it in the browser window.")
            print("Waiting for the challenge to clear (no keypress needed)...")
            if not wait_for_challenge_to_clear(page):
                raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)
            page.goto(
                "https://www.kaggle.com/datasets",
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            page.wait_for_timeout(1500)
        else:
            raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)
    if _wait_and_check_auth(page, timeout_ms=timeout_ms):
        return

    page.goto(
        "https://www.kaggle.com/account/login",
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )
    page.wait_for_timeout(1000)
    if is_browser_challenge(page):
        if manual_login:
            print("Kaggle browser challenge detected. Clear it in the browser window.")
            print("Waiting for the challenge to clear (no keypress needed)...")
            if not wait_for_challenge_to_clear(page):
                raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)
            page.goto(
                "https://www.kaggle.com/account/login",
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            page.wait_for_timeout(1000)
        else:
            raise RuntimeError(BROWSER_CHALLENGE_MESSAGE)

    # Kaggle uses a two-step login: click "Sign in with Email" first to reveal fields
    email_signin_btn = first_available(
        page.get_by_role(
            "button", name=re.compile(r"sign in with email", re.IGNORECASE)
        ).first,
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
            page.get_by_role(
                "button", name=re.compile(r"sign in|log in", re.IGNORECASE)
            ).first,
        )
        if submit_button is not None:
            submit_button.click(timeout=timeout_ms)
        else:
            page.keyboard.press("Enter")
        page.wait_for_timeout(2000)
        page.goto(
            "https://www.kaggle.com/datasets",
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        page.wait_for_timeout(1500)
        if _wait_and_check_auth(page, timeout_ms=timeout_ms):
            return

    if manual_login:
        page.goto(
            "https://www.kaggle.com/account/login",
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        print(
            "Manual login required: complete Kaggle login in the opened browser window."
        )
        print(
            f"Waiting up to {MANUAL_LOGIN_TIMEOUT_S // 60} minutes for login to complete "
            "(no keypress needed; polling for the signed-in state)."
        )
        # Polled rather than gated on input(): this runs from wrappers and shells
        # with no TTY attached, where reading stdin raises EOFError immediately
        # and the capture fails before the user can even log in.
        deadline = time.time() + MANUAL_LOGIN_TIMEOUT_S
        last_report = 0.0
        while time.time() < deadline:
            time.sleep(3)
            if session_is_signed_in(page.context):
                print("Login detected; capturing session.")
                return
            now = time.time()
            if now - last_report >= 30:
                last_report = now
                remaining = int(deadline - now)
                print(
                    f"  still waiting ({remaining}s left) — {describe_context(page.context)}"
                )
        raise RuntimeError(
            f"Timed out after {MANUAL_LOGIN_TIMEOUT_S}s waiting for manual Kaggle login. "
            "Sign in inside the 'Chrome for Testing' window this script opened, not your "
            "regular browser."
        )

    raise RuntimeError(
        "Kaggle login required but session appears signed out. "
        "Provide KAGGLE_EMAIL/KAGGLE_PASSWORD or run with --manual-login (headed)."
    )


def human_delay(base: float = 2.0, jitter: float = 1.5) -> None:
    """Sleep for a randomized duration to mimic human pacing."""
    time.sleep(base + random.uniform(0.0, jitter))
