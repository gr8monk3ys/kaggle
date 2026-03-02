#!/usr/bin/env python3
"""Execute campaign queue actions against Kaggle discussions/changelog topics.

This command processes due actions from `promotion_campaign_queue.json` and
posts supported channels (`kaggle-discussion`, `kaggle-changelog`) to the
dataset discussion board. Queue state is updated in-place:
  - planned -> in_progress (claimed) before execution
  - in_progress/planned -> done on successful post
  - stays in_progress on failure with `last_error`
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_QUEUE_PATH = Path("pi-automation") / "data" / "promotion_campaign_queue.json"
DEFAULT_STORAGE_STATE = Path("pi-automation") / "data" / "kaggle_storage_state.json"
SUPPORTED_CHANNELS = {"kaggle-discussion", "kaggle-changelog"}
PLANNED = "planned"
IN_PROGRESS = "in_progress"
DONE = "done"


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat().replace("+00:00", "Z")


def parse_iso_utc(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Campaign queue not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid queue payload: {path}")
    queue = payload.get("queue")
    if not isinstance(queue, list):
        raise SystemExit(f"Queue payload missing list: {path}")
    return payload


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def normalized_status(action: dict[str, Any]) -> str:
    return str(action.get("status", PLANNED)).strip().lower() or PLANNED


def normalized_channel(action: dict[str, Any]) -> str:
    return str(action.get("channel", "")).strip().lower()


def action_sort_key(action: dict[str, Any]) -> tuple[str, str]:
    return (
        str(action.get("scheduled_for", "")),
        str(action.get("id", "")),
    )


def due_supported_actions(
    queue: list[dict[str, Any]],
    *,
    now: datetime,
    limit: int,
    allowed_channels: set[str] | None,
    include_planned: bool,
    include_in_progress: bool,
    respect_schedule: bool,
) -> list[dict[str, Any]]:
    statuses: set[str] = set()
    if include_in_progress:
        statuses.add(IN_PROGRESS)
    if include_planned:
        statuses.add(PLANNED)

    filtered: list[dict[str, Any]] = []
    for action in sorted(queue, key=action_sort_key):
        channel = normalized_channel(action)
        if channel not in SUPPORTED_CHANNELS:
            continue
        if allowed_channels and channel not in allowed_channels:
            continue
        if normalized_status(action) not in statuses:
            continue
        if respect_schedule:
            scheduled_for = parse_iso_utc(action.get("scheduled_for"))
            if scheduled_for is not None and scheduled_for > now:
                continue
        filtered.append(action)

    # Prioritize already-claimed actions, then by schedule/id.
    filtered.sort(
        key=lambda action: (
            0 if normalized_status(action) == IN_PROGRESS else 1,
            *action_sort_key(action),
        )
    )
    if limit > 0:
        return filtered[:limit]
    return filtered


def claim_action(action: dict[str, Any], stamp: str) -> None:
    if normalized_status(action) == PLANNED:
        action["status"] = IN_PROGRESS
        action["claimed_at"] = stamp
        action["claim_count"] = int(action.get("claim_count") or 0) + 1


def mark_done(action: dict[str, Any], post_url: str, stamp: str) -> None:
    action["status"] = DONE
    action["completed_at"] = stamp
    action["note"] = f"posted: {post_url}"
    action.pop("last_error", None)


def mark_error(action: dict[str, Any], error_text: str) -> None:
    action["status"] = IN_PROGRESS
    action["last_error"] = error_text[:600]
    action["fail_count"] = int(action.get("fail_count") or 0) + 1


def topic_title_for_action(action: dict[str, Any]) -> str:
    title = str(action.get("dataset_title") or action.get("dataset_ref") or "Dataset").strip()
    channel = normalized_channel(action)
    if channel == "kaggle-changelog":
        return f"Changelog: {title}"
    return f"Usability Update: {title}"


def require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "playwright is not installed. Run:\n"
            "  pip install -r pi-automation/scripts/requirements.txt\n"
            "  python -m playwright install chromium"
        ) from exc
    return sync_playwright


def locator_count(locator) -> int:
    try:
        return locator.count()
    except Exception:
        return 0


def is_authenticated(page) -> bool:
    if "/account/login" in str(getattr(page, "url", "")).lower():
        return False
    sign_in = page.get_by_role("button", name=re.compile(r"^sign in$", re.IGNORECASE)).first
    sign_in_link = page.get_by_role("link", name=re.compile(r"^sign in$", re.IGNORECASE)).first
    return not (locator_count(sign_in) or locator_count(sign_in_link))


def maybe_login(page, *, timeout_ms: int, manual_login: bool, email: str, password: str) -> None:
    page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(400)
    if is_authenticated(page):
        return

    page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(400)
    if is_authenticated(page):
        return

    email = email.strip()
    password = password.strip()
    email_input = page.locator('input[name="email"]').first
    password_input = page.locator('input[name="password"]').first
    if email and password and locator_count(email_input) and locator_count(password_input):
        email_input.fill(email, timeout=timeout_ms)
        password_input.fill(password, timeout=timeout_ms)
        submit = page.locator('button[type="submit"]').first
        if locator_count(submit):
            submit.click(timeout=timeout_ms)
            page.wait_for_timeout(1500)
            if is_authenticated(page):
                return

    if manual_login:
        print("Manual Kaggle login required in browser window.")
        input("Press Enter after completing login...")
        page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(500)
        if is_authenticated(page):
            return

    raise RuntimeError("Kaggle authentication required. Provide credentials or use --manual-login.")


def post_dataset_discussion_topic(
    page,
    *,
    dataset_ref: str,
    topic_title: str,
    body: str,
    timeout_ms: int,
) -> str:
    discussion_url = f"https://www.kaggle.com/datasets/{dataset_ref}/discussion"
    page.goto(discussion_url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(500)

    cookie_ack = page.get_by_text("OK, Got it.", exact=True).first
    if locator_count(cookie_ack):
        cookie_ack.click(timeout=timeout_ms)
        page.wait_for_timeout(250)

    new_topic = page.get_by_role("button", name=re.compile(r"new\s+topic", re.IGNORECASE)).first
    if not locator_count(new_topic):
        raise RuntimeError("New Topic button not found")
    if new_topic.is_disabled():
        raise RuntimeError("New Topic button is disabled (insufficient permissions or auth state)")
    new_topic.click(timeout=timeout_ms)
    page.wait_for_timeout(400)

    title_box = page.get_by_role("textbox", name=re.compile(r"topic\s+title", re.IGNORECASE)).first
    content_box = page.get_by_role("textbox", name=re.compile(r"content", re.IGNORECASE)).first
    publish = page.get_by_role("button", name=re.compile(r"publish\s+topic|post", re.IGNORECASE)).first
    if not locator_count(title_box) or not locator_count(content_box) or not locator_count(publish):
        raise RuntimeError("Discussion editor controls not found")

    title_box.fill(topic_title, timeout=timeout_ms)
    content_box.fill(body, timeout=timeout_ms)
    publish.click(timeout=timeout_ms)

    deadline = time.time() + max(timeout_ms, 2000) / 1000.0
    url = ""
    while time.time() < deadline:
        url = str(getattr(page, "url", "") or "")
        if re.search(r"/discussion/\d+", url):
            return url
        page.wait_for_timeout(250)
    raise RuntimeError(f"Publish did not reach discussion topic URL: {url}")


def parse_channels(raw_channels: list[str]) -> set[str] | None:
    if not raw_channels:
        return None
    values = {item.strip().lower() for item in raw_channels if item.strip()}
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute due Kaggle campaign actions from queue.")
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH, help="Campaign queue JSON path.")
    parser.add_argument("--storage-state", type=Path, default=DEFAULT_STORAGE_STATE, help="Playwright storage state.")
    parser.add_argument("--limit", type=int, default=1, help="Max due actions to execute this run.")
    parser.add_argument("--channel", action="append", default=[], help="Optional channel filter (repeatable).")
    parser.add_argument("--no-include-planned", action="store_true", help="Do not auto-claim planned actions.")
    parser.add_argument("--no-include-in-progress", action="store_true", help="Do not execute in_progress actions.")
    parser.add_argument("--no-respect-schedule", action="store_true", help="Allow execution regardless of scheduled_for.")
    parser.add_argument("--sleep-between-actions-s", type=float, default=45.0, help="Cooldown between actions.")
    parser.add_argument("--sleep-jitter-s", type=float, default=15.0, help="Random jitter added to cooldown.")
    parser.add_argument("--timeout-ms", type=int, default=15000, help="Playwright timeout in ms.")
    parser.add_argument("--headed", action="store_true", help="Run browser headed.")
    parser.add_argument(
        "--manual-login",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow interactive login if session is unauthenticated.",
    )
    parser.add_argument("--email", default=os.environ.get("KAGGLE_EMAIL", ""), help="Kaggle login email.")
    parser.add_argument("--password", default=os.environ.get("KAGGLE_PASSWORD", ""), help="Kaggle login password.")
    parser.add_argument("--dry-run", action="store_true", help="Show selected actions without posting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.sleep_between_actions_s < 0 or args.sleep_jitter_s < 0:
        raise SystemExit("sleep values cannot be negative")

    payload = load_payload(args.queue_path)
    queue = payload["queue"]

    selected = due_supported_actions(
        queue,
        now=now_utc(),
        limit=args.limit,
        allowed_channels=parse_channels(args.channel),
        include_planned=not args.no_include_planned,
        include_in_progress=not args.no_include_in_progress,
        respect_schedule=not args.no_respect_schedule,
    )
    print(f"Selected due actions: {len(selected)}")
    for item in selected:
        print(
            f"- {item.get('id')} [{normalized_channel(item)}] "
            f"{item.get('dataset_ref')} status={normalized_status(item)}"
        )

    if args.dry_run or not selected:
        return 0

    stamp = now_iso()
    for action in selected:
        claim_action(action, stamp=stamp)

    sync_playwright = require_playwright()
    success = 0
    failures = 0
    with sync_playwright() as playwright:
        args.storage_state.parent.mkdir(parents=True, exist_ok=True)
        state_arg = str(args.storage_state) if args.storage_state.exists() else None
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(storage_state=state_arg)
        page = context.new_page()
        try:
            maybe_login(
                page,
                timeout_ms=args.timeout_ms,
                manual_login=args.manual_login,
                email=args.email,
                password=args.password,
            )
            context.storage_state(path=str(args.storage_state))

            for idx, action in enumerate(selected):
                action_id = str(action.get("id", ""))
                dataset_ref = str(action.get("dataset_ref", "")).strip()
                if not dataset_ref or "/" not in dataset_ref:
                    mark_error(action, "invalid dataset_ref")
                    failures += 1
                    print(f"[failed] {action_id}: invalid dataset_ref")
                    continue
                copy_text = str(action.get("copy", "")).strip()
                if not copy_text:
                    mark_error(action, "empty copy text")
                    failures += 1
                    print(f"[failed] {action_id}: empty copy text")
                    continue
                title = topic_title_for_action(action)
                try:
                    post_url = post_dataset_discussion_topic(
                        page,
                        dataset_ref=dataset_ref,
                        topic_title=title,
                        body=copy_text,
                        timeout_ms=args.timeout_ms,
                    )
                    mark_done(action, post_url=post_url, stamp=now_iso())
                    success += 1
                    print(f"[done] {action_id}: {post_url}")
                except Exception as exc:
                    mark_error(action, str(exc))
                    failures += 1
                    print(f"[failed] {action_id}: {exc}")

                if idx < len(selected) - 1 and (args.sleep_between_actions_s > 0 or args.sleep_jitter_s > 0):
                    delay = args.sleep_between_actions_s + random.uniform(0.0, args.sleep_jitter_s)
                    print(f"[pause] sleeping {delay:.1f}s before next action")
                    time.sleep(delay)
        finally:
            browser.close()

    payload["updated_at"] = now_iso()
    save_payload(args.queue_path, payload)
    print(f"Queue updated: {args.queue_path}")
    print(f"Execution summary: success={success} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
