"""Queue management for scheduled Kaggle discussion posts."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


POSTABLE_STATUSES = {"pending", "ready", "scheduled"}
POST_DAYS = {0, 2, 4}  # Mon, Wed, Fri
POSTS_PER_WEEK = len(POST_DAYS)
DEFAULT_SCHEDULE_WEEKS = 4


def normalize_status(value: str | None) -> str:
    if not value:
        return "pending"
    normalized = value.strip().lower()
    if normalized == "pending":
        return "scheduled"
    return normalized


def _parse_scheduled(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _next_post_day(dt: datetime) -> datetime:
    dt = dt.replace(hour=14, minute=0, second=0, microsecond=0)
    for _ in range(14):
        if dt.weekday() in POST_DAYS:
            return dt
        dt += timedelta(days=1)
    return dt


def next_pending(queue: list[dict], now: datetime | None = None) -> dict | None:
    """Return the first due scheduled item, else ready item if no future schedule exists."""
    if now is None:
        now = datetime.now(tz=timezone.utc)

    first_ready_unscheduled = None
    has_future_scheduled = False

    for item in queue:
        status = normalize_status(item.get("status"))
        if status not in POSTABLE_STATUSES:
            continue
        scheduled_value = item.get("scheduled_after")
        if not scheduled_value and status == "ready":
            if first_ready_unscheduled is None:
                first_ready_unscheduled = item
            continue
        if not scheduled_value:
            continue
        scheduled = _parse_scheduled(scheduled_value)
        if scheduled is None:
            continue
        if now >= scheduled:
            return item
        has_future_scheduled = True

    if has_future_scheduled:
        return None
    return first_ready_unscheduled


def _rebalance_schedule_window(
    queue: list[dict],
    now: datetime,
    schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS,
) -> None:
    """Keep a rolling schedule window by promoting ready backlog items."""
    if schedule_weeks < 1:
        return

    target_slots = schedule_weeks * POSTS_PER_WEEK
    scheduled_items = [
        item
        for item in queue
        if normalize_status(item.get("status")) == "scheduled" and _parse_scheduled(item.get("scheduled_after"))
    ]

    if len(scheduled_items) >= target_slots:
        return

    scheduled_dates = [
        _parse_scheduled(item.get("scheduled_after")) for item in scheduled_items
    ]
    scheduled_dates = [dt for dt in scheduled_dates if dt is not None]
    cursor = max(scheduled_dates) if scheduled_dates else now

    for item in queue:
        if len(scheduled_dates) >= target_slots:
            break
        if normalize_status(item.get("status")) != "ready":
            continue
        if item.get("scheduled_after"):
            continue

        next_slot = _next_post_day(cursor + timedelta(days=1))
        item["status"] = "scheduled"
        item["scheduled_after"] = next_slot.isoformat()
        scheduled_dates.append(next_slot)
        cursor = next_slot


def _resolve_schedule_weeks() -> int:
    raw = os.environ.get("DISCUSSION_SCHEDULE_WEEKS")
    if not raw:
        return DEFAULT_SCHEDULE_WEEKS
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_SCHEDULE_WEEKS
    return parsed if parsed >= 1 else DEFAULT_SCHEDULE_WEEKS


def mark_posted(queue_path: Path, item_id: str, post_url: str) -> None:
    """Update queue file: set item status to 'posted', record URL and timestamp."""
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    now = datetime.now(tz=timezone.utc)
    for item in queue:
        if item["id"] == item_id:
            item["status"] = "posted"
            item["post_url"] = post_url
            item["posted_at"] = now.isoformat()
            break
    _rebalance_schedule_window(queue, now=now, schedule_weeks=_resolve_schedule_weeks())
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def extract_body(drafts_content: str, section_name: str) -> str:
    """Extract post body from a markdown drafts file by section heading.

    Strips ops metadata lines (forum/category/priority/deadline/status, etc.)
    and the ### sub-heading (which becomes the post title, not the body).
    """
    patterns = [
        # Canonical section key, e.g. "Draft 7"
        rf"^## {re.escape(section_name)}:.*?\n(.*?)(?=\n## |\Z)",
        # Backward compatibility: queue stores the draft title
        rf"^## Draft \d+:\s*{re.escape(section_name)}\s*\n(.*?)(?=\n## |\Z)",
    ]
    body = None
    for pattern in patterns:
        match = re.search(pattern, drafts_content, flags=re.DOTALL | re.MULTILINE)
        if match:
            body = match.group(1).strip()
            break
    if body is None:
        raise ValueError(f"Section not found in drafts: {section_name}")

    # Strip metadata lines kept in drafts for ops scheduling/reporting.
    for meta in (
        r"\*\*Target forum:\*\*.*\n?",
        r"\*\*Category:\*\*.*\n?",
        r"\*\*Expected medal:\*\*.*\n?",
        r"\*\*Priority:\*\*.*\n?",
        r"\*\*Deadline:\*\*.*\n?",
        r"\*\*Status:\*\*.*\n?",
    ):
        body = re.sub(meta, "", body)
    # Strip the ### heading (it becomes the post title, not the body)
    body = re.sub(r"^###.+\n\n?", "", body.lstrip())
    return body.strip()
