#!/usr/bin/env python3
"""The Draft Queue: one model, imported by everything that touches it.

This existed twice — once in ``ops/discussion_scheduler`` and once in
``pi-automation/scripts/discussion_queue`` — joined only by a JSON file and three
environment variables across a process boundary. The copies drifted, most
consequentially in *which draft posts next*: the scheduler ordered by due-ness,
schedule, priority and id, while the poster returned nothing at all whenever any
future-scheduled item existed. See ``docs/adr/0003-one-draft-queue-model.md``.

The scheduler's implementations are the canonical ones and live here now.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

VALID_STATUSES = {
    "idea",
    "ready",
    "scheduled",
    "posted",
    "won-medal",
    "pending",
    "skipped",
    "expired",
    # Asserts a measured result the repo cannot back. Non-postable until the
    # number is either produced for real or removed from the draft.
    "unverified",
}
#: Statuses eligible to publish.
POSTABLE_STATUSES = {"ready", "scheduled", "pending"}
# Statuses the scheduler must carry through untouched instead of assigning a
# slot to. Derived from the postable set so a newly added status is excluded by
# default — hardcoding this list previously let "expired" drafts be rescheduled
# as postable.
TERMINAL_STATUSES = VALID_STATUSES - POSTABLE_STATUSES
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
POST_DAYS = {0, 2, 4}  # Mon, Wed, Fri
POSTS_PER_WEEK = len(POST_DAYS)
DEFAULT_SCHEDULE_WEEKS = 4


def normalize_status(value: str | None) -> str:
    """Canonicalise a draft's status.

    The poster's copy of this returned ``"pending"`` for a blank status and did
    no validation at all, so an annotated status ("expired - do not post") fell
    through to a postable default there while the scheduler stripped it.
    """
    if not value:
        return "ready"
    normalized = value.strip().lower()
    # Drafts annotate a status with a reason ("expired - do not post"); keep the
    # status word so the annotation does not fall through to the default.
    normalized = re.split(r"\s+[-–—]\s+", normalized, maxsplit=1)[0].strip()
    if normalized == "pending":
        return "scheduled"
    if normalized == "skipped":
        return "skipped"
    if normalized in VALID_STATUSES:
        return normalized
    return "ready"


def normalize_priority(value: str | None) -> str:
    if not value:
        return "medium"
    normalized = value.strip().lower()
    if normalized in PRIORITY_RANK:
        return normalized
    return "medium"


def parse_scheduled_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def select_next_post(queue: list[dict], now: datetime | None = None) -> dict | None:
    """Pick the next postable draft.

    Due items first (most overdue), then soonest upcoming; priority breaks ties.
    An unscheduled ``ready`` draft counts as due — the poster's old selector
    instead returned ``None`` whenever any future-scheduled item existed, which
    could starve a ready draft indefinitely behind an unrelated one.

    This is the single selector: the flywheel predicts against it and the poster
    publishes from it, so the two can no longer disagree about which draft was
    handled.
    """
    now = now or datetime.now(tz=timezone.utc)
    postable = [
        item
        for item in queue
        if normalize_status(item.get("status")) in POSTABLE_STATUSES
    ]
    if not postable:
        return None

    def sort_key(item: dict) -> tuple:
        sched = parse_scheduled_datetime(item.get("scheduled_after"))
        is_due = 0 if (sched is None or sched <= now) else 1
        sched_ord = sched.timestamp() if sched is not None else 0.0
        prio = PRIORITY_RANK.get(normalize_priority(item.get("priority")), 1)
        return (is_due, sched_ord, prio, str(item.get("id", "")))

    return sorted(postable, key=sort_key)[0]


def extract_post_body(drafts_path: Path | str, body_section: str) -> str:
    """Return a draft's post body from the drafts file, or "" if absent."""
    if not body_section:
        return ""
    try:
        content = Path(drafts_path).read_text(encoding="utf-8")
    except OSError:
        return ""
    return extract_body(content, body_section)


def extract_body(
    drafts_content: str,
    body_section: str,
    *,
    strict: bool = False,
    strip_heading: bool = False,
) -> str:
    """Extract a draft's body from the drafts markdown.

    The union of the two implementations this replaces. From the poster's: the
    title fallback (queues predating the canonical section key store the draft
    title), stripping the ``###`` sub-heading (it becomes the post title, not the
    body), and raising rather than silently posting an empty body. From the
    scheduler's: stripping *any* ``**Field:**`` metadata line rather than an
    enumerated list, so a newly added ops field cannot leak into a published post.

    ``strict`` raises ``ValueError`` when the section is missing — what the poster
    needs, since an empty body would otherwise be published as-is.

    ``strip_heading`` drops the leading ``###`` line. The two callers genuinely
    disagree here and both are right: the poster sets that text as the post title
    and must not repeat it in the body, while the scheduler renders it as part of
    the copy-paste block a human reads. It is a parameter rather than a winner.
    """
    if not body_section:
        if strict:
            raise ValueError("No body section given")
        return ""

    patterns = (
        # Canonical section key, e.g. "Draft 7"
        rf"^## {re.escape(body_section)}:.*?\n(.*?)(?=\n## |\Z)",
        # Backward compatibility: the queue stores the draft title
        rf"^## Draft \d+:\s*{re.escape(body_section)}\s*\n(.*?)(?=\n## |\Z)",
    )
    body = None
    for pattern in patterns:
        match = re.search(pattern, drafts_content, flags=re.DOTALL | re.MULTILINE)
        if match:
            body = match.group(1)
            break
    if body is None:
        if strict:
            raise ValueError(f"Section not found in drafts: {body_section}")
        return ""

    body = "\n".join(
        line
        for line in body.splitlines()
        if not re.match(r"\s*\*\*[A-Za-z /]+:\*\*", line)
    )
    body = body.strip()
    if strip_heading:
        body = re.sub(r"^###.+\n\n?", "", body.lstrip())
    return body.strip()


def next_post_day(dt: datetime) -> datetime:
    """The next Mon/Wed/Fri slot strictly after *dt*."""
    candidate = dt + timedelta(days=1)
    while candidate.weekday() not in POST_DAYS:
        candidate += timedelta(days=1)
    return candidate


def load_queue(queue_path: Path | str) -> list[dict]:
    try:
        payload = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def save_queue(queue_path: Path | str, queue: list[dict]) -> None:
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def mark_posted(
    queue_path: Path | str,
    item_id: str,
    post_url: str,
    *,
    now: datetime | None = None,
    schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS,
) -> None:
    """Record that *item_id* was published, and rebalance the remaining schedule."""
    now = now or datetime.now(tz=timezone.utc)
    queue = load_queue(queue_path)
    for item in queue:
        if item.get("id") == item_id:
            item["status"] = "posted"
            item["post_url"] = post_url
            item["posted_at"] = now.isoformat()
            break
    rebalance_schedule_window(queue, now=now, schedule_weeks=schedule_weeks)
    save_queue(queue_path, queue)


def rebalance_schedule_window(
    queue: list[dict],
    *,
    now: datetime | None = None,
    schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS,
) -> list[dict]:
    """Top up the rolling schedule window by promoting ready backlog items.

    Incremental on purpose: it only fills empty slots, leaving already-scheduled
    drafts where they are. The scheduler's own ``rebalance_existing_queue`` is a
    different operation — a full regeneration from the drafts file — and stays
    there. This is the one ``mark_posted`` needs after publishing.
    """
    now = now or datetime.now(tz=timezone.utc)
    if schedule_weeks < 1:
        return queue

    target_slots = schedule_weeks * POSTS_PER_WEEK
    scheduled_dates = [
        dt
        for dt in (
            parse_scheduled_datetime(item.get("scheduled_after"))
            for item in queue
            if normalize_status(item.get("status")) == "scheduled"
        )
        if dt is not None
    ]
    if len(scheduled_dates) >= target_slots:
        return queue

    cursor = max(scheduled_dates) if scheduled_dates else now
    for item in queue:
        if len(scheduled_dates) >= target_slots:
            break
        if normalize_status(item.get("status")) != "ready":
            continue
        if item.get("scheduled_after"):
            continue
        next_slot = next_post_day(cursor)
        item["status"] = "scheduled"
        item["scheduled_after"] = next_slot.isoformat()
        scheduled_dates.append(next_slot)
        cursor = next_slot
    return queue


def is_due(item: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now(tz=timezone.utc)
    if normalize_status(item.get("status")) not in POSTABLE_STATUSES:
        return False
    scheduled = parse_scheduled_datetime(item.get("scheduled_after"))
    return scheduled is None or scheduled <= now


def due_today(queue: list[dict], today: date | None = None) -> list[dict]:
    today = today or datetime.now(tz=timezone.utc).date()
    boundary = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)
    return [item for item in queue if is_due(item, boundary)]
