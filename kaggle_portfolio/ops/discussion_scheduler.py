#!/usr/bin/env python3
"""Parse discussion-drafts.md into a JSON queue and post the next ready draft.

Usage
-----
    python3 -m kaggle_portfolio.ops.discussion_scheduler --dry-run     # print next 3 queued drafts
    python3 -m kaggle_portfolio.ops.discussion_scheduler               # post next ready draft via Playwright
    python3 -m kaggle_portfolio.ops.discussion_scheduler --init        # (re-)generate queue from drafts file
    python3 -m kaggle_portfolio.ops.discussion_scheduler --init --schedule-weeks 4
                                                                       # keep a rolling 4-week scheduled window
    python3 -m kaggle_portfolio.ops.discussion_scheduler --ops-report  # show stage + backlog priority summary
    python3 -m kaggle_portfolio.ops.discussion_scheduler --health-check
                                                                       # fail on stale schedule SLAs
    python3 -m kaggle_portfolio.ops.discussion_scheduler --set-id draft_007 --priority high --deadline 2026-03-05
                                                                       # edit a draft and re-balance the window

Invoked by: ./manage.sh post-discussion [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import ceil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from kaggle_portfolio.shared.kaggle_utils import parse_iso_date

ROOT = Path(__file__).resolve().parents[2]
DRAFTS_FILE = ROOT / "docs" / "discussions" / "discussion-drafts.md"
QUEUE_FILE = ROOT / "pi-automation" / "data" / "discussion_queue.json"
PI_SCRIPTS = ROOT / "pi-automation" / "scripts"

# Kaggle forum URLs by forum key
FORUM_MAP = {
    "getting started": "https://www.kaggle.com/discussions/getting-started",
    "general": "https://www.kaggle.com/discussions/general",
    "med-gemma competition": "https://www.kaggle.com/competitions/med-gemma-challenge/discussion",
    "deep past competition": "https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion",
    "deep past akkadian": "https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion",
    "deep past initiative": "https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion",
    "nlp getting started": "https://www.kaggle.com/competitions/nlp-getting-started/discussion",
    "nlp disaster tweets": "https://www.kaggle.com/competitions/nlp-getting-started/discussion",
    "store sales": "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/discussion",
    "spaceship titanic": "https://www.kaggle.com/competitions/spaceship-titanic/discussion",
    "titanic": "https://www.kaggle.com/competitions/titanic/discussion",
    "digit recognizer": "https://www.kaggle.com/competitions/digit-recognizer/discussion",
}
DEFAULT_FORUM = "https://www.kaggle.com/discussions/getting-started"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"

VALID_STATUSES = {"idea", "ready", "scheduled", "posted", "won-medal", "pending", "skipped"}
POSTABLE_STATUSES = {"ready", "scheduled", "pending"}
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
POST_DAYS = {0, 2, 4}  # Mon, Wed, Fri
POSTS_PER_WEEK = len(POST_DAYS)
DEFAULT_SCHEDULE_WEEKS = 4


def normalize_status(value: str | None) -> str:
    if not value:
        return "ready"
    normalized = value.strip().lower()
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


def canonical_draft_id(value: str) -> str:
    text = value.strip()
    match = re.fullmatch(r"draft[-_](\d+)", text.lower())
    if not match:
        return text
    return f"draft_{int(match.group(1)):03d}"


def draft_number(value: str | None) -> int:
    if not value:
        return 999
    match = re.search(r"(\d+)", value)
    if not match:
        return 999
    return int(match.group(1))


def infer_priority(forum_url: str, category: str) -> str:
    category_l = category.lower()
    if "/competitions/" in forum_url:
        return "high"
    if "announcement" in category_l:
        return "low"
    return "medium"


def resolve_forum(forum_key: str) -> str:
    """Map a parsed 'Target forum' label to a forum URL.

    Matches the longest FORUM_MAP key contained in the label first, so specific
    boards (e.g. 'nlp getting started') win over generic substrings
    ('getting started').
    """
    for key in sorted(FORUM_MAP, key=len, reverse=True):
        if key in forum_key:
            return FORUM_MAP[key]
    return DEFAULT_FORUM


def parse_drafts(drafts_path: Path) -> list[dict]:
    """Parse discussion-drafts.md into a list of draft dicts."""
    content = drafts_path.read_text(encoding="utf-8")
    drafts = []

    # Each draft starts with "## Draft N: Title"
    pattern = re.compile(
        r"## (Draft \d+): (.+?)\n(.*?)(?=\n## Draft |\Z)", re.DOTALL
    )
    for m in pattern.finditer(content):
        draft_label = m.group(1)
        title_from_header = m.group(2).strip()
        body_block = m.group(3)

        # Extract metadata lines
        forum_m = re.search(r"\*\*Target forum:\*\*\s*(.+)", body_block)
        forum_key = forum_m.group(1).strip().lower() if forum_m else "getting started"
        forum_url = resolve_forum(forum_key)
        category_m = re.search(r"\*\*Category:\*\*\s*(.+)", body_block)
        category = category_m.group(1).strip() if category_m else ""
        expected_medal_m = re.search(r"\*\*Expected medal:\*\*\s*(.+)", body_block)
        expected_medal = expected_medal_m.group(1).strip() if expected_medal_m else ""
        deadline_m = re.search(r"\*\*Deadline:\*\*\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", body_block)
        deadline = deadline_m.group(1).strip() if deadline_m else None
        priority_m = re.search(r"\*\*Priority:\*\*\s*(.+)", body_block)
        status_m = re.search(r"\*\*Status:\*\*\s*(.+)", body_block)

        priority = normalize_priority(
            priority_m.group(1) if priority_m else infer_priority(forum_url, category)
        )
        status = normalize_status(status_m.group(1) if status_m else "ready")

        # Extract ### heading as post title
        heading_m = re.search(r"^### (.+)$", body_block, re.MULTILINE)
        post_title = heading_m.group(1).strip() if heading_m else title_from_header

        # Draft number for ordering
        num_m = re.search(r"\d+", draft_label)
        num = int(num_m.group()) if num_m else 999

        drafts.append({
            "id": f"draft_{num:03d}",
            "number": num,
            "title": post_title,
            "forum_url": forum_url,
            "body_section": draft_label,
            "body_title": title_from_header,
            "body_file": "discussion-drafts.md",
            "category": category,
            "expected_medal": expected_medal,
            "priority": priority,
            "deadline": deadline,
            "status": status,
        })

    return drafts


def generate_queue(
    drafts: list[dict],
    start_date: datetime | None = None,
    schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS,
) -> list[dict]:
    """Assign staggered schedule to drafts: ~2-3 posts per week (Mon/Wed/Fri)."""
    if schedule_weeks < 1:
        raise ValueError("schedule_weeks must be >= 1")
    if start_date is None:
        start_date = datetime.now(tz=timezone.utc)

    def next_post_day(dt: datetime) -> datetime:
        dt = dt.replace(hour=14, minute=0, second=0, microsecond=0)  # 14:00 UTC
        for _ in range(14):  # max 2 weeks lookahead
            if dt.weekday() in POST_DAYS:
                return dt
            dt += timedelta(days=1)
        return dt

    queue = []
    current_dt = next_post_day(start_date)
    max_scheduled = schedule_weeks * POSTS_PER_WEEK
    scheduled_count = 0

    def sort_key(draft: dict) -> tuple[int, date, int]:
        pr = PRIORITY_RANK.get(normalize_priority(draft.get("priority")), PRIORITY_RANK["medium"])
        dd = parse_iso_date(draft.get("deadline")) or date.max
        return (pr, dd, int(draft.get("number", 999)))

    for draft in sorted(drafts, key=sort_key):
        draft_status = normalize_status(draft.get("status"))
        if draft_status in {"posted", "won-medal", "idea", "skipped"}:
            scheduled_after = None
            item_status = draft_status
        elif scheduled_count >= max_scheduled:
            scheduled_after = None
            item_status = "ready"
        else:
            scheduled_after = current_dt.isoformat()
            item_status = "scheduled"
            scheduled_count += 1

        queue.append({
            "id": draft["id"],
            "title": draft["title"],
            "forum_url": draft["forum_url"],
            "body_section": draft["body_section"],
            "body_file": draft["body_file"],
            "priority": normalize_priority(draft.get("priority")),
            "deadline": draft.get("deadline"),
            "category": draft.get("category", ""),
            "expected_medal": draft.get("expected_medal", ""),
            "scheduled_after": scheduled_after,
            "status": item_status,
            "post_url": None,
            "posted_at": None,
        })
        if item_status == "scheduled":
            # Advance to next post day only for scheduled queue entries.
            current_dt += timedelta(days=1)
            current_dt = next_post_day(current_dt)

    return queue


def rebalance_existing_queue(
    queue: list[dict],
    start_date: datetime | None = None,
    schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS,
) -> list[dict]:
    drafts = [
        {
            "id": str(item.get("id", "")),
            "number": draft_number(str(item.get("id", ""))),
            "title": item.get("title", ""),
            "forum_url": item.get("forum_url", DEFAULT_FORUM),
            "body_section": item.get("body_section", ""),
            "body_file": item.get("body_file", "discussion-drafts.md"),
            "category": item.get("category", ""),
            "expected_medal": item.get("expected_medal", ""),
            "priority": normalize_priority(item.get("priority")),
            "deadline": item.get("deadline"),
            "status": normalize_status(item.get("status")),
        }
        for item in queue
    ]
    generated = generate_queue(drafts, start_date=start_date, schedule_weeks=schedule_weeks)
    by_id = {str(item.get("id")): item.copy() for item in queue}
    ordered: list[dict] = []

    for item in generated:
        current = by_id.get(item["id"], {}).copy()
        current["id"] = item["id"]
        current["title"] = item["title"]
        current["forum_url"] = item["forum_url"]
        current["body_section"] = item["body_section"]
        current["body_file"] = item["body_file"]
        current["priority"] = item["priority"]
        current["deadline"] = item["deadline"]
        current["category"] = item.get("category", "")
        current["expected_medal"] = item.get("expected_medal", "")
        current["status"] = item["status"]
        current["scheduled_after"] = item["scheduled_after"]
        current["post_url"] = current.get("post_url")
        current["posted_at"] = current.get("posted_at")
        ordered.append(current)

    return ordered


def update_draft(
    queue: list[dict],
    draft_id: str,
    *,
    status: str | None = None,
    priority: str | None = None,
    deadline: str | None = None,
    clear_deadline: bool = False,
    schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS,
    now: datetime | None = None,
) -> tuple[list[dict], dict]:
    if deadline and clear_deadline:
        raise SystemExit("--deadline and --clear-deadline are mutually exclusive.")
    now = now or datetime.now(tz=timezone.utc)
    target_id = canonical_draft_id(draft_id)

    match_item = None
    for item in queue:
        if canonical_draft_id(str(item.get("id", ""))) == target_id:
            match_item = item
            break
    if not match_item:
        raise SystemExit(f"Draft not found: {draft_id}")

    old_status = normalize_status(match_item.get("status"))

    if priority is not None:
        match_item["priority"] = normalize_priority(priority)

    if clear_deadline:
        match_item["deadline"] = None
    elif deadline is not None:
        parsed_deadline = parse_iso_date(deadline)
        if parsed_deadline is None:
            raise SystemExit(f"Invalid --deadline value: {deadline} (expected YYYY-MM-DD)")
        match_item["deadline"] = parsed_deadline.isoformat()

    if status is not None:
        next_status = normalize_status(status)
        match_item["status"] = next_status
        if next_status in {"idea", "skipped"}:
            match_item["scheduled_after"] = None
        if next_status in {"posted", "won-medal"} and not match_item.get("posted_at"):
            match_item["posted_at"] = now.isoformat()
        if old_status in {"posted", "won-medal"} and next_status not in {"posted", "won-medal"}:
            match_item["post_url"] = None
            match_item["posted_at"] = None

    updated_queue = rebalance_existing_queue(
        queue,
        start_date=now,
        schedule_weeks=schedule_weeks,
    )
    updated_id = str(match_item.get("id", ""))
    updated = next(item for item in updated_queue if str(item.get("id")) == updated_id)
    return updated_queue, updated


def load_queue() -> list[dict]:
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    return []


def save_queue(queue: list[dict]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def show_dry_run(queue: list[dict], n: int = 3) -> None:
    """Print next N pending drafts without posting."""
    now = datetime.now(tz=timezone.utc)
    status_counts = {}
    for item in queue:
        status = normalize_status(item.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    postable = [item for item in queue if normalize_status(item.get("status")) in POSTABLE_STATUSES]
    has_future_scheduled = any(
        normalize_status(item.get("status")) == "scheduled"
        and (parse_scheduled_datetime(item.get("scheduled_after")) or now) > now
        for item in postable
    )

    print(f"{BLUE}=== Next {n} queued discussion drafts ==={RESET}\n")
    print(
        "Queue status: "
        + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        + "\n"
    )

    ready = []
    upcoming = []
    backlog = []
    for item in postable:
        scheduled_dt = parse_scheduled_datetime(item.get("scheduled_after"))
        if not scheduled_dt and normalize_status(item.get("status")) == "ready":
            if has_future_scheduled:
                backlog.append(item)
            else:
                ready.append(item)
            continue
        if not scheduled_dt:
            continue
        if scheduled_dt <= now:
            ready.append(item)
        else:
            upcoming.append(item)

    for item in (ready + upcoming + backlog)[:n]:
        priority = item.get("priority", "medium")
        deadline = item.get("deadline") or "n/a"
        scheduled = (item.get("scheduled_after") or "n/a")[:10]
        if item in ready:
            status_tag = f"{GREEN}READY{RESET}"
        elif item in backlog:
            status_tag = f"{YELLOW}BACKLOG{RESET}"
        else:
            status_tag = f"{YELLOW}{scheduled}{RESET}"
        print(f"  [{status_tag}] {item['title']}")
        print(f"    Forum: {item['forum_url']}")
        print(f"    Priority: {priority} | Deadline: {deadline}")
        print(f"    Section: {item['body_section']}")
        print()


def build_ops_summary(queue: list[dict], now: datetime | None = None) -> dict:
    now = now or datetime.now(tz=timezone.utc)
    counts: dict[str, int] = {}
    backlog = [item for item in queue if normalize_status(item.get("status")) in {"idea", "ready", "scheduled"}]

    ready_backlog = 0
    due_scheduled = 0
    has_future_scheduled = False
    scheduled_next_7d = 0
    overdue_scheduled = 0
    scheduled_dates: list[date] = []
    postable_backlog = 0
    next_future_scheduled: datetime | None = None

    for item in queue:
        status = normalize_status(item.get("status"))
        counts[status] = counts.get(status, 0) + 1

    for item in backlog:
        status = normalize_status(item.get("status"))
        sched = parse_scheduled_datetime(item.get("scheduled_after"))
        if status in POSTABLE_STATUSES:
            postable_backlog += 1
        if status == "ready" and sched is None:
            ready_backlog += 1
        if sched is None:
            continue
        scheduled_dates.append(sched.date())
        if status in POSTABLE_STATUSES and sched <= now:
            due_scheduled += 1
        if status in POSTABLE_STATUSES and sched > now:
            if next_future_scheduled is None or sched < next_future_scheduled:
                next_future_scheduled = sched
        if now <= sched <= now + timedelta(days=7):
            scheduled_next_7d += 1
        if status == "scheduled" and sched < now:
            overdue_scheduled += 1
        if status == "scheduled" and sched > now:
            has_future_scheduled = True

    schedule_horizon = max(scheduled_dates).isoformat() if scheduled_dates else "n/a"
    estimated_weeks = (postable_backlog + 2) // 3 if postable_backlog else 0
    ready_now = due_scheduled if due_scheduled > 0 or has_future_scheduled else ready_backlog
    next_post_due: str | None = None
    days_until_next_post: int | None = None
    if due_scheduled > 0:
        next_post_due = now.isoformat()
        days_until_next_post = 0
    elif next_future_scheduled is not None:
        next_post_due = next_future_scheduled.isoformat()
        days_until_next_post = max(
            0,
            ceil((next_future_scheduled - now).total_seconds() / 86400.0),
        )
    elif ready_backlog > 0:
        next_post_due = now.isoformat()
        days_until_next_post = 0

    return {
        "stage_counts": counts,
        "backlog_total": len(backlog),
        "ready_now": ready_now,
        "ready_backlog": ready_backlog,
        "scheduled_next_7d": scheduled_next_7d,
        "overdue_scheduled": overdue_scheduled,
        "schedule_horizon": schedule_horizon,
        "estimated_weeks_to_clear": estimated_weeks,
        "next_post_due": next_post_due,
        "days_until_next_post": days_until_next_post,
    }


def show_ops_report(queue: list[dict], n: int = 10) -> None:
    """Print ops summary focused on backlog stage and prioritization."""
    now = datetime.now(tz=timezone.utc)
    summary = build_ops_summary(queue, now=now)
    counts = summary["stage_counts"]

    print(f"{BLUE}=== Draft Ops Report ==={RESET}\n")
    print("Stage counts:")
    for status in ("idea", "ready", "scheduled", "posted", "won-medal", "skipped"):
        print(f"  - {status}: {counts.get(status, 0)}")
    print()

    print("Flow health:")
    print(f"  - backlog_total: {summary['backlog_total']}")
    print(f"  - ready_now: {summary['ready_now']}")
    print(f"  - ready_backlog: {summary['ready_backlog']}")
    print(f"  - scheduled_next_7d: {summary['scheduled_next_7d']}")
    print(f"  - overdue_scheduled: {summary['overdue_scheduled']}")
    next_due = summary["next_post_due"][:10] if summary["next_post_due"] else "n/a"
    print(f"  - next_post_due: {next_due}")
    print(f"  - days_until_next_post: {summary['days_until_next_post'] if summary['days_until_next_post'] is not None else 'n/a'}")
    print(f"  - schedule_horizon: {summary['schedule_horizon']}")
    print(f"  - estimated_weeks_to_clear: {summary['estimated_weeks_to_clear']} (at 3 posts/week)")
    print()

    backlog = [item for item in queue if normalize_status(item.get("status")) in {"idea", "ready", "scheduled"}]
    if not backlog:
        print(f"{GREEN}No active backlog items.{RESET}")
        return

    def backlog_key(item: dict) -> tuple[int, date, str]:
        pr = PRIORITY_RANK.get(normalize_priority(item.get("priority")), PRIORITY_RANK["medium"])
        dd = parse_iso_date(item.get("deadline")) or date.max
        return (pr, dd, str(item.get("id", "")))

    print("Top backlog priorities:")
    for item in sorted(backlog, key=backlog_key)[:n]:
        status = normalize_status(item.get("status"))
        deadline = item.get("deadline") or "n/a"
        priority = normalize_priority(item.get("priority"))
        sched = item.get("scheduled_after") or "n/a"
        is_ready = False
        sched_dt = parse_scheduled_datetime(item.get("scheduled_after"))
        if status == "ready" and sched_dt is None:
            is_ready = True
        elif status in POSTABLE_STATUSES and sched_dt is not None:
            is_ready = sched_dt <= now
        readiness = "READY" if is_ready else status.upper()
        print(f"  - {item.get('id')}: {item.get('title')} [{readiness}]")
        print(f"    priority={priority} deadline={deadline} scheduled={sched[:10] if sched != 'n/a' else 'n/a'}")


def run_health_check(
    queue: list[dict],
    *,
    max_overdue_scheduled: int = 0,
    max_days_until_next_post: int = 7,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(tz=timezone.utc)
    summary = build_ops_summary(queue, now=now)
    failures: list[str] = []

    overdue = int(summary["overdue_scheduled"])
    if overdue > max_overdue_scheduled:
        failures.append(
            f"overdue_scheduled={overdue} exceeds max_overdue_scheduled={max_overdue_scheduled}"
        )

    next_gap = summary["days_until_next_post"]
    if next_gap is None:
        failures.append("No upcoming post found in queue.")
    elif next_gap > max_days_until_next_post:
        failures.append(
            f"days_until_next_post={next_gap} exceeds max_days_until_next_post={max_days_until_next_post}"
        )

    print(f"{BLUE}=== Draft Ops Health Check ==={RESET}")
    print(f"as_of={now.isoformat()}")
    print(f"backlog_total={summary['backlog_total']}")
    print(f"ready_backlog={summary['ready_backlog']}")
    print(f"ready_now={summary['ready_now']}")
    print(f"overdue_scheduled={overdue}")
    print(f"next_post_due={(summary['next_post_due'] or 'n/a')[:10]}")
    print(
        "days_until_next_post="
        + str(summary["days_until_next_post"] if summary["days_until_next_post"] is not None else "n/a")
    )

    if failures:
        print(f"{RED}SLA VIOLATION{RESET}")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"{GREEN}SLA OK{RESET}")
    return 0


def do_post(queue: list[dict], schedule_weeks: int = DEFAULT_SCHEDULE_WEEKS) -> int:
    """Invoke pi-automation/scripts/discussion_post.py to post next ready item."""
    import os
    env = os.environ.copy()
    env["QUEUE_PATH"] = str(QUEUE_FILE)
    env["REPO_PATH"] = str(ROOT)
    env["DISCUSSION_SCHEDULE_WEEKS"] = str(schedule_weeks)

    result = __import__("subprocess").run(
        [sys.executable, str(PI_SCRIPTS / "discussion_post.py")],
        env=env, cwd=str(ROOT),
    )
    return result.returncode


def select_next_post(queue: list[dict], now: datetime | None = None) -> dict | None:
    """Pick the next postable draft: due items first (most overdue), then soonest upcoming; priority breaks ties."""
    now = now or datetime.now(tz=timezone.utc)
    postable = [item for item in queue if normalize_status(item.get("status")) in POSTABLE_STATUSES]
    if not postable:
        return None

    def sort_key(item: dict) -> tuple:
        sched = parse_scheduled_datetime(item.get("scheduled_after"))
        is_due = 0 if (sched is None or sched <= now) else 1
        sched_ord = sched.timestamp() if sched is not None else 0.0
        prio = PRIORITY_RANK.get(normalize_priority(item.get("priority")), 1)
        return (is_due, sched_ord, prio, str(item.get("id", "")))

    return sorted(postable, key=sort_key)[0]


def extract_post_body(drafts_path: Path, body_section: str) -> str:
    """Return a draft's post content (ops `**Field:**` metadata lines stripped) from the drafts markdown."""
    if not body_section:
        return ""
    try:
        content = Path(drafts_path).read_text(encoding="utf-8")
    except OSError:
        return ""
    pattern = re.compile(rf"## {re.escape(body_section)}:.*?\n(.*?)(?=\n## Draft |\Z)", re.DOTALL)
    match = pattern.search(content)
    if not match:
        return ""
    body_lines = [
        line for line in match.group(1).splitlines()
        if not re.match(r"\s*\*\*[A-Za-z /]+:\*\*", line)
    ]
    return "\n".join(body_lines).strip()


def format_next_post(draft: dict, body: str) -> str:
    """Render a postable draft as a copy-paste block for manual posting."""
    draft_id = draft.get("id", "?")
    return "\n".join([
        f"Next post to publish - {draft.get('title', '(untitled)')}",
        f"Forum: {draft.get('forum_url', '?')}",
        f"After posting, mark it done: ./manage.sh draft-set {draft_id} --status posted",
        "",
        "----- copy below this line -----",
        body or "(no body found for this draft in discussion-drafts.md)",
        "----- end -----",
    ])


def cmd_next_post(queue: list[dict] | None = None, drafts_path: Path | None = None,
                  now: datetime | None = None) -> int:
    """Surface the next ready draft for manual posting (safe assist; never automates posting)."""
    queue = load_queue() if queue is None else queue
    draft = select_next_post(queue, now=now)
    if draft is None:
        print("No postable drafts queued. Mark a draft 'ready' first "
              "(./manage.sh draft-set <id> --status ready).")
        return 0
    body = extract_post_body(drafts_path or DRAFTS_FILE, draft.get("body_section", ""))
    print(format_next_post(draft, body))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discussion queue scheduler.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print next 3 queued drafts without posting.")
    parser.add_argument("--init", action="store_true",
                        help="(Re-)generate queue JSON from discussion-drafts.md.")
    parser.add_argument("--show", type=int, default=3,
                        help="Number of drafts to show in --dry-run (default 3).")
    parser.add_argument(
        "--schedule-weeks",
        type=int,
        default=DEFAULT_SCHEDULE_WEEKS,
        help=(
            "Rolling schedule window size used for --init and post-time refills "
            f"(default {DEFAULT_SCHEDULE_WEEKS})."
        ),
    )
    parser.add_argument("--ops-report", action="store_true",
                        help="Show stage counts and prioritized backlog view.")
    parser.add_argument("--health-check", action="store_true",
                        help="Run stale-draft SLA checks and exit non-zero on violation.")
    parser.add_argument("--today", default=None,
                        help="Optional YYYY-MM-DD override used by health-check for deterministic runs.")
    parser.add_argument("--max-overdue-scheduled", type=int, default=0,
                        help="SLA: maximum allowed overdue scheduled drafts (default 0).")
    parser.add_argument("--max-days-until-next-post", type=int, default=7,
                        help="SLA: maximum allowed days until next postable draft (default 7).")
    parser.add_argument("--set-id", default=None,
                        help="Update a queue item by id (e.g. draft_007).")
    parser.add_argument("--status", choices=["idea", "ready", "scheduled", "posted", "won-medal"],
                        help="Set draft status for --set-id.")
    parser.add_argument("--priority", choices=["high", "medium", "low"],
                        help="Set draft priority for --set-id.")
    parser.add_argument("--deadline", default=None,
                        help="Set draft deadline (YYYY-MM-DD) for --set-id.")
    parser.add_argument("--clear-deadline", action="store_true",
                        help="Clear draft deadline for --set-id.")
    parser.add_argument("--next-post", action="store_true",
                        help="Surface the next ready draft to post manually (safe assist; no automation).")
    return parser


def initialize_queue_if_needed(args: argparse.Namespace) -> bool:
    if not args.init and QUEUE_FILE.exists():
        return False

    print(f"Parsing {DRAFTS_FILE.name}...")
    drafts = parse_drafts(DRAFTS_FILE)
    print(f"  Found {len(drafts)} drafts")

    # Preserve posted status from existing queue
    existing = {i["id"]: i for i in load_queue()}
    queue = generate_queue(drafts, schedule_weeks=args.schedule_weeks)
    for item in queue:
        previous = existing.get(item["id"])
        if not previous:
            continue
        prev_status = normalize_status(previous.get("status"))
        if prev_status in {"posted", "won-medal"}:
            item["status"] = prev_status
            item["post_url"] = previous.get("post_url")
            item["posted_at"] = previous.get("posted_at")
        elif prev_status == "idea":
            item["status"] = "idea"
            item["scheduled_after"] = None

    save_queue(queue)
    print(f"  Queue saved to {QUEUE_FILE}")
    return True


def run_selected_mode(args: argparse.Namespace, queue: list[dict]) -> int:
    if args.set_id:
        if not any([args.status, args.priority, args.deadline, args.clear_deadline]):
            raise SystemExit(
                "--set-id requires at least one of --status/--priority/--deadline/--clear-deadline."
            )
        updated_queue, updated = update_draft(
            queue,
            args.set_id,
            status=args.status,
            priority=args.priority,
            deadline=args.deadline,
            clear_deadline=args.clear_deadline,
            schedule_weeks=args.schedule_weeks,
        )
        save_queue(updated_queue)
        scheduled_value = (updated.get("scheduled_after") or "n/a")[:10]
        print(
            f"Updated {updated.get('id')}: "
            f"status={updated.get('status')} "
            f"priority={updated.get('priority')} "
            f"deadline={updated.get('deadline') or 'n/a'} "
            f"scheduled={scheduled_value}"
        )
        return 0

    if args.next_post:
        return cmd_next_post(queue)
    if args.dry_run:
        show_dry_run(queue, n=args.show)
        return 0
    if args.ops_report:
        show_ops_report(queue)
        return 0
    if args.health_check:
        now_override = None
        if args.today is not None:
            today = parse_iso_date(args.today)
            now_override = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
        return run_health_check(
            queue,
            max_overdue_scheduled=args.max_overdue_scheduled,
            max_days_until_next_post=args.max_days_until_next_post,
            now=now_override,
        )

    return do_post(queue, schedule_weeks=args.schedule_weeks)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.schedule_weeks < 1:
        parser.error("--schedule-weeks must be >= 1")
    if args.max_overdue_scheduled < 0:
        parser.error("--max-overdue-scheduled must be >= 0")
    if args.max_days_until_next_post < 0:
        parser.error("--max-days-until-next-post must be >= 0")
    if args.today is not None and parse_iso_date(args.today) is None:
        parser.error("--today must be YYYY-MM-DD")

    mode_count = sum([
        1 if args.dry_run else 0,
        1 if args.ops_report else 0,
        1 if args.health_check else 0,
        1 if args.set_id else 0,
        1 if args.next_post else 0,
    ])
    if mode_count > 1:
        parser.error("Choose only one of --dry-run, --ops-report, --health-check, --set-id, or --next-post.")

    queue_initialized = initialize_queue_if_needed(args)
    if queue_initialized:
        if not args.dry_run and not args.ops_report and not args.health_check and not args.set_id and not args.next_post:
            return 0

    queue = load_queue()
    return run_selected_mode(args, queue)


if __name__ == "__main__":
    sys.exit(main())
