#!/usr/bin/env python3
"""Discussion draft scheduler – manages the posting queue in discussion_queue.json.

Supports three modes:
  --health-check   Validate queue health (overdue posts, gaps in schedule)
  --ops-report     Print a human-readable operations report
  --set-id ID      Update a specific draft's metadata

Used by manage.sh (post-discussion, draft-ops, draft-set) and CI workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

QUEUE_PATH = Path(__file__).resolve().parent / "pi-automation" / "data" / "discussion_queue.json"


def load_queue(path: Path = QUEUE_PATH) -> list[dict]:
    if not path.exists():
        print(f"Warning: queue file not found at {path}", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def health_check(queue: list[dict], max_overdue: int, max_days_gap: int) -> bool:
    """Return True if healthy, False if checks fail."""
    now = datetime.now(timezone.utc)
    ok = True

    pending = [d for d in queue if d.get("status") == "pending"]
    overdue = []
    for draft in pending:
        scheduled = draft.get("scheduled_after")
        if scheduled:
            dt = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
            if dt < now:
                overdue.append(draft)

    if len(overdue) > max_overdue:
        print(f"FAIL: {len(overdue)} overdue drafts (max allowed: {max_overdue})")
        for d in overdue:
            print(f"  - {d['id']}: {d['title']} (scheduled {d['scheduled_after']})")
        ok = False
    else:
        print(f"OK: {len(overdue)} overdue drafts (max allowed: {max_overdue})")

    future_pending = sorted(
        [d for d in pending if d.get("scheduled_after") and
         datetime.fromisoformat(d["scheduled_after"].replace("Z", "+00:00")) >= now],
        key=lambda d: d["scheduled_after"],
    )

    if future_pending:
        next_dt = datetime.fromisoformat(
            future_pending[0]["scheduled_after"].replace("Z", "+00:00")
        )
        days_until = (next_dt - now).days
        if days_until > max_days_gap:
            print(f"FAIL: next post in {days_until} days (max allowed: {max_days_gap})")
            ok = False
        else:
            print(f"OK: next post in {days_until} days (max allowed: {max_days_gap})")
    else:
        print("WARN: no future pending drafts scheduled")

    total = len(queue)
    posted = len([d for d in queue if d.get("status") == "posted"])
    print(f"Queue: {total} total, {posted} posted, {len(pending)} pending")

    return ok


def ops_report(queue: list[dict]) -> None:
    """Print an operations report."""
    now = datetime.now(timezone.utc)

    by_status: dict[str, list[dict]] = {}
    for d in queue:
        s = d.get("status", "unknown")
        by_status.setdefault(s, []).append(d)

    print("=== Discussion Draft Operations Report ===")
    print(f"Generated: {now.isoformat()}")
    print()
    for status, drafts in sorted(by_status.items()):
        print(f"[{status.upper()}] ({len(drafts)} drafts)")
        for d in drafts[:5]:
            line = f"  {d['id']}: {d['title']}"
            if d.get("scheduled_after"):
                line += f" (scheduled: {d['scheduled_after']})"
            print(line)
        if len(drafts) > 5:
            print(f"  ... and {len(drafts) - 5} more")
        print()


def set_draft(queue: list[dict], draft_id: str, updates: dict) -> list[dict]:
    """Update a draft by ID. Returns updated queue."""
    found = False
    for d in queue:
        if d["id"] == draft_id:
            d.update(updates)
            found = True
            print(f"Updated {draft_id}: {updates}")
            break
    if not found:
        print(f"ERROR: draft '{draft_id}' not found", file=sys.stderr)
        sys.exit(1)
    return queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discussion draft scheduler")
    parser.add_argument("--health-check", action="store_true", help="Run health checks")
    parser.add_argument("--ops-report", action="store_true", help="Print operations report")
    parser.add_argument("--set-id", metavar="DRAFT_ID", help="Update a specific draft")
    parser.add_argument("--status", help="Set draft status (with --set-id)")
    parser.add_argument("--priority", help="Set draft priority (with --set-id)")
    parser.add_argument("--deadline", help="Set deadline YYYY-MM-DD (with --set-id)")
    parser.add_argument("--clear-deadline", action="store_true", help="Clear deadline")
    parser.add_argument("--schedule-weeks", type=int, help="Reschedule pending drafts over N weeks")
    parser.add_argument("--max-overdue-scheduled", type=int, default=0,
                        help="Max allowed overdue drafts (health-check)")
    parser.add_argument("--max-days-until-next-post", type=int, default=7,
                        help="Max days until next scheduled post (health-check)")
    parser.add_argument("--queue-file", type=Path, default=QUEUE_PATH,
                        help="Path to discussion queue JSON")

    args = parser.parse_args(argv)
    queue = load_queue(args.queue_file)

    if args.health_check:
        ok = health_check(queue, args.max_overdue_scheduled, args.max_days_until_next_post)
        return 0 if ok else 1

    if args.ops_report:
        ops_report(queue)
        return 0

    if args.set_id:
        updates: dict = {}
        if args.status:
            updates["status"] = args.status
        if args.priority:
            updates["priority"] = args.priority
        if args.deadline:
            updates["deadline"] = args.deadline
        if args.clear_deadline:
            updates.pop("deadline", None)
            updates["deadline"] = None
        if args.schedule_weeks:
            now = datetime.now(timezone.utc)
            pending = [d for d in queue if d.get("status") == "pending"]
            interval = timedelta(weeks=args.schedule_weeks) / max(len(pending), 1)
            for i, d in enumerate(pending):
                d["scheduled_after"] = (now + interval * i).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"Rescheduled {len(pending)} pending drafts over {args.schedule_weeks} weeks")

        if updates:
            queue = set_draft(queue, args.set_id, updates)
            with open(args.queue_file, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
