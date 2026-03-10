#!/usr/bin/env python3
"""Operate the dataset promotion campaign queue.

Capabilities
------------
1. Show next planned campaign actions
2. Claim next N actions for execution (`planned` -> `in_progress`)
3. Complete claimed actions by ID (`in_progress` -> `done`)
4. Generate a runbook markdown file for the selected actions
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QUEUE_PATH = Path("pi-automation") / "data" / "promotion_campaign_queue.json"
DEFAULT_REPORT_PATH = Path("medal_ops") / "reports" / "latest-campaign-runbook.md"

PLANNED = "planned"
IN_PROGRESS = "in_progress"
DONE = "done"
BLOCKED = "blocked"
VALID_STATUSES = {PLANNED, IN_PROGRESS, DONE, BLOCKED}


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Campaign queue not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid campaign queue payload: {path}")
    queue = payload.get("queue")
    if not isinstance(queue, list):
        raise SystemExit(f"Campaign queue missing 'queue' list: {path}")
    return payload


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def action_status(action: dict[str, Any]) -> str:
    value = str(action.get("status", PLANNED)).strip().lower()
    return value if value in VALID_STATUSES else PLANNED


def sorted_queue(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[str, str]:
        return (str(item.get("scheduled_for", "")), str(item.get("id", "")))

    return sorted(queue, key=key)


def filter_actions(
    queue: list[dict[str, Any]],
    *,
    statuses: set[str],
    channels: set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in sorted_queue(queue):
        status = action_status(item)
        channel = str(item.get("channel", "")).strip().lower()
        if status not in statuses:
            continue
        if channels and channel not in channels:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def parse_channels(raw_channels: list[str]) -> set[str] | None:
    if not raw_channels:
        return None
    values = {value.strip().lower() for value in raw_channels if value.strip()}
    return values or None


def find_by_id(queue: list[dict[str, Any]], action_id: str) -> dict[str, Any] | None:
    for item in queue:
        if str(item.get("id", "")) == action_id:
            return item
    return None


def claim_actions(actions: list[dict[str, Any]]) -> None:
    stamp = now_iso()
    for item in actions:
        item["status"] = IN_PROGRESS
        item["claimed_at"] = stamp
        item["claim_count"] = int(item.get("claim_count") or 0) + 1


def complete_actions(
    queue: list[dict[str, Any]],
    action_ids: list[str],
    *,
    note: str | None,
) -> list[str]:
    completed: list[str] = []
    stamp = now_iso()
    for action_id in action_ids:
        item = find_by_id(queue, action_id)
        if item is None:
            continue
        item["status"] = DONE
        item["completed_at"] = stamp
        if note:
            item["note"] = note
        completed.append(action_id)
    return completed


def render_runbook(actions: list[dict[str, Any]]) -> str:
    lines = [
        "# Campaign Execution Runbook",
        "",
        f"- Generated: {now_iso()}",
        f"- Action count: {len(actions)}",
        "",
        "## Queue",
        "",
        "| ID | Scheduled (UTC) | Channel | Dataset | Status |",
        "|---|---|---|---|---|",
    ]
    for item in actions:
        lines.append(
            f"| {item.get('id')} | {item.get('scheduled_for')} | {item.get('channel')} | "
            f"`{item.get('dataset_ref')}` | {action_status(item)} |"
        )

    lines.extend(["", "## Copy Blocks", ""])
    for item in actions:
        lines.append(f"### {item.get('id')} - {item.get('channel')} - {item.get('dataset_ref')}")
        lines.append("")
        lines.append(str(item.get("copy", "")))
        lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operate campaign queue (show/claim/complete).")
    parser.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH), help="Campaign queue JSON path.")
    parser.add_argument("--limit", type=int, default=7, help="Max actions to select for show/claim.")
    parser.add_argument("--channel", action="append", default=[], help="Optional channel filter (repeatable).")
    parser.add_argument("--claim", action="store_true", help="Claim selected planned actions.")
    parser.add_argument("--complete-id", action="append", default=[], help="Complete a specific action ID (repeatable).")
    parser.add_argument("--note", default=None, help="Completion note (with --complete-id).")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="Runbook markdown output path.")
    parser.add_argument("--no-report", action="store_true", help="Do not write runbook markdown.")
    parser.add_argument("--print-copy", action="store_true", help="Print full copy for selected actions.")
    parser.add_argument("--dry-run", action="store_true", help="Preview updates without writing queue file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    queue_path = Path(args.queue_path)
    report_path = Path(args.report_path)
    payload = load_payload(queue_path)
    queue = payload["queue"]
    channels = parse_channels(args.channel)

    if args.complete_id:
        completed = complete_actions(queue, args.complete_id, note=args.note)
        missing = sorted(set(args.complete_id) - set(completed))
        if missing:
            print(f"Warning: IDs not found: {', '.join(missing)}")
        print(f"Completed actions: {len(completed)}")

    selected = filter_actions(
        queue,
        statuses={PLANNED},
        channels=channels,
        limit=args.limit,
    )
    if args.claim:
        claim_actions(selected)
        print(f"Claimed actions: {len(selected)}")
    else:
        print(f"Selected actions: {len(selected)}")

    if not args.no_report:
        write_text(report_path, render_runbook(selected))
        print(f"Runbook written: {report_path}")

    for item in selected:
        print(
            f"- {item.get('id')} {item.get('scheduled_for')} "
            f"[{item.get('channel')}] {item.get('dataset_ref')} status={action_status(item)}"
        )
        if args.print_copy:
            print("  copy:")
            for line in str(item.get("copy", "")).splitlines():
                print(f"    {line}")

    if not args.dry_run and (args.complete_id or args.claim):
        payload["updated_at"] = now_iso()
        save_payload(queue_path, payload)
        print(f"Queue updated: {queue_path}")
    elif args.dry_run and (args.complete_id or args.claim):
        print("Dry run: queue file not updated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
