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
from pathlib import Path
from typing import Any
from kaggle_portfolio.shared.errors import CommandError
from kaggle_portfolio.campaigns.campaign_queue import (
    DEFAULT_QUEUE_PATH,
    claim,
    complete,
    PLANNED,
    action_status,
    filter_actions,
    find_by_id,
    load_payload,
    now_iso,
    parse_channels,
    save_payload,
)
from kaggle_portfolio.shared.deps import Deps

DEFAULT_REPORT_PATH = Path("medal_ops") / "reports" / "latest-campaign-runbook.md"


def claim_actions(actions: list[dict[str, Any]]) -> list[str]:
    """Claim planned actions, returning the ids actually claimed.

    This used to claim unconditionally, so it could re-claim work already in
    flight or finished and bump claim_count — which counts retries and cannot
    also count "someone ran the command twice". The model refuses; a deliberate
    re-run is requeue().
    """
    stamp = now_iso()
    return [str(item.get("id", "")) for item in actions if claim(item, stamp=stamp)]


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
        complete(item, note=note, stamp=stamp)
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
        lines.append(
            f"### {item.get('id')} - {item.get('channel')} - {item.get('dataset_ref')}"
        )
        lines.append("")
        lines.append(str(item.get("copy", "")))
        lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operate campaign queue (show/claim/complete)."
    )
    parser.add_argument(
        "--queue-path",
        default=str(DEFAULT_QUEUE_PATH),
        help="Campaign queue JSON path.",
    )
    parser.add_argument(
        "--limit", type=int, default=7, help="Max actions to select for show/claim."
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=[],
        help="Optional channel filter (repeatable).",
    )
    parser.add_argument(
        "--claim", action="store_true", help="Claim selected planned actions."
    )
    parser.add_argument(
        "--complete-id",
        action="append",
        default=[],
        help="Complete a specific action ID (repeatable).",
    )
    parser.add_argument(
        "--note", default=None, help="Completion note (with --complete-id)."
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Runbook markdown output path.",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Do not write runbook markdown."
    )
    parser.add_argument(
        "--print-copy",
        action="store_true",
        help="Print full copy for selected actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview updates without writing queue file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parse_args(argv)
    deps = deps or Deps.resolve(today=getattr(args, "today", None))
    if args.limit < 1:
        raise CommandError("--limit must be >= 1")

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
