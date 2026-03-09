"""Alert via Telegram when a competition deadline is within the threshold."""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# Resolve paths for both container and local test environments
sys.path.insert(0, str(Path(__file__).parent))
_repo = Path(os.environ.get("REPO_PATH", str(Path(__file__).parent.parent.parent)))
sys.path.insert(0, str(_repo))

import notify
from kaggle_portfolio.ops.medal_ops import ParsedDeadline, parse_active_competitions

REPO = Path(os.environ.get("REPO_PATH", str(Path(__file__).parent.parent.parent)))
TRACKER_PATH = REPO / "grandmaster-tracker.md"
ALERT_HOURS = 72


def parse_deadlines(content: str, today: date) -> list[ParsedDeadline]:
    return parse_active_competitions(content, today)


def filter_urgent(deadlines: list[ParsedDeadline], hours: int = 72) -> list[ParsedDeadline]:
    threshold_days = hours / 24
    return [
        d for d in deadlines
        if d.days_to_deadline is not None and 0 <= d.days_to_deadline <= threshold_days
    ]


def format_alert(d: ParsedDeadline) -> str:
    hours_left = int((d.days_to_deadline or 0) * 24)
    return (
        f"⏰ *DEADLINE ALERT*\n"
        f"{d.competition}\n"
        f"Due in ~{hours_left}h ({d.deadline_raw})\n"
        f"Teams: {d.teams} | {d.difficulty}\n"
        f"Strategy: {d.strategy}"
    )


def main() -> None:
    today = date.today()
    if not TRACKER_PATH.exists():
        print(f"Tracker not found: {TRACKER_PATH}", file=sys.stderr)
        sys.exit(1)
    content = TRACKER_PATH.read_text(encoding="utf-8")
    deadlines = parse_deadlines(content, today)
    urgent = filter_urgent(deadlines, hours=ALERT_HOURS)
    if not urgent:
        print(f"No deadlines within {ALERT_HOURS}h.")
        return
    for d in urgent:
        msg = format_alert(d)
        notify.send(msg)
        print(f"Alert sent: {d.competition}")


if __name__ == "__main__":
    main()
