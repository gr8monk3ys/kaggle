#!/usr/bin/env python3
"""Scout active Kaggle competitions ranked by medal opportunity.

Scores each competition by:
  - Team count (fewer = better opportunity, <100 is ideal)
  - Deadline proximity (2-8 weeks is the sweet spot)
  - Category alignment with existing notebooks

Usage
-----
    python3 -m kaggle_portfolio.notebooks.competition_scout          # print ranked list
    python3 -m kaggle_portfolio.notebooks.competition_scout --update # also update docs/reports/competition-scout-report.md

Invoked by: ./manage.sh scout [--update]
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from kaggle_portfolio.shared.kaggle_utils import kaggle_command

ROOT = Path(__file__).resolve().parents[2]
SCOUT_REPORT = ROOT / "docs" / "reports" / "competition-scout-report.md"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RESET = "\033[0m"

# Topics our notebooks cover — used for alignment scoring
OUR_TOPICS = {
    "classification", "regression", "nlp", "text", "cnn", "image",
    "time series", "forecasting", "feature engineering", "ensemble",
    "deep learning", "bert", "tabular", "eda", "fraud", "medical",
}


def parse_deadline_datetime(value: str) -> datetime:
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        return deadline.replace(tzinfo=timezone.utc)
    return deadline.astimezone(timezone.utc)




def fetch_competitions(category: str = "all", page_size: int = 50) -> list[dict]:
    """Fetch competitions from Kaggle CLI as CSV."""
    cmd = [*kaggle_command(), "competitions", "list", "--csv",
           "--page-size", str(page_size),
           "--sort-by", "latestDeadline"]
    if category != "all":
        cmd += ["--category", category]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return _parse_csv(result.stdout)


def _parse_csv(raw: str) -> list[dict]:
    import csv, io
    rows = list(csv.DictReader(io.StringIO(raw)))
    return rows


def score_competition(row: dict, now: datetime) -> float:
    """Score a competition 0-100 (higher = better opportunity)."""
    score = 50.0

    # Team count (lower = better)
    try:
        teams = int(row.get("teamCount", row.get("team_count", 0)) or 0)
        if teams == 0:
            score += 20   # newly launched, no teams yet
        elif teams < 50:
            score += 15
        elif teams < 100:
            score += 10
        elif teams < 500:
            score += 0
        else:
            score -= 10   # saturated
    except (ValueError, TypeError):
        pass

    # Deadline proximity
    deadline_str = row.get("deadline", row.get("evaluationDate", ""))
    if deadline_str:
        try:
            deadline = parse_deadline_datetime(deadline_str)
            days_left = (deadline - now).days
            if days_left < 0:
                return -1  # already ended
            elif days_left < 7:
                score -= 20  # too close
            elif days_left < 21:
                score += 5
            elif days_left < 56:
                score += 15  # sweet spot: 3-8 weeks
            elif days_left < 120:
                score += 10
            else:
                score += 0   # too far out
        except ValueError:
            pass

    # Category/topic alignment
    title = (row.get("ref", "") + " " + row.get("title", "")).lower()
    matches = sum(1 for t in OUR_TOPICS if t in title)
    score += min(matches * 5, 15)

    # Reward "getting started" competitions
    category = row.get("category", "").lower()
    if "getting started" in category or "playground" in category:
        score += 10

    return round(score, 1)


def format_report(ranked: list[dict], now: datetime) -> str:
    """Format a markdown report of ranked competitions."""
    lines = [
        "# Competition Scout Report",
        "",
        f"*Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## Ranked by Medal Opportunity",
        "",
        "| Rank | Competition | Teams | Days Left | Score | Action |",
        "|------|-------------|-------|-----------|-------|--------|",
    ]
    for i, entry in enumerate(ranked[:20], 1):
        row = entry["row"]
        score = entry["score"]
        teams = row.get("teamCount", row.get("team_count", "?"))
        days = entry.get("days_left", "?")
        ref = row.get("ref", "?")
        action = "ENTER NOW" if score >= 70 else ("Consider" if score >= 55 else "Monitor")
        lines.append(f"| {i} | {ref} | {teams} | {days} | {score:.0f} | {action} |")
    lines += ["", "---", "*Scores: team count (40%) + deadline (30%) + topic alignment (30%)*"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Update competition-scout-report.md")
    args = parser.parse_args(argv)

    print(f"{BLUE}=== Competition Intelligence Scout ==={RESET}\n")
    print("Fetching competitions from Kaggle API...")

    now = datetime.now(tz=timezone.utc)
    rows = fetch_competitions()

    if not rows:
        print(f"{RED}Failed to fetch competitions. Check kaggle CLI credentials.{RESET}")
        return 1

    print(f"Found {len(rows)} competitions\n")

    scored = []
    for row in rows:
        s = score_competition(row, now)
        if s < 0:
            continue  # ended
        # Compute days_left for display
        deadline_str = row.get("deadline", row.get("evaluationDate", ""))
        days_left = "?"
        if deadline_str:
            try:
                deadline = parse_deadline_datetime(deadline_str)
                days_left = max(0, (deadline - now).days)
            except ValueError:
                pass
        scored.append({"row": row, "score": s, "days_left": days_left})

    ranked = sorted(scored, key=lambda x: -x["score"])

    print(f"{'Rank':<5} {'Competition':<50} {'Teams':>6}  {'Days':>5}  {'Score':>6}  Action")
    print("-" * 85)
    for i, entry in enumerate(ranked[:15], 1):
        row = entry["row"]
        score = entry["score"]
        teams = str(row.get("teamCount", row.get("team_count", "?")))
        days = str(entry.get("days_left", "?"))
        ref = row.get("ref", "?")[:48]
        action = f"{GREEN}ENTER NOW{RESET}" if score >= 70 else (f"{YELLOW}Consider{RESET}" if score >= 55 else "Monitor")
        score_col = GREEN if score >= 70 else (YELLOW if score >= 55 else RESET)
        print(f"  {i:<4} {ref:<50} {teams:>6}  {days:>5}  {score_col}{score:>5.0f}{RESET}  {action}")

    if args.update:
        report = format_report(ranked, now)
        SCOUT_REPORT.write_text(report, encoding="utf-8")
        print(f"\n{GREEN}Updated {SCOUT_REPORT.name}{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
