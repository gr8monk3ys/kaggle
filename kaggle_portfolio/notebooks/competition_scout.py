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

import sys
from datetime import datetime, timezone

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import Competition, KaggleClient, KaggleError

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RESET = "\033[0m"

# Topics our notebooks cover — used for alignment scoring
OUR_TOPICS = {
    "classification",
    "regression",
    "nlp",
    "text",
    "cnn",
    "image",
    "time series",
    "forecasting",
    "feature engineering",
    "ensemble",
    "deep learning",
    "bert",
    "tabular",
    "eda",
    "fraud",
    "medical",
}
SCOUT_CATEGORIES = ("featured", "research", "playground", "masters")


def parse_deadline_datetime(value: str) -> datetime:
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        return deadline.replace(tzinfo=timezone.utc)
    return deadline.astimezone(timezone.utc)


def fetch_competitions(
    client: KaggleClient, category: str = "all"
) -> list[Competition]:
    """Fetch competitions from Kaggle, deduplicated by ref across categories."""
    categories = SCOUT_CATEGORIES if category == "all" else (category,)
    merged: dict[str, Competition] = {}
    for category_name in categories:
        try:
            found = client.search_competitions(
                category=None if category_name == "all" else category_name
            )
        except KaggleError as exc:
            print(
                f"{YELLOW}Category {category_name} unavailable{RESET}: {exc}",
                file=sys.stderr,
            )
            continue
        for comp in found:
            if comp.slug:
                merged[comp.slug] = comp
    return list(merged.values())


def score_competition(comp: Competition, now: datetime) -> float:
    """Score a competition 0-100 (higher = better opportunity)."""
    score = 40.0

    # Category quality: prioritize live medal-relevant boards.
    category = comp.category.lower()
    if "featured" in category:
        score += 25
    elif "research" in category:
        score += 22
    elif "playground" in category:
        score += 16
    elif "masters" in category:
        score += 14
    elif "getting started" in category:
        score -= 20

    # Team count: tiny boards are often dead; healthy active boards are better.
    teams = comp.team_count
    if teams == 0:
        score -= 10
    elif teams < 50:
        score -= 6
    elif teams < 200:
        score += 6
    elif teams < 2500:
        score += 12
    elif teams < 5000:
        score += 5
    else:
        score -= 4

    # Deadline proximity: favor active boards, penalize evergreen training boards.
    deadline_str = comp.deadline
    if deadline_str:
        try:
            deadline = parse_deadline_datetime(deadline_str)
            days_left = (deadline - now).days
            if days_left < 0:
                return -1  # already ended
            elif days_left < 5:
                score -= 14
            elif days_left < 14:
                score += 16
            elif days_left < 45:
                score += 12
            elif days_left < 90:
                score += 8
            elif days_left < 180:
                score += 2
            elif days_left < 365:
                score -= 6
            else:
                score -= 24
        except ValueError:
            pass

    # Category/topic alignment
    title = f"{comp.slug} {comp.title}".lower()
    matches = sum(1 for t in OUR_TOPICS if t in title)
    score += min(matches * 4, 12)

    if comp.user_has_entered:
        score += 6

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
        comp: Competition = entry["competition"]
        score = entry["score"]
        days = entry.get("days_left", "?")
        action = (
            "ENTER NOW" if score >= 70 else ("Consider" if score >= 55 else "Monitor")
        )
        lines.append(
            f"| {i} | {comp.slug} | {comp.team_count} | {days} | {score:.0f} | {action} |"
        )
    lines += [
        "",
        "---",
        "*Scores favor live featured/research/playground boards with workable deadlines, healthy team counts, and topic overlap.*",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update", action="store_true", help="Update competition-scout-report.md"
    )
    parser.add_argument("--today", default=None, help="Override today (YYYY-MM-DD).")
    args = parser.parse_args(argv)

    deps = deps or Deps.resolve(today=args.today)

    print(f"{BLUE}=== Competition Intelligence Scout ==={RESET}\n")
    print("Fetching competitions from Kaggle API...")

    now = deps.clock.now()
    competitions = fetch_competitions(deps.client)

    if not competitions:
        print(
            f"{RED}Failed to fetch competitions. Check kaggle CLI credentials.{RESET}"
        )
        return 1

    print(f"Found {len(competitions)} competitions\n")

    scored = []
    for comp in competitions:
        s = score_competition(comp, now)
        if s < 0:
            continue  # ended
        days_left: int | str = "?"
        if comp.deadline:
            try:
                days_left = max(0, (parse_deadline_datetime(comp.deadline) - now).days)
            except ValueError:
                pass
        scored.append({"competition": comp, "score": s, "days_left": days_left})

    ranked = sorted(scored, key=lambda x: -x["score"])

    print(
        f"{'Rank':<5} {'Competition':<50} {'Teams':>6}  {'Days':>5}  {'Score':>6}  Action"
    )
    print("-" * 85)
    for i, entry in enumerate(ranked[:15], 1):
        comp = entry["competition"]
        score = entry["score"]
        action = (
            f"{GREEN}ENTER NOW{RESET}"
            if score >= 70
            else (f"{YELLOW}Consider{RESET}" if score >= 55 else "Monitor")
        )
        score_col = GREEN if score >= 70 else (YELLOW if score >= 55 else RESET)
        print(
            f"  {i:<4} {comp.slug[:48]:<50} {comp.team_count:>6}  "
            f"{str(entry['days_left']):>5}  {score_col}{score:>5.0f}{RESET}  {action}"
        )

    if args.update:
        report = format_report(ranked, now)
        target = deps.layout.scout_report
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"\n{GREEN}Updated {target.name}{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
