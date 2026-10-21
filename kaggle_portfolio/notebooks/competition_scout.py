#!/usr/bin/env python3
"""Scout active Kaggle competitions for the two things they can earn.

Competitions feed two different tiers, and a board that is good for one is
often useless for the other:

- **Competition medals** come only from Featured and Research boards
  (Playground, Getting Started and InClass award none). A first medal needs
  runway to iterate and a generous bronze cutoff: under 250 teams the top 40%
  earn bronze, from 1,000 teams only the top 10% do.
- **Notebook votes** come from readers, so any active board with a large field
  is worth a strong public notebook, Playground included, and posting early
  matters more than anything else.

Usage
-----
    python3 -m kaggle_portfolio.notebooks.competition_scout          # print both rankings
    python3 -m kaggle_portfolio.notebooks.competition_scout --update # also rewrite docs/reports/competition-scout-report.md

Invoked by: ./manage.sh scout [--update]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import Competition, KaggleClient, KaggleError

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"

# Topics the existing notebooks already cover; overlap shortens the ramp-up.
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
SCOUT_CATEGORIES = ("featured", "research", "playground")
#: Only these categories award competition medals and ranking points.
MEDAL_CATEGORIES = ("featured", "research")


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


def awards_medals(comp: Competition) -> bool:
    category = comp.category.lower()
    return any(name in category for name in MEDAL_CATEGORIES)


def bronze_cutoff(team_count: int) -> int:
    """How many of the top places earn at least bronze on a medal board."""
    if team_count < 250:
        return math.floor(team_count * 0.4)
    if team_count < 1000:
        return 100
    return math.floor(team_count * 0.1)


def days_left(comp: Competition, now: datetime) -> int | None:
    if not comp.deadline:
        return None
    try:
        return (parse_deadline_datetime(comp.deadline) - now).days
    except ValueError:
        return None


def _topic_bonus(comp: Competition) -> float:
    title = f"{comp.slug} {comp.title}".lower()
    return min(sum(4 for topic in OUR_TOPICS if topic in title), 12)


def medal_score(comp: Competition, now: datetime) -> float | None:
    """0-100 for a first competition medal, or None when the board cannot give one."""
    days = days_left(comp, now)
    if not awards_medals(comp) or days is None or days < 0:
        return None
    score = 50.0
    # Runway: a first medal takes weeks of iteration, not a final-week sprint.
    if days < 14:
        score -= 30
    elif days < 30:
        score += 5
    elif days <= 90:
        score += 20
    elif days <= 180:
        score += 10
    # How much of the field medals: 40% under 250 teams, 10% from 1,000.
    teams = comp.team_count
    if teams < 30:
        score -= 10  # too small to be a live board, or brand new and unproven
    elif teams:
        score += 30 * bronze_cutoff(teams) / teams
    score += _topic_bonus(comp)
    if comp.user_has_entered:
        score += 6
    return round(max(score, 0.0), 1)


def notebook_score(comp: Competition, now: datetime) -> float | None:
    """0-100 for a public notebook earning votes, or None for ended boards."""
    days = days_left(comp, now)
    if days is None or days < 0:
        return None
    score = 40.0
    # Readers: every team is a potential reader; returns diminish with size.
    score += min(25.0, 5 * math.log10(max(comp.team_count, 1)))
    # Early notebooks collect the forks and votes; late ones are buried.
    if days > 60:
        score += 15
    elif days > 30:
        score += 10
    elif days > 14:
        score += 3
    else:
        score -= 15
    if "getting started" in comp.category.lower():
        score -= 10  # evergreen boards are saturated with tutorials
    score += _topic_bonus(comp)
    return round(max(score, 0.0), 1)


@dataclass(frozen=True)
class Ranked:
    competition: Competition
    score: float
    days_left: int


def rank(competitions: list[Competition], now: datetime, scorer) -> list[Ranked]:
    ranked = []
    for comp in competitions:
        score = scorer(comp, now)
        if score is None:
            continue
        ranked.append(Ranked(comp, score, days_left(comp, now) or 0))
    return sorted(ranked, key=lambda r: (-r.score, r.competition.slug))


def format_report(medals: list[Ranked], notebooks: list[Ranked], now: datetime) -> str:
    lines = [
        "# Competition Scout Report",
        "",
        f"*Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}* — refresh with `./manage.sh scout --update`.",
        "",
        "## Competition medals (Featured and Research only)",
        "",
        "Playground, Getting Started and InClass boards award no medals. Prefer",
        "30-90 days of runway and boards where the bronze cutoff is generous.",
        "",
        "| Rank | Competition | Teams | Days left | Bronze if top | Score |",
        "|------|-------------|-------|-----------|---------------|-------|",
    ]
    for i, r in enumerate(medals[:10], 1):
        teams = r.competition.team_count
        cutoff = bronze_cutoff(teams)
        pct = f" ({cutoff / teams:.0%})" if teams else ""
        lines.append(
            f"| {i} | {r.competition.slug} | {teams} | {r.days_left} | {cutoff}{pct} | {r.score:.0f} |"
        )
    if not medals:
        lines.append("| — | no active medal-awarding board found | | | | |")
    lines += [
        "",
        "## Notebook votes (any active board)",
        "",
        "A strong public notebook early in a big competition is the most direct",
        "route to notebook medals, whether or not the board itself awards any.",
        "",
        "| Rank | Competition | Category | Teams | Days left | Score |",
        "|------|-------------|----------|-------|-----------|-------|",
    ]
    for i, r in enumerate(notebooks[:10], 1):
        c = r.competition
        lines.append(
            f"| {i} | {c.slug} | {c.category} | {c.team_count} | {r.days_left} | {r.score:.0f} |"
        )
    return "\n".join(lines) + "\n"


def _print_table(title: str, ranked: list[Ranked]) -> None:
    print(f"{BLUE}{title}{RESET}")
    print(f"  {'Competition':<50} {'Teams':>6}  {'Days':>5}  {'Score':>5}")
    for r in ranked[:10]:
        colour = GREEN if r.score >= 75 else (YELLOW if r.score >= 60 else RESET)
        print(
            f"  {r.competition.slug[:48]:<50} {r.competition.team_count:>6}  "
            f"{r.days_left:>5}  {colour}{r.score:>5.0f}{RESET}"
        )
    if not ranked:
        print("  (none)")
    print("")


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update", action="store_true", help="Rewrite competition-scout-report.md"
    )
    parser.add_argument("--today", default=None, help="Override today (YYYY-MM-DD).")
    args = parser.parse_args(argv)

    deps = deps or Deps.resolve(today=args.today)
    now = deps.clock.now()
    competitions = fetch_competitions(deps.client)
    if not competitions:
        print(
            f"{RED}Failed to fetch competitions. Check kaggle CLI credentials.{RESET}"
        )
        return 1

    medals = rank(competitions, now, medal_score)
    notebooks = rank(competitions, now, notebook_score)
    _print_table("Competition medals (Featured/Research only)", medals)
    _print_table("Notebook votes (any active board)", notebooks)

    if args.update:
        target = deps.layout.scout_report
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(format_report(medals, notebooks, now), encoding="utf-8")
        print(f"{GREEN}Updated {target.name}{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
