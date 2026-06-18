#!/usr/bin/env python3
"""Plan which datasets to push for the next Kaggle medal, and how to drive votes.

Read-only. Scores each local dataset on *vote-readiness*: distance to the next
dataset medal (bronze >= 5 / silver >= 20 / gold >= 50 votes), keyword /
discoverability gaps, starter-notebook and cover presence, and the
download-to-vote conversion rate. Then ranks the datasets and prints a
prioritized action plan — the "what do I fix to earn the next medal" view that
the usability rubric (which maxes out) does not give.

Usage:
    python3 -m kaggle_portfolio.datasets.dataset_vote_planner            # live stats
    python3 -m kaggle_portfolio.datasets.dataset_vote_planner --json
    python3 -m kaggle_portfolio.datasets.dataset_vote_planner --owner lorenzoscaturchio

Invoked by: ./manage.sh vote-plan [--owner OWNER] [--json]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from kaggle_portfolio.shared.kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
MEDALS = [(5, "bronze"), (20, "silver"), (50, "gold")]
MIN_KEYWORDS = 8
LOW_CONVERSION_PCT = 3.0

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
RESET = "\033[0m"


def next_medal(votes: int) -> tuple[int | None, str, int]:
    """Return (threshold, name, votes_needed) for the next dataset medal.

    Returns (None, 'gold (max)', 0) when the dataset already has a gold medal.
    """
    for threshold, name in MEDALS:
        if votes < threshold:
            return threshold, name, threshold - votes
    return None, "gold (max)", 0


def slug_of(meta: dict) -> str:
    return str(meta.get("id", "")).split("/")[-1]


def discover_datasets() -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    if not DATASETS_DIR.exists():
        return out
    for meta_path in sorted(DATASETS_DIR.glob("*/dataset-metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append((meta_path.parent, meta))
    return out


def score_dataset(dir_path: Path, meta: dict, votes: int, downloads: int) -> dict[str, Any]:
    threshold, medal_name, votes_needed = next_medal(votes)
    keywords = meta.get("keywords") or []
    has_notebook = (dir_path / "explore.ipynb").exists()
    has_cover = (dir_path / "cover.png").exists()
    conversion = (votes / downloads * 100) if downloads else 0.0

    actions: list[str] = []
    if threshold is not None:
        actions.append(f"{votes_needed} more vote(s) -> {medal_name} ({threshold})")
    if len(keywords) < MIN_KEYWORDS:
        actions.append(f"add {MIN_KEYWORDS - len(keywords)} keyword(s) for search (have {len(keywords)})")
    if not has_notebook:
        actions.append("attach a starter EDA notebook (explore.ipynb)")
    if not has_cover:
        actions.append("add a cover image (cover.png)")
    if downloads >= 50 and conversion < LOW_CONVERSION_PCT:
        actions.append(
            f"high downloads ({downloads}) but low conversion ({conversion:.1f}%) "
            "- cross-promote in a discussion post"
        )

    return {
        "slug": slug_of(meta),
        "title": meta.get("title", ""),
        "votes": votes,
        "downloads": downloads,
        "next_medal": medal_name,
        "votes_to_next": votes_needed,
        "keyword_count": len(keywords),
        "has_starter_notebook": has_notebook,
        "has_cover": has_cover,
        "conversion_pct": round(conversion, 2),
        "actions": actions,
    }


def fetch_live_stats(owner: str | None) -> dict[str, dict[str, int]]:
    """Best-effort slug -> {'votes','downloads'} via the kaggle CLI; {} on failure."""
    args = ["datasets", "list", "--csv"]
    args += ["--user", owner] if owner else ["--mine"]
    try:
        result = subprocess.run([*kaggle_command(), *args], capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{YELLOW}live stats unavailable{RESET}: {exc}", file=sys.stderr)
        return {}
    if result.returncode != 0:
        print(f"{YELLOW}live stats unavailable{RESET}: "
              f"{summarize_subprocess_error(result.stdout, result.stderr)}", file=sys.stderr)
        return {}

    def _int(row: dict, key: str) -> int:
        try:
            return int(row.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    # The kaggle CLI may prepend a "Warning: ... outdated version" line to stdout;
    # start parsing at the real CSV header so DictReader reads the right columns.
    lines = result.stdout.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("ref,")), 0)
    stats: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header_idx:]))):
        slug = (row.get("ref") or "").split("/")[-1]
        if slug:
            stats[slug] = {"votes": _int(row, "voteCount"), "downloads": _int(row, "downloadCount")}
    return stats


def build_plan(datasets: list[tuple[Path, dict]], stats: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    for dir_path, meta in datasets:
        s = stats.get(slug_of(meta), {"votes": 0, "downloads": 0})
        rows.append(score_dataset(dir_path, meta, s["votes"], s["downloads"]))
    # Priority: still-medalable first, then fewest votes to the next medal, then most downloads.
    rows.sort(key=lambda r: (r["votes_to_next"] == 0, r["votes_to_next"], -r["downloads"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan dataset votes toward the next medal.")
    parser.add_argument("--owner", default=None, help="Kaggle owner slug (default: your own datasets).")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    plan = build_plan(discover_datasets(), fetch_live_stats(args.owner))
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0

    has_live = any(r["votes"] or r["downloads"] for r in plan)
    print(f"Dataset vote plan ({len(plan)} datasets){'' if has_live else '  [no live stats - run with creds]'}\n")
    for r in plan:
        marker = f"{GREEN}[gold]{RESET}" if r["votes_to_next"] == 0 else f"-> {r['next_medal']}"
        print(f"{r['slug']:<30} votes={r['votes']:<4} dl={r['downloads']:<6} {marker}")
        for action in r["actions"]:
            print(f"    - {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
