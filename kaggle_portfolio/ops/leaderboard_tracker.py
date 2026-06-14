#!/usr/bin/env python3
"""Record the owner's Kaggle competition leaderboard rank/percentile over time.

Read-only telemetry: reads public leaderboards via the kaggle CLI and writes
timestamped standings snapshots to medal_ops/leaderboard/. It NEVER submits.

Usage
-----
    python3 -m kaggle_portfolio.ops.leaderboard_tracker record            # fetch + record
    python3 -m kaggle_portfolio.ops.leaderboard_tracker record --dry-run  # preview, no write
    python3 -m kaggle_portfolio.ops.leaderboard_tracker report            # latest + rank deltas
    python3 -m kaggle_portfolio.ops.leaderboard_tracker report --json     # machine-readable

Invoked by: ./manage.sh leaderboard <record|report> [args...]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_portfolio.ops.kaggle_auth_doctor import resolve_credentials
from kaggle_portfolio.shared.kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD_DIR = ROOT / "medal_ops" / "leaderboard"
BRONZE_TOP_FRACTION = 0.40  # approximate bronze zone: top 40% of teams

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
RESET = "\033[0m"


def parse_leaderboard_csv(text: str) -> list[dict[str, str]]:
    """Parse `kaggle competitions leaderboard --show --csv` output into ordered rows.

    The CLI prepends a 'Next Page Token = ...' line before each CSV header; those
    lines are stripped so the remainder parses as clean CSV.
    """
    lines = [ln for ln in text.splitlines() if not ln.startswith("Next Page Token")]
    if not lines:
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO("\n".join(lines)))]


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def compute_standing(
    rows: list[dict[str, str]],
    owner: str,
    *,
    owner_scores: set[float] | None = None,
    team_count: int | None = None,
) -> dict[str, Any]:
    """Find the owner's rank within ordered leaderboard rows.

    Matches by teamName (case-insensitive) first; if not found and owner_scores is
    given, matches the first row whose score is one of the owner's submission scores.
    """
    owner_l = (owner or "").strip().lower()
    rank: int | None = None
    matched_score: float | None = None

    for idx, row in enumerate(rows, start=1):
        if owner_l and str(row.get("teamName", "")).strip().lower() == owner_l:
            rank = idx
            matched_score = _to_float(row.get("score"))
            break

    if rank is None and owner_scores:
        for idx, row in enumerate(rows, start=1):
            score = _to_float(row.get("score"))
            if score is not None and score in owner_scores:
                rank = idx
                matched_score = score
                break

    total = team_count if isinstance(team_count, int) and team_count > 0 else (len(rows) or None)
    percentile: float | None = None
    top_fraction: float | None = None
    in_bronze = False
    if rank is not None and total:
        top_fraction = rank / total
        percentile = round((total - rank + 1) / total * 100, 1)
        in_bronze = top_fraction <= BRONZE_TOP_FRACTION

    return {
        "rank": rank,
        "team_count": total,
        "percentile": percentile,
        "top_fraction": round(top_fraction, 4) if top_fraction is not None else None,
        "score": matched_score,
        "in_bronze_zone": in_bronze,
    }


def _run_csv(args: list[str]) -> str | None:
    """Run a kaggle CLI command; return stdout, or None on failure (logged to stderr)."""
    try:
        result = subprocess.run([*kaggle_command(), *args], capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{YELLOW}kaggle call failed{RESET}: {exc}", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(
            f"{YELLOW}kaggle call failed{RESET}: "
            f"{summarize_subprocess_error(result.stdout, result.stderr)}",
            file=sys.stderr,
        )
        return None
    return result.stdout


def fetch_entered_competitions() -> list[dict[str, Any]]:
    """Return [{'slug':..., 'team_count': int|None}, ...] for entered competitions."""
    out = _run_csv(["competitions", "list", "--group", "entered", "--csv"])
    if out is None:
        return []
    comps: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(out)):
        ref = (row.get("ref") or "").strip()
        slug = ref.rsplit("/", 1)[-1] if ref else ""
        if not slug:
            continue
        try:
            team_count: int | None = int(row.get("teamCount") or 0) or None
        except (TypeError, ValueError):
            team_count = None
        comps.append({"slug": slug, "team_count": team_count})
    return comps


def fetch_leaderboard_rows(slug: str, *, page_size: int = 200) -> list[dict[str, str]]:
    """Fetch ordered public-leaderboard rows for a competition."""
    out = _run_csv(["competitions", "leaderboard", slug, "--show", "--csv", "--page-size", str(page_size)])
    return parse_leaderboard_csv(out) if out else []


def fetch_owner_scores(slug: str) -> set[float]:
    """Return the set of the owner's public submission scores for a competition."""
    out = _run_csv(["competitions", "submissions", slug, "--csv"])
    if out is None:
        return set()
    scores: set[float] = set()
    for row in csv.DictReader(io.StringIO(out)):
        score = _to_float(row.get("publicScore"))
        if score is not None:
            scores.add(score)
    return scores


def build_standings(owner: str, competitions: list[dict[str, Any]], *, today: date | None = None) -> dict[str, Any]:
    standings: list[dict[str, Any]] = []
    for comp in competitions:
        slug = comp["slug"]
        rows = fetch_leaderboard_rows(slug)
        standing = compute_standing(
            rows, owner, owner_scores=fetch_owner_scores(slug), team_count=comp.get("team_count")
        )
        standing["competition"] = slug
        standings.append(standing)
    return {
        "generated_on": (today or datetime.now(tz=timezone.utc).date()).isoformat(),
        "owner": owner,
        "standings": standings,
    }


def write_standings(snapshot: dict[str, Any], history_dir: Path | None = None) -> Path:
    history_dir = history_dir or LEADERBOARD_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = history_dir / f"leaderboard-{stamp}.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return path


def load_all_standings(history_dir: Path | None = None) -> list[dict[str, Any]]:
    history_dir = history_dir or LEADERBOARD_DIR
    if not history_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(history_dir.glob("leaderboard-*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build_report(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure report builder: latest standings + rank delta vs the previous snapshot."""
    if not history:
        return {"generated_on": None, "competitions": []}
    latest = history[-1]
    previous = history[-2] if len(history) >= 2 else None
    prev_map = {s["competition"]: s for s in previous.get("standings", [])} if previous else {}
    comps: list[dict[str, Any]] = []
    for standing in latest.get("standings", []):
        prev = prev_map.get(standing["competition"], {})
        rank_delta = None
        if isinstance(standing.get("rank"), int) and isinstance(prev.get("rank"), int):
            rank_delta = prev["rank"] - standing["rank"]  # positive = moved up
        comps.append({**standing, "rank_delta": rank_delta})
    return {"generated_on": latest.get("generated_on"), "competitions": comps}


def cmd_record(dry_run: bool = False) -> int:
    creds, err = resolve_credentials()
    owner = creds.username if creds else None
    if not owner:
        print(f"{RED}Cannot resolve Kaggle username{RESET}: {err or 'no credentials'}", file=sys.stderr)
        return 1
    competitions = fetch_entered_competitions()
    if not competitions:
        print(f"{YELLOW}No entered competitions found (or kaggle CLI unavailable).{RESET}")
        return 0
    snapshot = build_standings(owner, competitions)
    ranked = [s for s in snapshot["standings"] if s.get("rank")]
    if dry_run:
        print(f"{YELLOW}DRY RUN{RESET} — {len(snapshot['standings'])} competitions, {len(ranked)} ranked")
        for standing in snapshot["standings"]:
            print(f"  {standing['competition']}: rank={standing['rank']} "
                  f"of {standing['team_count']} ({standing['percentile']}%)")
        return 0
    path = write_standings(snapshot)
    print(f"{GREEN}Recorded{RESET} {len(snapshot['standings'])} competitions "
          f"({len(ranked)} ranked) -> {path}")
    return 0


def cmd_report(as_json: bool = False) -> int:
    history = load_all_standings()
    report = build_report(history)
    if not history:
        print("No leaderboard history yet. Run `leaderboard record` first.")
        return 0
    if as_json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"Leaderboard standings — {report['generated_on']}")
    for comp in report["competitions"]:
        rank = comp["rank"] if comp["rank"] is not None else "—"
        pct = f"{comp['percentile']}%" if comp["percentile"] is not None else "n/a"
        delta = ""
        if isinstance(comp.get("rank_delta"), int) and comp["rank_delta"] != 0:
            arrow = "▲" if comp["rank_delta"] > 0 else "▼"
            delta = f" {arrow}{abs(comp['rank_delta'])}"
        zone = " [bronze zone]" if comp.get("in_bronze_zone") else ""
        print(f"  {comp['competition']}: rank {rank}/{comp['team_count']} ({pct}){delta}{zone}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track competition leaderboard rank/percentile (read-only; never submits)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Fetch + record current standings for entered competitions.")
    rec.add_argument("--dry-run", action="store_true", help="Preview without writing a snapshot.")
    rep = sub.add_parser("report", help="Show latest standings + rank deltas.")
    rep.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)
    if args.command == "record":
        return cmd_record(dry_run=args.dry_run)
    if args.command == "report":
        return cmd_report(as_json=args.json)
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
