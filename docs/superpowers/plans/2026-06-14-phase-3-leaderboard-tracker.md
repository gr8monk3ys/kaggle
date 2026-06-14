# Phase 3 (telemetry slice) — Leaderboard Rank Tracker Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Give the owner read-only visibility into where their submissions land — record each entered competition's rank / team-count / percentile / distance-to-bronze over time, mirroring the Phase 1 snapshot-history pattern. This is the non-gated, read-only slice of Phase 3 (it never submits).

**Architecture:** A self-contained module `kaggle_portfolio/ops/leaderboard_tracker.py` with pure parsing/ranking functions (fully unit-tested) and thin kaggle-CLI I/O wrappers (mocked in tests). Standings snapshots are written to `medal_ops/leaderboard/` (committed, like `medal_ops/history/`). Exposed via `./manage.sh leaderboard`.

**Tech Stack:** Python 3.14 / pytest 9; kaggle CLI 2.2.1 (in `.venv`); conventional commits.

**Verified integration facts:**
- `kaggle competitions list --group entered --csv` → columns `ref,deadline,category,reward,teamCount,userHasEntered`; slug = `ref.rsplit("/",1)[-1]`; `teamCount` = total teams.
- `kaggle competitions leaderboard <slug> --show --csv --page-size 200` → ordered rows `teamId,teamName,submissionDate,score`, each page **prefixed by** a `Next Page Token = ...` line that must be stripped.
- `kaggle competitions submissions <slug> --csv` → `...,publicScore,privateScore`; owner's scores from `publicScore`.
- Owner username via `kaggle_auth_doctor.resolve_credentials()` → `(Credentials(username=...), err)`.
- CLI access via `kaggle_portfolio.shared.kaggle_utils.kaggle_command()` + `summarize_subprocess_error()`.
- Bronze is approximated as **top 40% of teams** (Kaggle's exact tier table is team-count-dependent and volatile; top-40% is a documented heuristic, clearly labelled).

**Test runner:** `.venv/bin/python -m pytest`.

---

### Task 1: Build `leaderboard_tracker.py` + its tests

**Files:**
- Create: `kaggle_portfolio/ops/leaderboard_tracker.py`
- Create: `tests/test_leaderboard_tracker.py`

- [ ] **Step 1: Write the test file (it fails because the module doesn't exist yet)**

Create `tests/test_leaderboard_tracker.py`:

```python
import json
from types import SimpleNamespace

from kaggle_portfolio.ops import leaderboard_tracker as lb

LEADERBOARD_CSV = (
    "Next Page Token = ABC123\n"
    "teamId,teamName,submissionDate,score\n"
    "1,alpha,2026-06-10,0.95\n"
    "2,lorenzoscaturchio,2026-06-11,0.90\n"
    "3,gamma,2026-06-12,0.80\n"
)


class TestParse:
    def test_strips_page_token_and_parses(self):
        rows = lb.parse_leaderboard_csv(LEADERBOARD_CSV)
        assert len(rows) == 3
        assert rows[1]["teamName"] == "lorenzoscaturchio"
        assert rows[0]["score"] == "0.95"

    def test_empty(self):
        assert lb.parse_leaderboard_csv("") == []
        assert lb.parse_leaderboard_csv("Next Page Token = X\n") == []


class TestComputeStanding:
    def setup_method(self):
        self.rows = lb.parse_leaderboard_csv(LEADERBOARD_CSV)

    def test_name_match_rank_and_percentile(self):
        s = lb.compute_standing(self.rows, "lorenzoscaturchio", team_count=3)
        assert s["rank"] == 2
        assert s["team_count"] == 3
        assert s["percentile"] == 66.7
        assert s["score"] == 0.90
        assert s["in_bronze_zone"] is False

    def test_case_insensitive_name(self):
        assert lb.compute_standing(self.rows, "LorenzoScaturchio", team_count=3)["rank"] == 2

    def test_score_fallback_when_name_differs(self):
        s = lb.compute_standing(self.rows, "display-name", owner_scores={0.80}, team_count=3)
        assert s["rank"] == 3
        assert s["score"] == 0.80

    def test_not_found(self):
        s = lb.compute_standing(self.rows, "nobody", team_count=3)
        assert s["rank"] is None
        assert s["percentile"] is None
        assert s["in_bronze_zone"] is False

    def test_bronze_zone_top_fraction(self):
        s = lb.compute_standing(self.rows, "alpha", team_count=100)
        assert s["rank"] == 1
        assert s["in_bronze_zone"] is True

    def test_team_count_defaults_to_row_count(self):
        s = lb.compute_standing(self.rows, "alpha")
        assert s["team_count"] == 3
        assert s["rank"] == 1


class TestFetchers:
    def test_fetch_entered_parses_slug_and_teamcount(self, monkeypatch):
        csv_out = (
            "ref,deadline,category,reward,teamCount,userHasEntered\n"
            "https://www.kaggle.com/competitions/hull-tactical,2026-09-01,Featured,$,1200,True\n"
            "https://www.kaggle.com/competitions/orbit-wars,2026-07-15,Research,Swag,45,True\n"
        )
        monkeypatch.setattr(lb, "_run_csv", lambda args: csv_out)
        assert lb.fetch_entered_competitions() == [
            {"slug": "hull-tactical", "team_count": 1200},
            {"slug": "orbit-wars", "team_count": 45},
        ]

    def test_fetch_entered_handles_cli_failure(self, monkeypatch):
        monkeypatch.setattr(lb, "_run_csv", lambda args: None)
        assert lb.fetch_entered_competitions() == []

    def test_fetch_owner_scores(self, monkeypatch):
        csv_out = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "x,sub1.csv,2026-06-10,first,SubmissionStatus.COMPLETE,0.90,0.91\n"
            "x,sub2.csv,2026-06-11,second,SubmissionStatus.COMPLETE,0.88,\n"
        )
        monkeypatch.setattr(lb, "_run_csv", lambda args: csv_out)
        assert lb.fetch_owner_scores("slug") == {0.90, 0.88}


class TestRecordAndReport:
    def test_cmd_record_dry_run(self, monkeypatch, capsys):
        monkeypatch.setattr(lb, "resolve_credentials",
                            lambda: (SimpleNamespace(username="lorenzoscaturchio"), None))
        monkeypatch.setattr(lb, "fetch_entered_competitions",
                            lambda: [{"slug": "hull-tactical", "team_count": 3}])
        monkeypatch.setattr(lb, "fetch_leaderboard_rows",
                            lambda slug, **k: lb.parse_leaderboard_csv(LEADERBOARD_CSV))
        monkeypatch.setattr(lb, "fetch_owner_scores", lambda slug: set())
        rc = lb.cmd_record(dry_run=True)
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY RUN" in out
        assert "hull-tactical" in out and "rank=2" in out

    def test_cmd_record_writes_snapshot(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lb, "LEADERBOARD_DIR", tmp_path)
        monkeypatch.setattr(lb, "resolve_credentials",
                            lambda: (SimpleNamespace(username="alpha"), None))
        monkeypatch.setattr(lb, "fetch_entered_competitions",
                            lambda: [{"slug": "c1", "team_count": 3}])
        monkeypatch.setattr(lb, "fetch_leaderboard_rows",
                            lambda slug, **k: lb.parse_leaderboard_csv(LEADERBOARD_CSV))
        monkeypatch.setattr(lb, "fetch_owner_scores", lambda slug: set())
        rc = lb.cmd_record()
        assert rc == 0
        files = list(tmp_path.glob("leaderboard-*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["owner"] == "alpha"
        assert data["standings"][0]["rank"] == 1

    def test_cmd_record_no_credentials(self, monkeypatch):
        monkeypatch.setattr(lb, "resolve_credentials", lambda: (None, "no creds"))
        assert lb.cmd_record() == 1

    def test_build_report_rank_delta(self):
        h = [
            {"generated_on": "2026-06-13",
             "standings": [{"competition": "c1", "rank": 5, "team_count": 100,
                            "percentile": 96.0, "in_bronze_zone": True}]},
            {"generated_on": "2026-06-14",
             "standings": [{"competition": "c1", "rank": 3, "team_count": 100,
                            "percentile": 98.0, "in_bronze_zone": True}]},
        ]
        rep = lb.build_report(h)
        assert rep["generated_on"] == "2026-06-14"
        assert rep["competitions"][0]["rank_delta"] == 2


class TestMain:
    def test_main_record_dispatch(self, monkeypatch):
        called = {}
        monkeypatch.setattr(lb, "cmd_record", lambda dry_run: called.setdefault("record", dry_run) or 0)
        assert lb.main(["record", "--dry-run"]) == 0
        assert called["record"] is True

    def test_main_report_dispatch(self, monkeypatch):
        monkeypatch.setattr(lb, "cmd_report", lambda as_json: 0)
        assert lb.main(["report"]) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_leaderboard_tracker.py -q`
Expected: collection error / FAIL — `leaderboard_tracker` does not exist.

- [ ] **Step 3: Create the module**

Create `kaggle_portfolio/ops/leaderboard_tracker.py` with EXACTLY this content:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_leaderboard_tracker.py -q`
Expected: PASS (all classes).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/ops/leaderboard_tracker.py tests/test_leaderboard_tracker.py
git commit -m "feat: add read-only leaderboard rank tracker (record + report)"
```

---

### Task 2: Expose `./manage.sh leaderboard`

**Files:**
- Modify: `kaggle_portfolio/manage_commands.py`
- Test: `tests/test_manage_commands.py` (append one function)

- [ ] **Step 1: Read the pattern, then write the failing test**

Read `manage_commands.py` for the `Command(...)` shape (it has `requires_kaggle`). Append to `tests/test_manage_commands.py`:

```python
def test_leaderboard_command_is_registered():
    from kaggle_portfolio import manage_commands
    names = [c.name for c in manage_commands.COMMANDS] if hasattr(manage_commands, "COMMANDS") \
        else list(manage_commands.command_table().keys())
    assert "leaderboard" in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manage_commands.py::test_leaderboard_command_is_registered -v`
Expected: FAIL.

- [ ] **Step 3: Register the command**

Add to the `COMMANDS` table (matching the surrounding `Command(...)` form), marked `requires_kaggle=True` since it reads live leaderboards:

```python
    Command(
        "leaderboard",
        "Record/report competition leaderboard rank history",
        lambda a: run_module("kaggle_portfolio.ops.leaderboard_tracker", a),
        "<record|report> [--dry-run] [--json]",
        requires_kaggle=True,
    ),
```

Match the real `Command` field order/keywords used by other `requires_kaggle=True` entries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manage_commands.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/manage_commands.py tests/test_manage_commands.py
git commit -m "feat: expose leaderboard tracker via manage.sh"
```

---

### Task 3: Full-suite verification

- [ ] Run `.venv/bin/python -m pytest tests -q` and `.venv/bin/python -m pytest pi-automation/tests -q` → both green.
- [ ] Smoke (offline-safe): `.venv/bin/python -m kaggle_portfolio.ops.leaderboard_tracker report` → prints "No leaderboard history yet…" without error.
- [ ] `git diff --stat main..HEAD` → only the module, its test, manage_commands.py, test_manage_commands.py, and this plan doc.

## Self-Review
Pure functions (`parse_leaderboard_csv`, `compute_standing`, `build_report`) are fully unit-tested; I/O fetchers are tested with mocked `_run_csv`; `cmd_record`/`main` tested with mocked fetchers and `tmp_path`. No live network in tests. The module never calls submit. `medal_ops/leaderboard/` is committed by default (consistent with `history/`), so recorded standings persist for trend tracking.
