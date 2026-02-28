import os
from datetime import date

import medal_ops
import pytest


SAMPLE_TRACKER = """
# Kaggle Grandmaster Tracker
**Last Updated:** 2026-01-25

## Tier Requirements
| Category | Novice | Contributor | Expert | Master | Grandmaster |
|----------|--------|-------------|--------|--------|-------------|
| **Competitions** | 0 | 0 | 2 bronze | 1 gold + 2 silver | 5 gold (1 solo) |
| **Datasets** | 0 | 0 | 3 bronze | 5 silver | 5 gold |
| **Notebooks** | 0 | 0 | 5 bronze | 10 silver | 15 gold |
| **Discussion** | 0 | 0 | 50 bronze | 50 silver + 200 total | 50 gold + 500 total |

## Current Progress

### Competitions
| Status | Target | Current |
|--------|--------|---------|
| Tier | Grandmaster (5 gold, 1 solo) | Novice |
| Gold medals | 5 | 0 |
| Silver medals | — | 0 |
| Bronze medals | — | 0 |
| Entered | — | 1 (Titanic) |

**Active competitions to enter:**
| Competition | Teams | Deadline | Medal Difficulty | Strategy |
|-------------|-------|----------|-----------------|----------|
| Med-Gemma Impact Challenge | 58 | Feb 24, 2026 | Easiest | Fine-tune MedGemma |
| Vesuvius Surface Detection | 759 | Feb 13, 2026 | Medium | 3D U-Net baseline |

### Notebooks
| Status | Target | Current |
|--------|--------|---------|
| Tier | Grandmaster (15 gold) | Novice |
| Total notebooks | 26+ | 21 (on Kaggle) |
| Gold medals (50+ votes) | 15 | 0 |
| Silver medals (20+ votes) | — | 0 |
| Bronze medals (5+ votes) | — | 0 |
| Total votes | — | 4 |

### Datasets
| Status | Target | Current |
|--------|--------|---------|
| Tier | Grandmaster (5 gold) | Novice |
| Total datasets | 6+ | 4 (on Kaggle) |
| Gold medals (50+ votes) | 5 | 0 |
| Bronze medals (5+ votes) | — | 0 |
| Total votes | — | 0 |

### Discussion
| Status | Target | Current |
|--------|--------|---------|
| Tier | Grandmaster (50 gold + 500 total) | Novice |
| Total posts | 500+ | 0 |
| Gold medals (10+ votes) | 50 | 0 |
| Silver medals (5+ votes) | — | 0 |
| Bronze medals (1+ vote) | — | 0 |
"""


def test_build_snapshot_extracts_core_metrics():
    snapshot = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 22))

    assert snapshot["tracker_last_updated"] == "2026-01-25"
    assert snapshot["tracker_stale_days"] == 28
    assert snapshot["categories"]["competitions"]["entered"] == 1
    assert snapshot["categories"]["competitions"]["gold_goal"] == 5
    assert snapshot["categories"]["notebooks"]["total_votes"] == 4
    assert snapshot["categories"]["discussion"]["gold_goal"] == 50


def test_competition_deadline_parsing_and_sorting():
    snapshot = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 22))
    by_name = {item["competition"]: item for item in snapshot["active_competitions"]}

    assert by_name["Med-Gemma Impact Challenge"]["days_to_deadline"] == 2
    assert by_name["Vesuvius Surface Detection"]["days_to_deadline"] == -9


def test_scorecard_markdown_contains_actionable_sections():
    snapshot = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 22))
    md = medal_ops.generate_scorecard_markdown(snapshot, previous=None)

    assert "Kaggle Medal Ops Scorecard" in md
    assert "Deadline Radar" in md
    assert "Top Actions" in md
    assert "stale" in md


def test_weekly_plan_contains_kpis_and_cadence():
    snapshot = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 22))
    md = medal_ops.generate_weekly_plan_markdown(snapshot)

    assert "Primary Objectives (7 days)" in md
    assert "Daily Cadence" in md
    assert "KPI Targets" in md


def test_pace_report_computes_velocity():
    first = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 15))
    second = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 22))
    second["categories"]["notebooks"]["total_votes"] = 14
    second["categories"]["datasets"]["total_votes"] = 5
    second["categories"]["discussion"]["total_posts"] = 14
    second["categories"]["competitions"]["entered"] = 3

    md = medal_ops.generate_pace_markdown([first, second])

    assert "Kaggle Medal Ops Pace Analysis" in md
    assert "Outcome Pace" in md
    assert "Leading Indicators" in md
    assert "Notebook votes velocity" in md
    assert "/wk" in md


def test_write_snapshot_dedupes_same_day_state(tmp_path):
    history = tmp_path / "history"
    snapshot = medal_ops.build_snapshot(SAMPLE_TRACKER, today=date(2026, 2, 22))
    first_path = medal_ops.write_snapshot(history, snapshot)
    second_path = medal_ops.write_snapshot(history, snapshot)

    assert first_path == second_path


def test_apply_tracker_sync_updates_summary_metrics():
    live = {
        "notebooks_count": 42,
        "notebooks_total_votes": 77,
        "notebooks_vote_key": "totalVotes",
        "datasets_count": 12,
        "datasets_total_votes": 34,
        "datasets_vote_key": "voteCount",
        "competitions_entered": 5,
        "competitions_entered_key": "userHasEntered",
    }
    updated, changes = medal_ops.apply_tracker_sync(
        SAMPLE_TRACKER, today=date(2026, 2, 22), live=live
    )

    notebooks = medal_ops.parse_progress_metrics(updated, "Notebooks")
    datasets = medal_ops.parse_progress_metrics(updated, "Datasets")
    competitions = medal_ops.parse_progress_metrics(updated, "Competitions")

    assert notebooks["total_notebooks"] == 42
    assert notebooks["total_votes"] == 77
    assert datasets["total_datasets"] == 12
    assert datasets["total_votes"] == 34
    assert competitions["entered"] == 5
    assert "Notebooks.Total votes" in changes["changed_fields"]
    assert "Datasets.Total votes" in changes["changed_fields"]


def test_generate_sync_markdown_includes_change_summary():
    live = {
        "notebooks_count": 10,
        "notebooks_total_votes": 20,
        "notebooks_vote_key": "totalVotes",
        "datasets_count": 5,
        "datasets_total_votes": 6,
        "datasets_vote_key": "voteCount",
        "competitions_entered": None,
        "competitions_entered_key": None,
    }
    _, changes = medal_ops.apply_tracker_sync(
        SAMPLE_TRACKER, today=date(2026, 2, 22), live=live
    )
    md = medal_ops.generate_sync_markdown(
        tracker_path=medal_ops.DEFAULT_TRACKER_PATH,
        today=date(2026, 2, 22),
        live=live,
        changes=changes,
        dry_run=True,
    )

    assert "Kaggle Medal Ops Sync Report" in md
    assert "Mode: dry-run" in md
    assert "Changed fields" in md
    assert "Notebook votes pulled" in md


def test_fetch_metrics_from_csv(tmp_path):
    kernels_csv = tmp_path / "kernels.csv"
    datasets_csv = tmp_path / "datasets.csv"
    competitions_csv = tmp_path / "competitions.csv"

    kernels_csv.write_text(
        "title,totalVotes\nA,10\nB,3\n",
        encoding="utf-8",
    )
    datasets_csv.write_text(
        "title,voteCount\nD1,2\nD2,7\n",
        encoding="utf-8",
    )
    competitions_csv.write_text(
        "competition,userHasEntered\nC1,true\nC2,false\nC3,1\n",
        encoding="utf-8",
    )

    live = medal_ops.fetch_metrics_from_csv(kernels_csv, datasets_csv, competitions_csv)

    assert live["notebooks_count"] == 2
    assert live["notebooks_total_votes"] == 13
    assert live["datasets_count"] == 2
    assert live["datasets_total_votes"] == 9
    assert live["competitions_entered"] == 2


def test_fetch_metrics_from_csv_requires_vote_column(tmp_path):
    kernels_csv = tmp_path / "kernels.csv"
    datasets_csv = tmp_path / "datasets.csv"

    kernels_csv.write_text(
        "title,score\nA,10\n",
        encoding="utf-8",
    )
    datasets_csv.write_text(
        "title,voteCount\nD1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing a vote column"):
        medal_ops.fetch_metrics_from_csv(kernels_csv, datasets_csv, competitions_csv=None)


def test_fetch_metrics_from_csv_requires_entered_column_when_competitions_present(tmp_path):
    kernels_csv = tmp_path / "kernels.csv"
    datasets_csv = tmp_path / "datasets.csv"
    competitions_csv = tmp_path / "competitions.csv"

    kernels_csv.write_text(
        "title,totalVotes\nA,10\n",
        encoding="utf-8",
    )
    datasets_csv.write_text(
        "title,voteCount\nD1,2\n",
        encoding="utf-8",
    )
    competitions_csv.write_text(
        "competition,teams\nC1,100\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing an entered column"):
        medal_ops.fetch_metrics_from_csv(kernels_csv, datasets_csv, competitions_csv=competitions_csv)


def test_generate_sync_template_assets(tmp_path):
    out_dir = tmp_path / "sync_inputs"

    statuses = medal_ops.generate_sync_template_assets(out_dir, force=False)

    kernels_path = out_dir / "kernels.csv"
    datasets_path = out_dir / "datasets.csv"
    competitions_path = out_dir / "competitions.csv"
    script_path = out_dir / "export_kaggle_sync_csv.sh"
    readme_path = out_dir / "README.md"

    assert kernels_path.exists()
    assert datasets_path.exists()
    assert competitions_path.exists()
    assert script_path.exists()
    assert readme_path.exists()
    assert os.access(script_path, os.X_OK)
    assert statuses[str(kernels_path)] == "created"

    rerun_statuses = medal_ops.generate_sync_template_assets(out_dir, force=False)
    assert rerun_statuses[str(kernels_path)] == "skipped"


def test_run_preflight_checks_validates_csv_bundle(tmp_path):
    tracker_path = tmp_path / "tracker.md"
    output_root = tmp_path / "out"
    kernels_csv = tmp_path / "kernels.csv"
    datasets_csv = tmp_path / "datasets.csv"
    competitions_csv = tmp_path / "competitions.csv"

    tracker_path.write_text(SAMPLE_TRACKER, encoding="utf-8")
    kernels_csv.write_text("title,totalVotes\nA,10\n", encoding="utf-8")
    datasets_csv.write_text("title,voteCount\nD1,2\n", encoding="utf-8")
    competitions_csv.write_text("competition,userHasEntered\nC1,true\n", encoding="utf-8")

    checks = medal_ops.run_preflight_checks(
        tracker_path=tracker_path,
        output_root=output_root,
        today=date(2026, 1, 25),
        kernels_csv=kernels_csv,
        datasets_csv=datasets_csv,
        competitions_csv=competitions_csv,
        require_kaggle=False,
        max_stale_days=7,
    )

    assert checks["errors"] == []
    assert any("CSV sync inputs validated" in item for item in checks["infos"])
    assert checks["csv_metrics"]["notebooks_total_votes"] == 10


def test_run_preflight_checks_reports_missing_tracker(tmp_path):
    checks = medal_ops.run_preflight_checks(
        tracker_path=tmp_path / "missing-tracker.md",
        output_root=tmp_path / "out",
        today=date(2026, 2, 22),
        kernels_csv=None,
        datasets_csv=None,
        competitions_csv=None,
        require_kaggle=False,
        max_stale_days=7,
    )

    assert any("Tracker file not found" in item for item in checks["errors"])


def test_generate_doctor_markdown_contains_status():
    checks = {
        "errors": ["Tracker file not found: grandmaster-tracker.md"],
        "warnings": ["kaggle CLI not found; live sync is unavailable."],
        "infos": ["Output root writable: medal_ops"],
    }

    md = medal_ops.generate_doctor_markdown(
        tracker_path=medal_ops.DEFAULT_TRACKER_PATH,
        output_root=medal_ops.DEFAULT_OUTPUT_ROOT,
        today=date(2026, 2, 22),
        checks=checks,
        strict=True,
        max_stale_days=7,
    )

    assert "Kaggle Medal Ops Preflight Report" in md
    assert "Status: BLOCKED" in md
    assert "Strict mode: enabled" in md
    assert "Max stale days: 7" in md


def test_run_preflight_checks_respects_max_stale_days(tmp_path):
    tracker_path = tmp_path / "tracker.md"
    tracker_path.write_text(SAMPLE_TRACKER, encoding="utf-8")

    checks = medal_ops.run_preflight_checks(
        tracker_path=tracker_path,
        output_root=tmp_path / "out",
        today=date(2026, 2, 22),
        kernels_csv=None,
        datasets_csv=None,
        competitions_csv=None,
        require_kaggle=False,
        max_stale_days=30,
    )

    assert not any("Tracker is stale" in item for item in checks["warnings"])
