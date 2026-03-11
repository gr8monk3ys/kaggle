from __future__ import annotations

import json
import sys

from kaggle_portfolio.campaigns import campaign_pack


def test_prioritize_datasets_orders_by_live_status_and_rating():
    rows = [
        {"dataset_ref": "u/strong", "path": "datasets/strong", "title": "Strong", "kaggle_usability_rating": 0.86},
        {"dataset_ref": "u/watch", "path": "datasets/watch", "title": "Watch", "kaggle_usability_rating": 0.74},
        {"dataset_ref": "u/critical", "path": "datasets/critical", "title": "Critical", "kaggle_usability_rating": 0.66},
    ]

    ranked = campaign_pack.prioritize_datasets(
        rows,
        alert_under=0.7,
        target_rating=0.8,
        max_datasets=10,
    )

    assert [item["dataset_ref"] for item in ranked] == ["u/critical", "u/watch", "u/strong"]
    assert ranked[0]["status"] == "critical"
    assert ranked[1]["status"] == "watch"
    assert ranked[2]["status"] == "strong"


def test_filter_rows_by_refs_returns_exact_matches_and_warnings():
    rows = [
        {"dataset_ref": "u/one", "title": "One"},
        {"dataset_ref": "u/two", "title": "Two"},
    ]

    filtered, warnings = campaign_pack.filter_rows_by_refs(rows, {"u/two", "u/missing"})

    assert [row["dataset_ref"] for row in filtered] == ["u/two"]
    assert warnings == ["Requested dataset ref not found in report: u/missing"]


def test_main_writes_campaign_reports_and_queue(tmp_path, monkeypatch):
    report = {
        "generated_on": "2026-02-24",
        "datasets": [
            {
                "dataset_ref": "u/critical",
                "path": "datasets/critical",
                "title": "Critical",
                "kaggle_usability_rating": 0.66,
            },
            {
                "dataset_ref": "u/strong",
                "path": "datasets/strong",
                "title": "Strong",
                "kaggle_usability_rating": 0.86,
            },
        ],
    }
    report_path = tmp_path / "dataset-usability.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    queue_path = tmp_path / "promotion_queue.json"
    out_root = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_pack.py",
            "--dataset-report",
            str(report_path),
            "--output-root",
            str(out_root),
            "--queue-path",
            str(queue_path),
            "--today",
            "2026-02-24",
            "--days",
            "2",
            "--posts-per-day",
            "2",
            "--max-datasets",
            "2",
        ],
    )

    rc = campaign_pack.main()

    assert rc == 0
    assert (out_root / "reports" / "latest-promotion-campaign.md").exists()
    assert (out_root / "reports" / "latest-promotion-campaign.json").exists()
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(payload["queue"]) == 4
    assert payload["queue"][0]["dataset_ref"] == "u/critical"


def test_main_filters_to_requested_refs(tmp_path, monkeypatch):
    report = {
        "generated_on": "2026-02-24",
        "datasets": [
            {
                "dataset_ref": "u/critical",
                "path": "datasets/critical",
                "title": "Critical",
                "kaggle_usability_rating": 0.66,
            },
            {
                "dataset_ref": "u/strong",
                "path": "datasets/strong",
                "title": "Strong",
                "kaggle_usability_rating": 0.86,
            },
        ],
    }
    report_path = tmp_path / "dataset-usability.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    queue_path = tmp_path / "promotion_queue.json"
    out_root = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_pack.py",
            "--dataset-report",
            str(report_path),
            "--output-root",
            str(out_root),
            "--queue-path",
            str(queue_path),
            "--today",
            "2026-02-24",
            "--days",
            "1",
            "--posts-per-day",
            "1",
            "--max-datasets",
            "5",
            "--refs",
            "u/strong",
        ],
    )

    rc = campaign_pack.main()

    assert rc == 0
    payload = json.loads((out_root / "reports" / "latest-promotion-campaign.json").read_text(encoding="utf-8"))
    assert payload["ref_filter"] == ["u/strong"]
    assert [item["dataset_ref"] for item in payload["prioritized_datasets"]] == ["u/strong"]


def test_build_channel_copy_uses_distinct_discussion_and_changelog_language():
    dataset = {
        "dataset_ref": "u/example",
        "title": "Example Dataset",
        "dataset_url": "https://www.kaggle.com/datasets/u/example",
        "rating": 0.647,
        "objective": "Improve discoverability and README depth before promotion.",
    }

    copy_map = campaign_pack.build_channel_copy(dataset, target_rating=0.8)

    assert "want concrete feedback" in copy_map["kaggle-discussion"]
    assert "Refresh plan for Example Dataset" in copy_map["kaggle-changelog"]
    assert "one improvement" in copy_map["kaggle-discussion"]
