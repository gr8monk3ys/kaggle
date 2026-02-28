from __future__ import annotations

import json
import sys

import campaign_pack


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
