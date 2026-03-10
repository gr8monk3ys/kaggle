from __future__ import annotations

import json
import sys

from kaggle_portfolio.campaigns import campaign_dispatcher


def _write_queue(path, queue):
    path.write_text(
        json.dumps({"generated_on": "2026-02-24", "queue": queue}),
        encoding="utf-8",
    )


def test_claim_updates_status_and_writes_queue(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.json"
    report_path = tmp_path / "runbook.md"
    _write_queue(
        queue_path,
        [
            {
                "id": "campaign_001",
                "scheduled_for": "2026-02-24T14:00:00Z",
                "channel": "kaggle-discussion",
                "dataset_ref": "u/a",
                "copy": "A",
                "status": "planned",
            },
            {
                "id": "campaign_002",
                "scheduled_for": "2026-02-24T15:00:00Z",
                "channel": "x",
                "dataset_ref": "u/b",
                "copy": "B",
                "status": "planned",
            },
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_dispatcher.py",
            "--queue-path",
            str(queue_path),
            "--report-path",
            str(report_path),
            "--limit",
            "1",
            "--claim",
        ],
    )

    rc = campaign_dispatcher.main()
    assert rc == 0
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue"][0]["status"] == "in_progress"
    assert payload["queue"][1]["status"] == "planned"
    assert report_path.exists()


def test_complete_marks_done(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.json"
    _write_queue(
        queue_path,
        [
            {
                "id": "campaign_001",
                "scheduled_for": "2026-02-24T14:00:00Z",
                "channel": "kaggle-discussion",
                "dataset_ref": "u/a",
                "copy": "A",
                "status": "in_progress",
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_dispatcher.py",
            "--queue-path",
            str(queue_path),
            "--complete-id",
            "campaign_001",
            "--no-report",
        ],
    )

    rc = campaign_dispatcher.main()
    assert rc == 0
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue"][0]["status"] == "done"
