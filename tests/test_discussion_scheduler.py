from datetime import datetime, timedelta, timezone
from pathlib import Path

from kaggle_portfolio.ops import discussion_scheduler


def test_parse_drafts_uses_draft_label_for_body_section(tmp_path):
    drafts_path = tmp_path / "discussion-drafts.md"
    drafts_path.write_text(
        "\n".join(
            [
                "## Draft 1: Feature Engineering Tricks",
                "**Target forum:** Getting Started",
                "",
                "### Feature Engineering Tricks",
                "",
                "Body text here.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    drafts = discussion_scheduler.parse_drafts(drafts_path)

    assert len(drafts) == 1
    assert drafts[0]["id"] == "draft_001"
    assert drafts[0]["body_section"] == "Draft 1"
    assert drafts[0]["body_title"] == "Feature Engineering Tricks"


def test_parse_drafts_extracts_ops_metadata(tmp_path):
    drafts_path = tmp_path / "discussion-drafts.md"
    drafts_path.write_text(
        "\n".join(
            [
                "## Draft 2: Deadline Draft",
                "**Target forum:** General",
                "**Category:** Strategy",
                "**Expected medal:** Bronze",
                "**Priority:** high",
                "**Deadline:** 2026-03-01",
                "**Status:** ready",
                "",
                "### Deadline Draft",
                "",
                "Body text.",
            ]
        ),
        encoding="utf-8",
    )

    drafts = discussion_scheduler.parse_drafts(drafts_path)

    assert len(drafts) == 1
    assert drafts[0]["priority"] == "high"
    assert drafts[0]["deadline"] == "2026-03-01"
    assert drafts[0]["status"] == "ready"
    assert drafts[0]["expected_medal"] == "Bronze"


def test_generate_queue_prioritizes_high_priority_and_leaves_idea_unscheduled():
    drafts = [
        {
            "id": "draft_001",
            "number": 1,
            "title": "Low Priority",
            "forum_url": "https://www.kaggle.com/discussions/general",
            "body_section": "Draft 1",
            "body_file": "discussion-drafts.md",
            "priority": "low",
            "deadline": None,
            "category": "",
            "expected_medal": "",
            "status": "ready",
        },
        {
            "id": "draft_002",
            "number": 2,
            "title": "High Priority",
            "forum_url": "https://www.kaggle.com/competitions/titanic/discussion",
            "body_section": "Draft 2",
            "body_file": "discussion-drafts.md",
            "priority": "high",
            "deadline": "2026-03-01",
            "category": "",
            "expected_medal": "",
            "status": "ready",
        },
        {
            "id": "draft_003",
            "number": 3,
            "title": "Idea Draft",
            "forum_url": "https://www.kaggle.com/discussions/general",
            "body_section": "Draft 3",
            "body_file": "discussion-drafts.md",
            "priority": "medium",
            "deadline": None,
            "category": "",
            "expected_medal": "",
            "status": "idea",
        },
    ]
    start = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)

    queue = discussion_scheduler.generate_queue(drafts, start_date=start)
    by_id = {item["id"]: item for item in queue}

    assert by_id["draft_002"]["status"] == "scheduled"
    assert by_id["draft_001"]["status"] == "scheduled"
    assert by_id["draft_003"]["status"] == "idea"
    assert by_id["draft_003"]["scheduled_after"] is None

    # High-priority draft should get the earliest schedule slot.
    assert by_id["draft_002"]["scheduled_after"] < by_id["draft_001"]["scheduled_after"]


def test_generate_queue_preserves_skipped_status_outside_schedule():
    drafts = [
        {
            "id": "draft_001",
            "number": 1,
            "title": "Skipped Draft",
            "forum_url": "https://www.kaggle.com/discussions/general",
            "body_section": "Draft 1",
            "body_file": "discussion-drafts.md",
            "priority": "medium",
            "deadline": None,
            "category": "",
            "expected_medal": "",
            "status": "skipped",
        },
        {
            "id": "draft_002",
            "number": 2,
            "title": "Scheduled Draft",
            "forum_url": "https://www.kaggle.com/discussions/general",
            "body_section": "Draft 2",
            "body_file": "discussion-drafts.md",
            "priority": "medium",
            "deadline": None,
            "category": "",
            "expected_medal": "",
            "status": "ready",
        },
    ]

    queue = discussion_scheduler.generate_queue(drafts, start_date=datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc))
    by_id = {item["id"]: item for item in queue}

    assert by_id["draft_001"]["status"] == "skipped"
    assert by_id["draft_001"]["scheduled_after"] is None
    assert by_id["draft_002"]["status"] == "scheduled"


def test_build_ops_summary_reports_flow_health_metrics():
    now = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    queue = [
        {"id": "draft_001", "status": "scheduled", "scheduled_after": (now - timedelta(days=1)).isoformat()},
        {"id": "draft_002", "status": "scheduled", "scheduled_after": (now + timedelta(days=2)).isoformat()},
        {"id": "draft_003", "status": "ready", "scheduled_after": None},
        {"id": "draft_004", "status": "idea", "scheduled_after": None},
        {"id": "draft_005", "status": "posted", "scheduled_after": None},
    ]

    summary = discussion_scheduler.build_ops_summary(queue, now=now)

    assert summary["stage_counts"]["scheduled"] == 2
    assert summary["stage_counts"]["ready"] == 1
    assert summary["backlog_total"] == 4
    assert summary["ready_now"] == 1
    assert summary["ready_backlog"] == 1
    assert summary["scheduled_next_7d"] == 1
    assert summary["overdue_scheduled"] == 1
    assert summary["days_until_next_post"] == 0
    assert summary["next_post_due"] is not None
    assert summary["schedule_horizon"] == (now + timedelta(days=2)).date().isoformat()
    assert summary["estimated_weeks_to_clear"] == 1


def test_build_ops_summary_ignores_skipped_items_in_backlog_counts():
    now = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    queue = [
        {"id": "draft_001", "status": "skipped", "scheduled_after": None},
        {"id": "draft_002", "status": "scheduled", "scheduled_after": (now + timedelta(days=2)).isoformat()},
    ]

    summary = discussion_scheduler.build_ops_summary(queue, now=now)

    assert summary["stage_counts"]["skipped"] == 1
    assert summary["backlog_total"] == 1
    assert summary["overdue_scheduled"] == 0


def test_generate_queue_limits_scheduled_window_and_leaves_rest_ready():
    drafts = []
    for idx in range(1, 8):
        drafts.append(
            {
                "id": f"draft_{idx:03d}",
                "number": idx,
                "title": f"Draft {idx}",
                "forum_url": "https://www.kaggle.com/discussions/general",
                "body_section": f"Draft {idx}",
                "body_file": "discussion-drafts.md",
                "priority": "medium",
                "deadline": None,
                "category": "",
                "expected_medal": "",
                "status": "ready",
            }
        )

    start = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    queue = discussion_scheduler.generate_queue(drafts, start_date=start, schedule_weeks=2)

    scheduled = [item for item in queue if item["status"] == "scheduled"]
    ready = [item for item in queue if item["status"] == "ready"]

    assert len(scheduled) == 6  # 2 weeks * 3 posts per week
    assert len(ready) == 1
    assert ready[0]["id"] == "draft_007"
    assert ready[0]["scheduled_after"] is None


def test_run_health_check_fails_when_overdue_exceeds_threshold():
    now = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    queue = [
        {"id": "draft_001", "status": "scheduled", "scheduled_after": (now - timedelta(days=1)).isoformat()},
        {"id": "draft_002", "status": "scheduled", "scheduled_after": (now + timedelta(days=2)).isoformat()},
    ]

    rc = discussion_scheduler.run_health_check(
        queue,
        max_overdue_scheduled=0,
        max_days_until_next_post=7,
        now=now,
    )

    assert rc == 1


def test_run_health_check_fails_when_next_post_gap_too_large():
    now = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    queue = [
        {"id": "draft_001", "status": "scheduled", "scheduled_after": (now + timedelta(days=10)).isoformat()},
    ]

    rc = discussion_scheduler.run_health_check(
        queue,
        max_overdue_scheduled=0,
        max_days_until_next_post=7,
        now=now,
    )

    assert rc == 1


def test_update_draft_rebalances_schedule_with_canonical_id():
    now = datetime(2026, 2, 24, 10, 0, tzinfo=timezone.utc)
    queue = []
    for idx in range(1, 6):
        queue.append(
            {
                "id": f"draft_{idx:03d}",
                "title": f"Draft {idx}",
                "forum_url": "https://www.kaggle.com/discussions/general",
                "body_section": f"Draft {idx}",
                "body_file": "discussion-drafts.md",
                "priority": "medium",
                "deadline": None,
                "category": "",
                "expected_medal": "",
                "status": "ready",
                "scheduled_after": None,
                "post_url": None,
                "posted_at": None,
            }
        )

    updated_queue, updated = discussion_scheduler.update_draft(
        queue,
        "draft-005",
        priority="high",
        deadline="2026-02-25",
        schedule_weeks=1,
        now=now,
    )
    by_id = {item["id"]: item for item in updated_queue}

    assert updated["id"] == "draft_005"
    assert updated["priority"] == "high"
    assert updated["deadline"] == "2026-02-25"
    assert sum(1 for item in updated_queue if item["status"] == "scheduled") == 3
    assert by_id["draft_005"]["status"] == "scheduled"
    assert by_id["draft_005"]["scheduled_after"] <= by_id["draft_001"]["scheduled_after"]
