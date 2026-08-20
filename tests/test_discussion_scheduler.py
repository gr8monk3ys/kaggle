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


def test_resolve_forum_prefers_longest_matching_key():
    resolve = discussion_scheduler.resolve_forum
    assert resolve("nlp getting started") == \
        "https://www.kaggle.com/competitions/nlp-getting-started/discussion"
    assert resolve("getting started") == \
        "https://www.kaggle.com/discussions/getting-started"
    assert resolve("deep past akkadian") == \
        "https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion"
    assert resolve("something unmapped") == discussion_scheduler.DEFAULT_FORUM


def test_parse_drafts_routes_nlp_getting_started_to_competition(tmp_path):
    drafts_path = tmp_path / "discussion-drafts.md"
    drafts_path.write_text(
        "\n".join(
            [
                "## Draft 1: NLP Tips",
                "**Target forum:** NLP Getting Started",
                "",
                "### NLP Tips",
                "",
                "Body.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    drafts = discussion_scheduler.parse_drafts(drafts_path)

    assert drafts[0]["forum_url"] == \
        "https://www.kaggle.com/competitions/nlp-getting-started/discussion"
    assert drafts[0]["priority"] == "high"


def test_select_next_post_prefers_most_overdue_due_item():
    from datetime import datetime, timezone
    now = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
    queue = [
        {"id": "draft_010", "status": "scheduled", "priority": "high",
         "scheduled_after": "2026-06-20T10:00:00+00:00"},   # future
        {"id": "draft_011", "status": "ready", "priority": "low",
         "scheduled_after": "2026-06-10T10:00:00+00:00"},   # due, most overdue
        {"id": "draft_012", "status": "ready", "priority": "high",
         "scheduled_after": "2026-06-13T10:00:00+00:00"},   # due, newer
        {"id": "draft_013", "status": "posted", "priority": "high",
         "scheduled_after": "2026-06-09T10:00:00+00:00"},   # not postable
    ]
    assert discussion_scheduler.select_next_post(queue, now=now)["id"] == "draft_011"


def test_select_next_post_none_when_no_postable():
    assert discussion_scheduler.select_next_post([{"id": "d", "status": "posted"}]) is None


def test_extract_post_body_strips_ops_metadata(tmp_path):
    md = tmp_path / "drafts.md"
    md.write_text("\n".join([
        "## Draft 7: Sample",
        "**Target forum:** General",
        "**Category:** Strategy",
        "**Status:** ready",
        "",
        "### Sample Post Title",
        "",
        "This is the real post body.",
        "Second line.",
        "",
    ]), encoding="utf-8")
    body = discussion_scheduler.extract_post_body(md, "Draft 7")
    assert "**Target forum:**" not in body
    assert "### Sample Post Title" in body
    assert "This is the real post body." in body


def test_cmd_next_post_outputs_copy_block(tmp_path, capsys):
    md = tmp_path / "drafts.md"
    md.write_text("## Draft 7: Sample\n**Target forum:** General\n\n### Title\n\nBody text.\n", encoding="utf-8")
    queue = [{"id": "draft_007", "status": "ready", "priority": "high", "title": "Sample",
              "forum_url": "https://www.kaggle.com/discussions/general",
              "body_section": "Draft 7", "scheduled_after": "2026-06-01T10:00:00+00:00"}]
    rc = discussion_scheduler.cmd_next_post(queue=queue, drafts_path=md)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Sample" in out and "Body text." in out
    assert "draft-set draft_007 --status posted" in out


def test_cmd_next_post_no_postable(capsys):
    rc = discussion_scheduler.cmd_next_post(queue=[{"id": "d", "status": "posted"}])
    assert rc == 0
    assert "No postable drafts" in capsys.readouterr().out


# ── fabricated-results guard ─────────────────────────────────────────────────

def test_asserts_unbacked_results_flags_measurement_claims():
    from kaggle_portfolio.ops.discussion_scheduler import asserts_unbacked_results

    assert asserts_unbacked_results("| Strategy | AUC |\n|---|---|\n| Mean | 0.813 |")
    assert asserts_unbacked_results("I benchmarked 7 strategies. Best was 0.833 AUC.")
    # An explicit evidence pointer clears the draft.
    assert not asserts_unbacked_results(
        "**Evidence:** projects/competitions/playground-series-s6e6/README.md\n"
        "| Model | AUC |\n|---|---|\n| GBM | 0.968 |"
    )
    # Hyperparameters in code are numbers, not claimed measurements.
    assert not asserts_unbacked_results("Set learning_rate=0.05 and subsample to 0.800.")
    assert not asserts_unbacked_results("I tested this approach and liked it.")


def test_no_postable_draft_reports_unbacked_results():
    """A draft claiming measurements must cite evidence or be unpostable.

    Six drafts shipped invented benchmark tables ("I benchmarked 7 strategies
    on 3 datasets" with AUCs nothing in the repo produced). Posting those under
    a real identity is the same failure as a fabricated medal claim.
    """
    import json
    import re as _re

    from kaggle_portfolio.ops.discussion_scheduler import (
        DRAFTS_FILE,
        POSTABLE_STATUSES,
        QUEUE_FILE,
        asserts_unbacked_results,
    )

    statuses = {i["id"]: i.get("status") for i in json.loads(QUEUE_FILE.read_text())}
    # parse_drafts() does not return body text, so read the sections directly.
    sections = _re.findall(
        r"## Draft (\d+):.*?\n(.*?)(?=\n## Draft |\Z)",
        DRAFTS_FILE.read_text(encoding="utf-8"),
        _re.S,
    )
    assert sections, "no draft sections parsed — the guard would be vacuous"

    offenders = [
        f"draft_{int(num):03d}"
        for num, body in sections
        if statuses.get(f"draft_{int(num):03d}") in POSTABLE_STATUSES
        and asserts_unbacked_results(body)
    ]
    assert not offenders, f"Postable drafts report unbacked results: {offenders}"


def test_measurement_claim_allows_auxiliaries_but_not_code():
    """"I have compared" asserts a measurement; `for i in range(...)` does not."""
    from kaggle_portfolio.ops.discussion_scheduler import MEASUREMENT_CLAIM

    for claim in ("I benchmarked 7 strategies", "I have compared five methods",
                  "I've tested this", "I recently measured", "I then ran a check"):
        assert MEASUREMENT_CLAIM.search(claim), claim
    # "ran\\w*" previously matched "range", flagging every for-loop in a code sample.
    for benign in ("for i in range(10):", "for i in range(n_splits):",
                   "I like gradient boosting", "I am new to Kaggle"):
        assert not MEASUREMENT_CLAIM.search(benign), benign
