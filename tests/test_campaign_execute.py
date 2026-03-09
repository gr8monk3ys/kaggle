from __future__ import annotations

from datetime import datetime, timezone

from kaggle_portfolio.campaigns import campaign_execute


def _action(
    action_id: str,
    *,
    status: str = "planned",
    channel: str = "kaggle-discussion",
    scheduled_for: str = "2026-03-02T10:00:00Z",
) -> dict[str, str]:
    return {
        "id": action_id,
        "status": status,
        "channel": channel,
        "scheduled_for": scheduled_for,
        "dataset_ref": "owner/dataset",
        "copy": "test copy",
    }


def test_due_supported_actions_prioritizes_in_progress_and_filters_unsupported():
    queue = [
        _action("a", status="planned", channel="kaggle-discussion", scheduled_for="2026-03-02T10:00:00Z"),
        _action("b", status="in_progress", channel="kaggle-changelog", scheduled_for="2026-03-02T11:00:00Z"),
        _action("c", status="planned", channel="kaggle-discussion", scheduled_for="2026-03-03T10:00:00Z"),
        _action("d", status="planned", channel="x", scheduled_for="2026-03-02T09:00:00Z"),
        _action("e", status="done", channel="kaggle-discussion", scheduled_for="2026-03-02T08:00:00Z"),
    ]
    now = datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)

    selected = campaign_execute.due_supported_actions(
        queue,
        now=now,
        limit=10,
        allowed_channels=None,
        include_planned=True,
        include_in_progress=True,
        respect_schedule=True,
    )

    assert [item["id"] for item in selected] == ["b", "a"]


def test_due_supported_actions_respects_channel_filter_and_can_ignore_schedule():
    queue = [
        _action("a", status="planned", channel="kaggle-discussion", scheduled_for="2026-03-03T10:00:00Z"),
        _action("b", status="planned", channel="kaggle-changelog", scheduled_for="2026-03-03T11:00:00Z"),
    ]
    now = datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)

    selected = campaign_execute.due_supported_actions(
        queue,
        now=now,
        limit=10,
        allowed_channels={"kaggle-changelog"},
        include_planned=True,
        include_in_progress=False,
        respect_schedule=False,
    )

    assert [item["id"] for item in selected] == ["b"]


def test_claim_action_only_claims_planned():
    planned = _action("a", status="planned")
    in_progress = _action("b", status="in_progress")

    campaign_execute.claim_action(planned, stamp="2026-03-02T12:00:00Z")
    campaign_execute.claim_action(in_progress, stamp="2026-03-02T12:00:00Z")

    assert planned["status"] == "in_progress"
    assert planned["claimed_at"] == "2026-03-02T12:00:00Z"
    assert planned["claim_count"] == 1
    assert "claim_count" not in in_progress


def test_mark_done_and_mark_error_update_expected_fields():
    action = _action("a", status="in_progress")
    action["last_error"] = "previous failure"

    campaign_execute.mark_done(action, post_url="https://www.kaggle.com/discussion/123", stamp="2026-03-02T12:01:00Z")
    assert action["status"] == "done"
    assert action["completed_at"] == "2026-03-02T12:01:00Z"
    assert action["note"] == "posted: https://www.kaggle.com/discussion/123"
    assert "last_error" not in action

    long_error = "x" * 800
    campaign_execute.mark_error(action, long_error)
    assert action["status"] == "in_progress"
    assert action["fail_count"] == 1
    assert len(action["last_error"]) == 600


def test_topic_title_for_action_defaults_and_changelog_prefix():
    assert (
        campaign_execute.topic_title_for_action(
            {"channel": "kaggle-changelog", "dataset_title": "Great Dataset"}
        )
        == "Changelog: Great Dataset"
    )
    assert (
        campaign_execute.topic_title_for_action(
            {"channel": "kaggle-discussion", "dataset_ref": "owner/dataset"}
        )
        == "Usability Update: owner/dataset"
    )


def test_parse_channels_normalizes_and_skips_empty_values():
    assert campaign_execute.parse_channels([" Kaggle-Discussion ", "", "kaggle-changelog"]) == {
        "kaggle-discussion",
        "kaggle-changelog",
    }
