"""Tests for the shared Campaign queue model.

Three modules used to hold three path constants and two reader/writer pairs over
one file. These cover the model itself and, specifically, the two hazards that
duplication was hiding.
"""

from __future__ import annotations

import json

import pytest

from kaggle_portfolio.campaigns import campaign_queue as cq
from kaggle_portfolio.shared.errors import CommandError


def _action(action_id: str, **fields):
    return {"id": action_id, "channel": "linkedin", "status": cq.PLANNED, **fields}


class TestClaimRule:
    """One rule. campaign_execute guarded on status; campaign_dispatcher did not."""

    def test_a_planned_action_is_claimed(self):
        action = _action("a")
        assert cq.claim(action, stamp="2026-01-01T00:00:00Z") is True
        assert action["status"] == cq.IN_PROGRESS
        assert action["claim_count"] == 1

    @pytest.mark.parametrize("status", [cq.IN_PROGRESS, cq.DONE, cq.BLOCKED])
    def test_anything_else_is_refused_and_the_retry_count_is_untouched(self, status):
        action = _action("a", status=status, claim_count=1)
        assert cq.claim(action) is False
        assert action["claim_count"] == 1, "claim_count counts retries, not invocations"
        assert action["status"] == status

    def test_requeue_is_the_deliberate_way_back(self):
        action = _action("a", status=cq.IN_PROGRESS, claimed_at="x")
        cq.requeue(action)
        assert action["status"] == cq.PLANNED
        assert "claimed_at" not in action
        assert cq.claim(action) is True


class TestMergeOnRegenerate:
    """campaign-pack regenerates the plan; it must not rewrite what the run did."""

    def test_completed_work_survives_regeneration(self):
        existing = [
            _action(
                "a", status=cq.DONE, completed_at="2026-09-01T00:00:00Z", claim_count=1
            ),
            _action("b"),
        ]
        merged = cq.merge_regenerated(existing, [_action("a"), _action("b")])
        done = cq.find_by_id(merged, "a")
        assert done["status"] == cq.DONE, "regenerating would have re-posted this"
        assert done["completed_at"] == "2026-09-01T00:00:00Z"
        assert done["claim_count"] == 1

    def test_blocked_work_is_not_revived(self):
        merged = cq.merge_regenerated(
            [_action("a", status=cq.BLOCKED, note="rejected")], [_action("a")]
        )
        assert cq.find_by_id(merged, "a")["status"] == cq.BLOCKED

    def test_the_fresh_plan_still_updates_the_planning_fields(self):
        existing = [_action("a", status=cq.DONE, copy="old copy")]
        merged = cq.merge_regenerated(existing, [_action("a", copy="new copy")])
        item = cq.find_by_id(merged, "a")
        assert item["copy"] == "new copy", "the plan is what regeneration is for"
        assert item["status"] == cq.DONE, "the run's record is not"

    def test_new_actions_are_added(self):
        merged = cq.merge_regenerated([_action("a")], [_action("a"), _action("b")])
        assert {i["id"] for i in merged} == {"a", "b"}

    def test_finished_work_dropped_from_the_new_plan_is_kept_as_history(self):
        existing = [_action("a", status=cq.DONE), _action("b")]
        merged = cq.merge_regenerated(existing, [_action("b")])
        assert cq.find_by_id(merged, "a") is not None, "history must not vanish"

    def test_unfinished_work_dropped_from_the_plan_is_not_kept(self):
        merged = cq.merge_regenerated([_action("a"), _action("b")], [_action("b")])
        assert cq.find_by_id(merged, "a") is None


class TestFileIO:
    def test_round_trip_sets_updated_at(self, tmp_path):
        path = tmp_path / "q.json"
        cq.save_payload(path, {"generated_on": "2026-09-06", "queue": [_action("a")]})
        payload = cq.load_payload(path)
        assert payload["updated_at"], "the envelope's updated_at is the model's job"
        assert len(payload["queue"]) == 1

    def test_a_missing_queue_says_so(self, tmp_path):
        with pytest.raises(CommandError, match="not found"):
            cq.load_payload(tmp_path / "absent.json")

    def test_a_malformed_queue_says_so(self, tmp_path):
        path = tmp_path / "q.json"
        path.write_text(json.dumps({"queue": "not a list"}))
        with pytest.raises(CommandError, match="missing 'queue' list"):
            cq.load_payload(path)


class TestOrderingAndFilters:
    def test_sorted_by_schedule_then_id(self):
        queue = [
            _action("b", scheduled_for="2026-01-02"),
            _action("a", scheduled_for="2026-01-01"),
        ]
        assert [i["id"] for i in cq.sorted_queue(queue)] == ["a", "b"]

    def test_filter_by_status_channel_and_limit(self):
        queue = [
            _action("a", channel="x"),
            _action("b", channel="linkedin"),
            _action("c", channel="linkedin", status=cq.DONE),
        ]
        got = cq.filter_actions(
            queue, statuses={cq.PLANNED}, channels={"linkedin"}, limit=5
        )
        assert [i["id"] for i in got] == ["b"]
