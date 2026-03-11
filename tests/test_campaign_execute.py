from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
        == "Refresh Plan: Great Dataset"
    )
    assert (
        campaign_execute.topic_title_for_action(
            {"channel": "kaggle-discussion", "dataset_ref": "owner/dataset"}
        )
        == "Feedback Wanted: improving owner/dataset"
    )


def test_parse_channels_normalizes_and_skips_empty_values():
    assert campaign_execute.parse_channels([" Kaggle-Discussion ", "", "kaggle-changelog"]) == {
        "kaggle-discussion",
        "kaggle-changelog",
    }


def test_post_dataset_discussion_topic_raises_clear_error_on_browser_challenge(monkeypatch):
    class FakePage:
        def goto(self, *args, **kwargs):
            return None

        def wait_for_timeout(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(campaign_execute.kb, "is_browser_challenge", lambda page: True)

    with pytest.raises(RuntimeError, match="browser challenge"):
        campaign_execute.post_dataset_discussion_topic(
            FakePage(),
            dataset_ref="owner/dataset",
            topic_title="Title",
            body="Body",
            timeout_ms=1000,
        )


def test_extract_submission_error_reads_similarity_message():
    class FakeBody:
        def inner_text(self, timeout=None):
            return "This comment is too similar to a previous one. We prevent redundant comments to reduce spam."

    class FakePage:
        def locator(self, selector):
            assert selector == "body"
            return FakeBody()

    assert (
        campaign_execute.extract_submission_error(FakePage())
        == "This comment is too similar to a previous one. We prevent redundant comments to reduce spam."
    )


def test_post_dataset_discussion_topic_uses_direct_composer_url(monkeypatch):
    class FakeLocator:
        def __init__(self, count=0, *, page=None, click_url=None):
            self._count = count
            self.page = page
            self.click_url = click_url
            self.filled = None

        @property
        def first(self):
            return self

        def count(self):
            return self._count

        def fill(self, value, timeout=None):
            self.filled = value

        def click(self, timeout=None):
            if self.page is not None and self.click_url:
                self.page.url = self.click_url

        def is_disabled(self):
            return False

        def filter(self, **kwargs):
            return self

    class FakePage:
        def __init__(self):
            self.url = ""
            self.goto_urls = []
            self._title = FakeLocator(1)
            self._content = FakeLocator(1)
            self._publish = FakeLocator(
                1,
                page=self,
                click_url="https://www.kaggle.com/datasets/owner/dataset/discussion/123",
            )

        def goto(self, url, **kwargs):
            self.url = url
            self.goto_urls.append(url)

        def wait_for_timeout(self, *_args, **_kwargs):
            return None

        def get_by_role(self, role, name=None):
            pattern = getattr(name, "pattern", "")
            if role == "textbox" and "topic" in pattern.lower():
                return self._title
            if role == "textbox" and "content" in pattern.lower():
                return self._content
            if role == "button" and ("publish" in pattern.lower() or "post" in pattern.lower()):
                return self._publish
            return FakeLocator(0)

        def locator(self, selector):
            if selector == "body":
                class FakeBody:
                    def inner_text(self, timeout=None):
                        return ""

                return FakeBody()
            return FakeLocator(0)

        def get_by_text(self, *args, **kwargs):
            return FakeLocator(0)

    monkeypatch.setattr(campaign_execute.kb, "is_browser_challenge", lambda page: False)

    page = FakePage()
    post_url = campaign_execute.post_dataset_discussion_topic(
        page,
        dataset_ref="owner/dataset",
        topic_title="Title",
        body="Body",
        timeout_ms=1000,
    )

    assert page.goto_urls[0].endswith("/discussion/new")
    assert post_url.endswith("/discussion/123")
