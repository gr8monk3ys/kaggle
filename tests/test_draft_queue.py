"""Tests for the shared Draft Queue model.

Moved here from pi-automation/tests when the poster's duplicate model was
deleted: the tests belong beside the model, not beside one of its two callers.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from kaggle_portfolio.discussions import draft_queue as dq

SAMPLE_QUEUE = [
    {
        "id": "d1",
        "title": "Post One",
        "forum_url": "https://kaggle.com/discussions/getting-started",
        "body_file": "discussion-drafts.md",
        "body_section": "Draft 1",
        "status": "pending",
        "scheduled_after": "2026-02-01T00:00:00Z",
    },
    {
        "id": "d2",
        "title": "Post Two",
        "forum_url": "https://kaggle.com/discussions/general",
        "body_file": "discussion-drafts.md",
        "body_section": "Draft 2",
        "status": "posted",
        "scheduled_after": "2026-01-01T00:00:00Z",
    },
    {
        "id": "d3",
        "title": "Future Post",
        "forum_url": "https://kaggle.com/discussions/general",
        "body_file": "discussion-drafts.md",
        "body_section": "Draft 3",
        "status": "pending",
        "scheduled_after": "2099-01-01T00:00:00Z",
    },
]
NOW = datetime(2026, 2, 22, 10, 0, tzinfo=timezone.utc)


def test_select_next_post_returns_first_eligible():
    item = dq.select_next_post(SAMPLE_QUEUE, now=NOW)
    assert item is not None
    assert item["id"] == "d1"


def test_select_next_post_skips_posted():
    item = dq.select_next_post(SAMPLE_QUEUE, now=NOW)
    assert item["id"] != "d2"


def test_select_next_post_skips_future_scheduled():
    item = dq.select_next_post(SAMPLE_QUEUE, now=NOW)
    assert item["id"] != "d3"


def test_select_next_post_returns_none_when_nothing_ready():
    assert dq.select_next_post([], now=NOW) is None


def test_select_next_post_accepts_scheduled_status():
    queue = [
        {
            "id": "s1",
            "status": "scheduled",
            "scheduled_after": "2026-02-20T00:00:00Z",
        }
    ]
    item = dq.select_next_post(queue, now=NOW)
    assert item is not None
    assert item["id"] == "s1"


def test_select_next_post_skips_idea_status():
    queue = [
        {
            "id": "i1",
            "status": "idea",
            "scheduled_after": None,
        },
        {
            "id": "s1",
            "status": "scheduled",
            "scheduled_after": "2026-02-20T00:00:00Z",
        },
    ]
    item = dq.select_next_post(queue, now=NOW)
    assert item is not None
    assert item["id"] == "s1"


def test_select_next_post_ready_without_schedule_is_postable():
    queue = [
        {
            "id": "r1",
            "status": "ready",
            "scheduled_after": None,
        }
    ]
    item = dq.select_next_post(queue, now=NOW)
    assert item is not None
    assert item["id"] == "r1"


def test_an_unscheduled_ready_draft_is_due_even_behind_a_future_schedule():
    """A deliberate behaviour change — see docs/adr/0003-one-draft-queue-model.md.

    The poster's old selector returned None here, so a ready draft could be
    starved indefinitely by an unrelated future-scheduled one. The scheduler's
    selector — now the only one — treats an unscheduled ready draft as due, which
    is also what the flywheel has always predicted.
    """
    queue = [
        {
            "id": "s-future",
            "status": "scheduled",
            "scheduled_after": "2026-03-01T00:00:00Z",
        },
        {"id": "r1", "status": "ready", "scheduled_after": None},
    ]
    assert dq.select_next_post(queue, now=NOW)["id"] == "r1"


def test_mark_posted_updates_queue_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_QUEUE, f)
        path = Path(f.name)
    dq.mark_posted(path, "d1", post_url="https://kaggle.com/discussion/42")
    updated = json.loads(path.read_text())
    item = next(i for i in updated if i["id"] == "d1")
    assert item["status"] == "posted"
    assert item["post_url"] == "https://kaggle.com/discussion/42"
    assert "posted_at" in item
    path.unlink()


def test_mark_posted_refills_scheduled_window_from_ready_backlog():
    queue = [
        {
            "id": "d1",
            "status": "scheduled",
            "scheduled_after": "2026-02-01T00:00:00Z",
        },
        {
            "id": "r1",
            "status": "ready",
            "scheduled_after": None,
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(queue, f)
        path = Path(f.name)
    dq.mark_posted(path, "d1", post_url="https://kaggle.com/discussion/42")
    updated = json.loads(path.read_text())
    d1 = next(i for i in updated if i["id"] == "d1")
    r1 = next(i for i in updated if i["id"] == "r1")
    assert d1["status"] == "posted"
    assert r1["status"] == "scheduled"
    assert r1["scheduled_after"] is not None
    path.unlink()


def test_extract_body_finds_section():
    content = (
        "## Draft 1: Feature Engineering\n\n"
        "**Target forum:** Getting Started\n\n"
        "### Feature Engineering\n\nBody content here.\n\n---\n\n"
        "## Draft 2: Other\n\nOther content.\n"
    )
    body = dq.extract_body(content, "Draft 1", strip_heading=True)
    assert "Body content here" in body
    assert "Draft 2" not in body
    assert "Target forum" not in body


def test_extract_body_accepts_title_for_backward_compatibility():
    content = (
        "## Draft 1: Feature Engineering\n\n"
        "**Target forum:** Getting Started\n\n"
        "### Feature Engineering\n\nBody content here.\n\n---\n\n"
        "## Draft 2: Other\n\nOther content.\n"
    )
    body = dq.extract_body(content, "Feature Engineering", strip_heading=True)
    assert "Body content here" in body


def test_extract_body_strips_ops_metadata_lines():
    content = (
        "## Draft 7: Ops Metadata\n\n"
        "**Target forum:** General\n"
        "**Category:** Strategy\n"
        "**Expected medal:** Bronze\n"
        "**Priority:** high\n"
        "**Deadline:** 2026-03-01\n"
        "**Status:** ready\n\n"
        "### Ops Metadata\n\n"
        "Actual body content.\n"
    )

    body = dq.extract_body(content, "Draft 7", strip_heading=True)

    assert "Actual body content." in body
    assert "Priority" not in body
    assert "Deadline" not in body
    assert "Status" not in body
