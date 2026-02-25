import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "pi-automation/scripts")

import discussion_queue as dq

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


def test_next_pending_returns_first_eligible():
    item = dq.next_pending(SAMPLE_QUEUE, now=NOW)
    assert item is not None
    assert item["id"] == "d1"


def test_next_pending_skips_posted():
    item = dq.next_pending(SAMPLE_QUEUE, now=NOW)
    assert item["id"] != "d2"


def test_next_pending_skips_future_scheduled():
    item = dq.next_pending(SAMPLE_QUEUE, now=NOW)
    assert item["id"] != "d3"


def test_next_pending_returns_none_when_nothing_ready():
    assert dq.next_pending([], now=NOW) is None


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


def test_extract_body_finds_section():
    content = (
        "## Draft 1: Feature Engineering\n\n"
        "**Target forum:** Getting Started\n\n"
        "### Feature Engineering\n\nBody content here.\n\n---\n\n"
        "## Draft 2: Other\n\nOther content.\n"
    )
    body = dq.extract_body(content, "Draft 1")
    assert "Body content here" in body
    assert "Draft 2" not in body
    assert "Target forum" not in body
