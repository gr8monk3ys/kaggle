"""Queue management for scheduled Kaggle discussion posts."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def next_pending(queue: list[dict], now: datetime | None = None) -> dict | None:
    """Return the first pending item whose scheduled_after time has passed."""
    if now is None:
        now = datetime.now(tz=timezone.utc)
    for item in queue:
        if item.get("status") != "pending":
            continue
        try:
            scheduled = datetime.fromisoformat(
                item.get("scheduled_after", "").replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if now >= scheduled:
            return item
    return None


def mark_posted(queue_path: Path, item_id: str, post_url: str) -> None:
    """Update queue file: set item status to 'posted', record URL and timestamp."""
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    for item in queue:
        if item["id"] == item_id:
            item["status"] = "posted"
            item["post_url"] = post_url
            item["posted_at"] = datetime.now(tz=timezone.utc).isoformat()
            break
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def extract_body(drafts_content: str, section_name: str) -> str:
    """Extract post body from a markdown drafts file by section heading.

    Strips metadata lines (Target forum, Category, Expected medal) and the
    ### sub-heading (which becomes the post title, not the body).
    """
    pattern = re.compile(
        rf"## {re.escape(section_name)}:.*?\n(.*?)(?=\n## |\Z)", re.DOTALL
    )
    match = pattern.search(drafts_content)
    if not match:
        raise ValueError(f"Section not found in drafts: {section_name}")
    body = match.group(1).strip()
    # Strip metadata lines
    for meta in (
        r"\*\*Target forum:\*\*.*\n?",
        r"\*\*Category:\*\*.*\n?",
        r"\*\*Expected medal:\*\*.*\n?",
    ):
        body = re.sub(meta, "", body)
    # Strip the ### heading (it becomes the post title, not the body)
    body = re.sub(r"^###.+\n\n?", "", body.lstrip())
    return body.strip()
