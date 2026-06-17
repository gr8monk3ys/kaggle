"""Read-only GrowthState: current votes/followers/medals from existing trackers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from kaggle_portfolio.ops import medal_ops
from kaggle_portfolio.ops import metadata_tracker

ROOT = Path(__file__).resolve().parents[2]
TRACKER_PATH = ROOT / "docs" / "reports" / "grandmaster-tracker.md"


def _default_growth_dir() -> Path:
    """Flywheel state directory.

    Overridable via the FLYWHEEL_DIR env var so deployments where the repo is
    mounted read-only (e.g. the pi-automation container at /repo:ro) can point
    writes at a writable volume such as /data/growth.
    """
    override = os.environ.get("FLYWHEEL_DIR")
    return Path(override) if override else ROOT / "medal_ops" / "growth"


GROWTH_DIR = _default_growth_dir()


@dataclass(frozen=True)
class ItemState:
    slug: str
    kind: str  # "notebook" | "dataset"
    votes: int
    title: str


@dataclass(frozen=True)
class GrowthState:
    followers: int
    items: list[ItemState]
    discussion_medals: int
    discussion_total_posts: int
    snapshot: dict = field(default_factory=dict)


def _read_tracker() -> str:
    return TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.exists() else ""


def _read_followers() -> int:
    path = GROWTH_DIR / "followers.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("followers", 0))
    except (ValueError, json.JSONDecodeError):
        return 0


def build(today: date | None = None) -> GrowthState:
    today = today or date.today()
    snapshot = medal_ops.build_snapshot(_read_tracker(), today)
    cats = snapshot.get("categories", {})
    discussion = cats.get("discussion", {})

    votes = metadata_tracker.fetch_vote_counts() or {}
    items = [
        ItemState(slug=slug, kind="notebook", votes=int(v), title=slug)
        for slug, v in sorted(votes.items())
    ]
    return GrowthState(
        followers=_read_followers(),
        items=items,
        discussion_medals=int(discussion.get("bronze", 0) or 0),
        discussion_total_posts=int(discussion.get("total_posts", 0) or 0),
        snapshot=snapshot,
    )
