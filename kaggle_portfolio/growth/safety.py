"""The dispatch gate: kill switch, posting window, dedupe, rate caps."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from .actions import Action
from .config import FlywheelConfig


def _aware(dt: datetime) -> datetime:
    """Treat a tz-naive datetime as UTC so aware/naive comparisons never crash."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def in_posting_window(hour: int, cfg: FlywheelConfig) -> bool:
    return cfg.window_start_hour <= hour <= cfg.window_end_hour


def _parse(ts: str) -> datetime | None:
    try:
        return _aware(datetime.fromisoformat(ts))
    except (ValueError, TypeError):
        return None


def recent_counts(history: list[dict], now: datetime) -> dict:
    now = _aware(now)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    posts_today = posts_week = 0
    forum_drops_week: dict[str, int] = {}
    for row in history:
        if row.get("status") != "done":
            continue  # only confirmed posts consume caps
        ts = _parse(str(row.get("tick_ts", "")))
        if ts is None:
            continue
        kind = row.get("kind")
        if kind == "discussion_post":
            if ts >= day_ago:
                posts_today += 1
            if ts >= week_ago:
                posts_week += 1
        elif kind == "forum_drop" and ts >= week_ago:
            comp = str(row.get("competition") or "")
            forum_drops_week[comp] = forum_drops_week.get(comp, 0) + 1
    return {"posts_today": posts_today, "posts_week": posts_week,
            "forum_drops_week": forum_drops_week}


def gate(ranked, history, cfg: FlywheelConfig, now: datetime):
    """Filter (action, score) pairs down to the dispatchable, safe subset."""
    now = _aware(now)
    if not cfg.enabled or os.getenv("FLYWHEEL_DISABLED") == "1":
        return []
    if not in_posting_window(now.hour, cfg):
        return []

    done_targets = {r.get("target_id") for r in history if r.get("status") == "done"}
    counts = recent_counts(history, now)
    posts_left_day = cfg.max_posts_per_day - counts["posts_today"]
    posts_left_week = cfg.max_posts_per_week - counts["posts_week"]

    kept = []
    for action, score in ranked:
        if action.target_id in done_targets:
            continue  # dedupe
        if action.kind == "discussion_post":
            if posts_left_day <= 0 or posts_left_week <= 0:
                continue
            posts_left_day -= 1
            posts_left_week -= 1
        elif action.kind == "forum_drop":
            comp = str(action.payload.get("competition") or "")
            used = counts["forum_drops_week"].get(comp, 0)
            if used >= cfg.max_forum_drops_per_comp_per_week:
                continue
            counts["forum_drops_week"][comp] = used + 1
        kept.append((action, score))
    return kept
