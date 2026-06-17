"""Closed feedback loop: attribute vote deltas to recent actions, reweight (EMA)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import FlywheelConfig


def _aware(dt: datetime) -> datetime:
    """Treat a tz-naive datetime as UTC so aware/naive comparisons never crash."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def load_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {k: float(v) for k, v in raw.items()}
    except (ValueError, TypeError, json.JSONDecodeError, OSError):
        return {}


def save_weights(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2, sort_keys=True), encoding="utf-8")


def _notebook_votes(snapshot: dict | None) -> int:
    if not snapshot:
        return 0
    cats = snapshot.get("categories") or {}
    notebooks = cats.get("notebooks") or {}
    return int(notebooks.get("total_votes", 0) or 0)


def attribute(history, prev_snapshot, cur_snapshot, weights, cfg: FlywheelConfig, now: datetime):
    """Time-correlate the vote delta to action kinds taken since the last snapshot,
    then nudge each acting kind's EMA weight toward its observed effectiveness.

    Effectiveness signal per acting kind = (vote_gain / n_acting_done) around a
    1.0 baseline; kinds that acted but saw no gain decay back toward 1.0.
    """
    # No prior snapshot -> there is no delta baseline to attribute. Returning the
    # weights unchanged avoids crediting an action with the entire historical
    # vote count on the very first tick (which would inflate its weight ~30x).
    if prev_snapshot is None:
        return dict(weights)

    alpha = cfg.ema_alpha
    now = _aware(now)
    gain = max(0, _notebook_votes(cur_snapshot) - _notebook_votes(prev_snapshot))

    window_start = now - timedelta(days=1)
    acting: dict[str, int] = {}
    for row in history:
        if row.get("status") != "done":
            continue
        try:
            ts = _aware(datetime.fromisoformat(str(row.get("tick_ts", ""))))
        except (ValueError, TypeError):
            continue
        kind = row.get("kind")
        if kind and ts >= window_start:
            acting[kind] = acting.get(kind, 0) + 1

    updated = dict(weights)
    if not acting:
        return updated
    per_action_gain = gain / sum(acting.values())
    for kind in acting:
        observed = 1.0 + per_action_gain
        prior = updated.get(kind, 1.0)
        updated[kind] = (1 - alpha) * prior + alpha * observed
    return updated
