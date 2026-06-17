"""Closed feedback loop: attribute vote deltas to recent actions, reweight (EMA)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from .config import FlywheelConfig


def load_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (ValueError, json.JSONDecodeError):
        return {}


def save_weights(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2, sort_keys=True), encoding="utf-8")


def _notebook_votes(snapshot: dict | None) -> int:
    if not snapshot:
        return 0
    return int(snapshot.get("categories", {}).get("notebooks", {}).get("total_votes", 0) or 0)


def attribute(history, prev_snapshot, cur_snapshot, weights, cfg: FlywheelConfig, now: datetime):
    """Time-correlate the vote delta to action kinds taken since the last snapshot,
    then nudge each acting kind's EMA weight toward its observed effectiveness.

    Effectiveness signal per acting kind = (vote_gain / n_acting_done) around a
    1.0 baseline; kinds that acted but saw no gain decay back toward 1.0.
    """
    alpha = cfg.ema_alpha
    gain = max(0, _notebook_votes(cur_snapshot) - _notebook_votes(prev_snapshot))

    window_start = now - timedelta(days=1)
    acting: dict[str, int] = {}
    for row in history:
        if row.get("status") != "done":
            continue
        try:
            ts = datetime.fromisoformat(str(row.get("tick_ts", "")))
        except (ValueError, TypeError):
            continue
        if ts >= window_start:
            acting[row["kind"]] = acting.get(row["kind"], 0) + 1

    updated = dict(weights)
    if not acting:
        return updated
    per_action_gain = gain / sum(acting.values())
    for kind in acting:
        observed = 1.0 + per_action_gain
        prior = updated.get(kind, 1.0)
        updated[kind] = (1 - alpha) * prior + alpha * observed
    return updated
