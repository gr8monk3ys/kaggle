"""The Conductor: orchestrate state->enumerate->score->gate->dispatch->log->attribute."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from . import actions, feedback, safety, scorer
from .config import FlywheelConfig, load_config
from .state import GROWTH_DIR as _STATE_GROWTH_DIR
from .state import build as _build_state

GROWTH_DIR = _STATE_GROWTH_DIR
HISTORY_NAME = "flywheel_history.jsonl"
WEIGHTS_NAME = "flywheel_weights.json"
CONFIG_NAME = "flywheel_config.json"
LAST_SNAPSHOT_NAME = "flywheel_last_snapshot.json"
_QUEUE_PATH = Path(__file__).resolve().parents[2] / "pi-automation" / "data" / "discussion_queue.json"


@dataclass(frozen=True)
class DispatchResult:
    ok: bool
    post_url: str | None = None
    error: str | None = None


def _load_state(today: date):
    return _build_state(today)


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip a corrupted line; never drop the whole history
                       # (losing it would break dedupe -> risk double-posting)
    return rows


def append_history(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_last_snapshot() -> dict | None:
    path = GROWTH_DIR / LAST_SNAPSHOT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_last_snapshot(snapshot: dict) -> None:
    path = GROWTH_DIR / LAST_SNAPSHOT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")


def _default_executor(action: actions.Action) -> DispatchResult:  # pragma: no cover - live only
    """Live dispatch seam.

    Posting is wired during rollout Phase 2 (see the design spec): `discussion_post`
    through the discussion poster and `forum_drop` once `notebook_promoter --auto`
    lands. Until then this raises loudly so a live `flywheel-tick` (without
    `--dry-run`) fails SAFE and visibly instead of silently pretending to post.
    `tick()` wraps this call and records a `failed` history row, so the engine
    no-ops rather than crashing. Run `flywheel-tick --dry-run` until this is wired.
    """
    raise NotImplementedError(
        f"live dispatch for {action.kind!r} is wired during rollout Phase 2; "
        "use `flywheel-tick --dry-run` until then"
    )


def _coerce_int(value) -> int:
    """Parse an int from messy tracker strings like '3,677' or '—'; 0 on failure."""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0


def _audience_by_comp(gs) -> dict[str, int]:
    out = {}
    for comp in (gs.snapshot.get("active_competitions") or []):
        name = str(comp.get("competition", "")).strip().lower().replace(" ", "-")
        teams = _coerce_int(comp.get("teams"))
        if name and teams:
            out[name] = teams
    return out


def _ranked(gs, cfg: FlywheelConfig, weights: dict[str, float]):
    candidates = actions.enumerate_actions(
        gs, discussion_queue_path=_QUEUE_PATH, audience_by_comp=_audience_by_comp(gs),
    )
    scored = [
        (a, scorer.expected_lift(a.kind, a.audience, a.item_votes, cfg, weights.get(a.kind, 1.0)))
        for a in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def tick(*, now: datetime | None = None, dry_run: bool = False,
         executor=None, gs=None, cfg=None) -> int:
    now = now or datetime.now(timezone.utc)
    executor = executor or _default_executor
    cfg = cfg or load_config(GROWTH_DIR / CONFIG_NAME)
    gs = gs or _load_state(now.date())

    history_path = GROWTH_DIR / HISTORY_NAME
    history = load_history(history_path)
    weights = feedback.load_weights(GROWTH_DIR / WEIGHTS_NAME)

    ranked = _ranked(gs, cfg, weights)
    safe = safety.gate(ranked, history, cfg, now)

    if dry_run:
        for action, score in safe:
            print(f"WOULD DISPATCH [{score:.2f}] {action.kind}: {action.target_id}")
        return 0

    prev_snapshot = _load_last_snapshot()
    dispatched = 0
    for action, score in safe:
        try:
            result = executor(action)
        except Exception as exc:  # wrap every executor call (account-health guard)
            result = DispatchResult(ok=False, error=str(exc))
        append_history(history_path, {
            "tick_ts": now.isoformat(),
            "kind": action.kind,
            "target_id": action.target_id,
            "competition": action.payload.get("competition"),
            "score": round(score, 3),
            "status": "done" if result.ok else "failed",
            "post_url": result.post_url,
            "error": result.error,
        })
        if result.ok:
            dispatched += 1
        else:
            break  # stop on first failure (captcha/login/not-wired) this tick

    # Learn + advance the attribution baseline only when we actually acted, so a
    # capped / closed-window / killed tick doesn't drift the baseline and starve
    # the next real dispatch of its vote delta.
    if dispatched > 0:
        new_weights = feedback.attribute(
            load_history(history_path), prev_snapshot, gs.snapshot, weights, cfg, now,
        )
        feedback.save_weights(GROWTH_DIR / WEIGHTS_NAME, new_weights)
        _save_last_snapshot(gs.snapshot)
    return dispatched


def status(*, gs=None, cfg=None) -> int:
    cfg = cfg or load_config(GROWTH_DIR / CONFIG_NAME)
    gs = gs or _load_state(date.today())
    score = scorer.reach_score(gs.followers, [i.votes for i in gs.items],
                               gs.discussion_medals, cfg)
    near = [i for i in gs.items
            if (c := scorer.next_cut(i.votes)) is not None and 0 < (c - i.votes) <= cfg.near_window]
    print(f"Reach Score: {score:.2f}")
    print(f"Followers: {gs.followers}  |  Notebooks tracked: {len(gs.items)}  |  "
          f"Discussion medals: {gs.discussion_medals}")
    print(f"Near-threshold items ({len(near)}): "
          + ", ".join(f"{i.slug}={i.votes}" for i in near[:10]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flywheel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tick_p = sub.add_parser("tick", help="Score, gate, and dispatch the top safe actions.")
    tick_p.add_argument("--dry-run", action="store_true", help="Show would-dispatch; post nothing.")
    sub.add_parser("status", help="Print the Reach-Score dashboard.")
    args = parser.parse_args(argv)
    if args.command == "tick":
        tick(dry_run=args.dry_run)
        return 0
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
