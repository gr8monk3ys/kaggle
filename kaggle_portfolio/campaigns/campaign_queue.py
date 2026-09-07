#!/usr/bin/env python3
"""The Campaign queue: one model, imported by everything that touches it.

Three modules each held their own copy of this file's shape — ``campaign_pack``
creates the queue, ``campaign_dispatcher`` edits it, ``campaign_execute``
consumes it — with three separate path constants and two separate
reader/writer pairs. That is the shape ``discussions/draft_queue`` was in before
ADR-0003, where the copies drifted on *which item is next* and nobody noticed
until posting was attempted.

Deliberately NOT shared with the Draft Queue. The two look alike from a distance
and are different underneath: Campaigns are a dict envelope rather than a bare
list, their statuses (planned/in_progress/done/blocked) do not overlap with Draft
statuses, and they carry a claim/lease concept Drafts have no equivalent of. A
common base would be abstraction for a similarity that does not exist.

See ``docs/adr/0006-campaign-queue-merge-on-regenerate.md``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_portfolio.shared.errors import CommandError

#: The queue's home. Three modules each carried their own copy of this.
DEFAULT_QUEUE_PATH = Path("pi-automation") / "data" / "promotion_campaign_queue.json"

PLANNED = "planned"
IN_PROGRESS = "in_progress"
DONE = "done"
BLOCKED = "blocked"
VALID_STATUSES = {PLANNED, IN_PROGRESS, DONE, BLOCKED}

#: Statuses whose work is finished or deliberately stopped. Regeneration must
#: never drag one of these back to planned — see merge_regenerated().
TERMINAL_STATUSES = {DONE, BLOCKED}

#: Fields recording that work was claimed or finished. These belong to the run,
#: not to the plan, so a regenerated action must inherit them rather than reset.
CLAIM_FIELDS = ("status", "claimed_at", "claim_count", "completed_at", "note")


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def action_status(action: dict[str, Any]) -> str:
    value = str(action.get("status", PLANNED)).strip().lower()
    return value if value in VALID_STATUSES else PLANNED


# --- file I/O ---------------------------------------------------------------


def load_payload(path: Path) -> dict[str, Any]:
    """Read the queue envelope, or raise with a message a user can act on."""
    if not Path(path).exists():
        raise CommandError(f"Campaign queue not found: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CommandError(f"Invalid campaign queue payload: {path}")
    if not isinstance(payload.get("queue"), list):
        raise CommandError(f"Campaign queue missing 'queue' list: {path}")
    return payload


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_queue(path: Path) -> list[dict[str, Any]]:
    return load_payload(path)["queue"]


# --- ordering and lookup ----------------------------------------------------


def sorted_queue(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        queue,
        key=lambda item: (str(item.get("scheduled_for", "")), str(item.get("id", ""))),
    )


def find_by_id(queue: list[dict[str, Any]], action_id: str) -> dict[str, Any] | None:
    for item in queue:
        if str(item.get("id", "")) == action_id:
            return item
    return None


def filter_actions(
    queue: list[dict[str, Any]],
    *,
    statuses: set[str],
    channels: set[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in sorted_queue(queue):
        if action_status(item) not in statuses:
            continue
        if channels and str(item.get("channel", "")).strip().lower() not in channels:
            continue
        selected.append(item)
        if limit and len(selected) >= limit:
            break
    return selected


def parse_channels(raw_channels: list[str]) -> set[str] | None:
    values = {value.strip().lower() for value in raw_channels if value.strip()}
    return values or None


# --- the claim/lease ---------------------------------------------------------


def claim(action: dict[str, Any], *, stamp: str | None = None) -> bool:
    """Take a planned action, returning whether it was actually claimed.

    Only ``planned`` actions can be claimed. The two callers disagreed here:
    campaign_execute guarded on status, campaign_dispatcher claimed
    unconditionally — so it could re-claim work already in flight or finished and
    bump ``claim_count``, which exists to count retries and cannot mean two
    things at once. A deliberate re-run is ``requeue()``, which says so.
    """
    if action_status(action) != PLANNED:
        return False
    action["status"] = IN_PROGRESS
    action["claimed_at"] = stamp or now_iso()
    action["claim_count"] = int(action.get("claim_count") or 0) + 1
    return True


def requeue(action: dict[str, Any]) -> None:
    """Put an action back to planned so it can be claimed again, on purpose."""
    action["status"] = PLANNED
    action.pop("claimed_at", None)


def complete(
    action: dict[str, Any], *, note: str | None = None, stamp: str | None = None
) -> None:
    action["status"] = DONE
    action["completed_at"] = stamp or now_iso()
    if note:
        action["note"] = note
    action.pop("last_error", None)


def block(action: dict[str, Any], *, reason: str) -> None:
    action["status"] = BLOCKED
    action["note"] = reason


# --- regeneration -----------------------------------------------------------


def merge_regenerated(
    existing: list[dict[str, Any]], regenerated: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fold a freshly generated plan onto the queue, keeping what already ran.

    ``campaign-pack`` used to write the new plan wholesale, dropping
    claim_count/claimed_at/completed_at — so re-running it reset finished work to
    planned and made an already-published promotion eligible to post again. Two
    of the four channels are x and linkedin, where that is a public duplicate.

    Regeneration changes the *plan*; it must not rewrite what the run recorded.
    """
    by_id = {str(item.get("id", "")): item for item in existing}
    merged: list[dict[str, Any]] = []
    for fresh in regenerated:
        action_id = str(fresh.get("id", ""))
        prior = by_id.pop(action_id, None)
        if prior is None:
            merged.append(fresh)
            continue
        combined = {**fresh}
        for field in CLAIM_FIELDS:
            if field in prior:
                combined[field] = prior[field]
        merged.append(combined)
    # Anything the new plan no longer proposes but which already ran stays, so
    # history is not lost when a Dataset drops out of the campaign criteria.
    merged.extend(
        item for item in by_id.values() if action_status(item) in TERMINAL_STATUSES
    )
    return merged
