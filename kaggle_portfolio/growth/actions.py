"""ToS-safe action catalog + generators. Own-content actions only."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kaggle_portfolio.notebooks import notebook_promoter
from .state import GrowthState

ALLOWED_KINDS = frozenset({"discussion_post", "forum_drop", "cross_link"})
_POSTABLE_STATUSES = {"ready", "scheduled"}


@dataclass(frozen=True)
class Action:
    kind: str
    target_id: str
    title: str
    payload: dict = field(default_factory=dict)
    audience: int = 0
    item_votes: int | None = None

    def __post_init__(self):
        if self.kind not in ALLOWED_KINDS:
            raise ValueError(f"Disallowed action kind: {self.kind!r}")


def _discussion_actions(queue_path: Path) -> list[Action]:
    if not queue_path.exists():
        return []
    try:
        entries = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []
    out = []
    for e in entries:
        if str(e.get("status", "")).strip().lower() in _POSTABLE_STATUSES:
            did = str(e.get("id", ""))
            out.append(Action(
                kind="discussion_post",
                target_id=f"discussion_post:{did}",
                title=str(e.get("title", did)),
                # The live queue uses 'forum_url'; tolerate the older 'forum' too.
                payload={"draft_id": did, "forum": e.get("forum_url") or e.get("forum") or ""},
            ))
    return out


def _lookup_audience(comp_slug: str, audience_by_comp: dict[str, int]) -> int:
    """Audience (team count) for a competition slug.

    The tracker keys audience by normalized display name (e.g. 'hull-tactical-
    market') while `notebook_promoter` returns the full Kaggle slug (e.g. 'hull-
    tactical-market-prediction'); match on exact, then either-way containment so
    the two namespaces line up when they overlap.
    """
    if comp_slug in audience_by_comp:
        return audience_by_comp[comp_slug]
    for name, teams in audience_by_comp.items():
        if name and (name in comp_slug or comp_slug in name):
            return teams
    return 0


def _notebook_slug(nb: dict) -> str:
    """Kernel slug for a notebook dict, matching metadata_tracker.fetch_vote_counts
    keys (ref/id tail). load_notebooks() exposes id as 'user/kernel-slug'."""
    nb_id = str(nb.get("id") or nb.get("ref") or "")
    return nb_id.split("/")[-1] if nb_id else str(nb.get("title", ""))


def _forum_drop_actions(gs: GrowthState, audience_by_comp: dict[str, int]) -> list[Action]:
    notebooks, _ = notebook_promoter.load_notebooks()
    votes_by_slug = {i.slug: i.votes for i in gs.items}
    out = []
    for nb in notebooks:
        slug = _notebook_slug(nb)
        for comp in notebook_promoter.match_notebook_to_competitions(nb):
            out.append(Action(
                kind="forum_drop",
                target_id=f"forum_drop:{slug}:{comp}",
                title=f"Share {slug} in {comp}",
                payload={
                    "competition": comp,
                    "comment": notebook_promoter.generate_promo_comment(nb, comp),
                    "notebook_url": notebook_promoter.notebook_url(nb),
                },
                audience=_lookup_audience(comp, audience_by_comp),
                item_votes=votes_by_slug.get(slug),
            ))
    return out


def enumerate_actions(
    gs: GrowthState,
    *,
    discussion_queue_path: Path,
    audience_by_comp: dict[str, int] | None = None,
) -> list[Action]:
    """All candidate actions for this tick (un-scored, un-gated)."""
    audience_by_comp = audience_by_comp or {}
    return _discussion_actions(discussion_queue_path) + _forum_drop_actions(gs, audience_by_comp)
