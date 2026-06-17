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
                payload={"draft_id": did, "forum": e.get("forum", "")},
            ))
    return out


def _forum_drop_actions(gs: GrowthState, audience_by_comp: dict[str, int]) -> list[Action]:
    notebooks, _ = notebook_promoter.load_notebooks()
    votes_by_slug = {i.slug: i.votes for i in gs.items}
    out = []
    for nb in notebooks:
        slug = str(nb.get("slug") or nb.get("ref") or nb.get("title", ""))
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
                audience=int(audience_by_comp.get(comp, 0)),
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
