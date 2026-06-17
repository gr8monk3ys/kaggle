"""Pure Reach-Score scoring. Imports only `config`; no side effects."""
from __future__ import annotations

from .config import CUTS, TIER_VALUE, FlywheelConfig

# Floor for an item action's proximity so cold (0-vote) items remain rankable
# by audience instead of collapsing to a zero score.
MIN_ITEM_PROXIMITY = 0.1


def next_cut(votes: int) -> int | None:
    """Smallest medal cut strictly greater than `votes`, or None if maxed."""
    for cut in CUTS:
        if votes < cut:
            return cut
    return None


def vote_progress(votes: int, cfg: FlywheelConfig) -> float:
    """Progress toward the next medal cut, with a near-threshold bonus."""
    cut = next_cut(votes)
    if cut is None:
        return 0.0
    base = TIER_VALUE[cut] * (min(votes, cut) / cut)
    if 0 < (cut - votes) <= cfg.near_window:
        base *= cfg.near_bonus
    return base


def reach_score(
    followers: int,
    item_votes: list[int],
    discussion_medals: int,
    cfg: FlywheelConfig,
) -> float:
    """The north-star: followers-weighted composite over current state."""
    votes_term = sum(vote_progress(v, cfg) for v in item_votes)
    return (
        cfg.w_followers * followers
        + cfg.w_votes * votes_term
        + cfg.w_discussion * discussion_medals
    )


def expected_lift(
    action_kind: str,
    audience: int,
    item_votes: int | None,
    cfg: FlywheelConfig,
    weight: float,
) -> float:
    """Estimated Reach-Score gain if this action lands.

    For item-targeted actions (forum drops / cross links), the proximity to the
    next medal cut is the lift signal: `vote_progress` already amplifies items
    within `near_window` of a cut, so near-threshold items dominate. A small
    floor keeps cold items rankable by audience. Generic discussion posts (no
    target item) get a flat discussion weight. Everything scales by audience and
    the action type's learned effectiveness `weight`.
    """
    if item_votes is None:
        proximity = cfg.w_discussion
    else:
        proximity = max(cfg.w_votes * vote_progress(item_votes, cfg), MIN_ITEM_PROXIMITY)
    audience_factor = 1.0 + (audience / 1000.0)
    return weight * proximity * audience_factor
