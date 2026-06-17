import json
from pathlib import Path

from kaggle_portfolio.growth import config as cfgmod
from kaggle_portfolio.growth import scorer


def _cfg(**over):
    base = cfgmod.FlywheelConfig()
    return base if not over else base.__class__(**{**base.__dict__, **over})


def test_next_cut_returns_smallest_cut_above_votes():
    assert scorer.next_cut(0) == 5
    assert scorer.next_cut(4) == 5
    assert scorer.next_cut(5) == 20
    assert scorer.next_cut(18) == 20
    assert scorer.next_cut(20) == 50
    assert scorer.next_cut(50) is None  # already gold; nothing higher


def test_vote_progress_applies_near_threshold_bonus():
    cfg = _cfg()
    far = scorer.vote_progress(10, cfg)      # 10 votes, next cut 20, gap 10 (not near)
    near = scorer.vote_progress(18, cfg)     # 18 votes, next cut 20, gap 2 (near -> bonus)
    assert near > far  # the 2-from-silver item must outrank the far one


def test_vote_progress_zero_when_maxed():
    assert scorer.vote_progress(50, _cfg()) == 0.0  # no next cut -> no progress term


def test_reach_score_is_weighted_sum():
    cfg = _cfg(w_followers=5.0, w_votes=1.0, w_discussion=0.5)
    score = scorer.reach_score(followers=10, item_votes=[18], discussion_medals=4, cfg=cfg)
    expected = 5.0 * 10 + 1.0 * scorer.vote_progress(18, cfg) + 0.5 * 4
    assert abs(score - expected) < 1e-9


def test_expected_lift_prefers_near_threshold_item_in_large_forum():
    cfg = _cfg()
    near_big = scorer.expected_lift("forum_drop", audience=4000, item_votes=18, cfg=cfg, weight=1.0)
    cold_post = scorer.expected_lift("discussion_post", audience=0, item_votes=None, cfg=cfg, weight=1.0)
    assert near_big > cold_post


def test_expected_lift_ranks_near_threshold_item_above_far_one():
    # Same kind, same audience: the item 2-from-silver must outrank the far one.
    cfg = _cfg()
    near = scorer.expected_lift("forum_drop", audience=1000, item_votes=18, cfg=cfg, weight=1.0)
    far = scorer.expected_lift("forum_drop", audience=1000, item_votes=6, cfg=cfg, weight=1.0)
    assert near > far


def test_expected_lift_cold_item_still_rankable_by_audience():
    # A 0-vote item scores via the floor, scaled by audience (not a flat zero).
    cfg = _cfg()
    big = scorer.expected_lift("forum_drop", audience=4000, item_votes=0, cfg=cfg, weight=1.0)
    small = scorer.expected_lift("forum_drop", audience=0, item_votes=0, cfg=cfg, weight=1.0)
    assert big > small > 0.0


def test_load_config_defaults_when_missing(tmp_path):
    loaded = cfgmod.load_config(tmp_path / "nope.json")
    assert loaded == cfgmod.FlywheelConfig()


def test_load_config_overrides_from_json(tmp_path):
    p = tmp_path / "flywheel_config.json"
    p.write_text(json.dumps({"max_posts_per_day": 1, "enabled": False}), encoding="utf-8")
    loaded = cfgmod.load_config(p)
    assert loaded.max_posts_per_day == 1
    assert loaded.enabled is False
    assert loaded.w_followers == cfgmod.FlywheelConfig().w_followers  # untouched field keeps default
