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


# --- Task 2: state -----------------------------------------------------------
from datetime import date
from kaggle_portfolio.growth import state as stmod


def test_build_assembles_state_from_snapshot_and_votes(tmp_path, monkeypatch):
    fake_snapshot = {
        "categories": {
            "notebooks": {"total_votes": 50},
            "datasets": {"total_votes": 54},
            "discussion": {"bronze": 4, "total_posts": 9},
        }
    }
    monkeypatch.setattr(stmod.medal_ops, "build_snapshot", lambda content, today: fake_snapshot)
    monkeypatch.setattr(stmod, "_read_tracker", lambda: "ignored")
    monkeypatch.setattr(stmod.metadata_tracker, "fetch_vote_counts", lambda: {"nb-a": 18, "nb-b": 2})
    monkeypatch.setattr(stmod, "GROWTH_DIR", tmp_path)
    (tmp_path / "followers.json").write_text('{"followers": 12}', encoding="utf-8")

    gs = stmod.build(today=date(2026, 6, 17))
    assert gs.followers == 12
    assert gs.discussion_medals == 4
    assert gs.discussion_total_posts == 9
    votes_by_slug = {i.slug: i.votes for i in gs.items}
    assert votes_by_slug == {"nb-a": 18, "nb-b": 2}
    assert all(i.kind == "notebook" for i in gs.items)


def test_build_defaults_followers_to_zero_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(stmod.medal_ops, "build_snapshot",
                        lambda content, today: {"categories": {"discussion": {}}})
    monkeypatch.setattr(stmod, "_read_tracker", lambda: "ignored")
    monkeypatch.setattr(stmod.metadata_tracker, "fetch_vote_counts", lambda: {})
    monkeypatch.setattr(stmod, "GROWTH_DIR", tmp_path)  # no followers.json inside
    gs = stmod.build(today=date(2026, 6, 17))
    assert gs.followers == 0
    assert gs.items == []


def test_build_tolerates_vote_fetch_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(stmod.medal_ops, "build_snapshot",
                        lambda content, today: {"categories": {"discussion": {}}})
    monkeypatch.setattr(stmod, "_read_tracker", lambda: "ignored")
    monkeypatch.setattr(stmod.metadata_tracker, "fetch_vote_counts", lambda: None)  # CLI failed
    monkeypatch.setattr(stmod, "GROWTH_DIR", tmp_path)
    gs = stmod.build(today=date(2026, 6, 17))
    assert gs.items == []  # no items rather than a crash
