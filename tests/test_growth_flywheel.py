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


# --- Task 3: actions ---------------------------------------------------------
from kaggle_portfolio.growth import actions as actmod
from kaggle_portfolio.growth.state import GrowthState, ItemState


def _state(items=()):
    return GrowthState(followers=0, items=list(items), discussion_medals=0,
                       discussion_total_posts=0, snapshot={})


def test_enumerate_includes_ready_discussion_drafts(tmp_path, monkeypatch):
    q = tmp_path / "discussion_queue.json"
    q.write_text(json.dumps([
        {"id": "057", "title": "Draft A", "status": "ready", "forum": "https://k/f"},
        {"id": "058", "title": "Draft B", "status": "posted", "forum": "https://k/f"},
    ]), encoding="utf-8")
    monkeypatch.setattr(actmod.notebook_promoter, "load_notebooks", lambda: ([], []))
    acts = actmod.enumerate_actions(_state(), discussion_queue_path=q)
    posts = [a for a in acts if a.kind == "discussion_post"]
    assert [a.target_id for a in posts] == ["discussion_post:057"]  # 'posted' excluded


def test_enumerate_includes_forum_drops_for_matched_notebooks(tmp_path, monkeypatch):
    nb = {"slug": "nb-a", "title": "NB A"}
    monkeypatch.setattr(actmod.notebook_promoter, "load_notebooks", lambda: ([nb], []))
    monkeypatch.setattr(actmod.notebook_promoter, "match_notebook_to_competitions",
                        lambda n: ["hull-tactical-market-prediction"])
    monkeypatch.setattr(actmod.notebook_promoter, "generate_promo_comment", lambda n, s: "comment")
    monkeypatch.setattr(actmod.notebook_promoter, "notebook_url", lambda n: "https://k/nb-a")
    gs = _state([ItemState("nb-a", "notebook", 18, "NB A")])
    acts = actmod.enumerate_actions(
        gs, discussion_queue_path=tmp_path / "missing.json",
        audience_by_comp={"hull-tactical-market-prediction": 3677},
    )
    drops = [a for a in acts if a.kind == "forum_drop"]
    assert len(drops) == 1
    assert drops[0].target_id == "forum_drop:nb-a:hull-tactical-market-prediction"
    assert drops[0].audience == 3677
    assert drops[0].item_votes == 18  # carries the item's vote count for scoring


def test_enumerate_never_emits_disallowed_kinds(tmp_path, monkeypatch):
    monkeypatch.setattr(actmod.notebook_promoter, "load_notebooks", lambda: ([], []))
    acts = actmod.enumerate_actions(_state(), discussion_queue_path=tmp_path / "x.json")
    assert all(a.kind in actmod.ALLOWED_KINDS for a in acts)


# --- Task 4: safety ----------------------------------------------------------
from datetime import datetime, timezone
from kaggle_portfolio.growth import safety as safetymod


def _now(hour=15):
    return datetime(2026, 6, 17, hour, 0, tzinfo=timezone.utc)


def _post_action(i):
    return actmod.Action("discussion_post", f"discussion_post:{i}", f"D{i}", {})


def test_kill_switch_blocks_everything():
    cfg = _cfg(enabled=False)
    ranked = [(_post_action("057"), 9.0)]
    assert safetymod.gate(ranked, [], cfg, _now()) == []


def test_posting_window_blocks_off_hours():
    cfg = _cfg(window_start_hour=13, window_end_hour=23)
    assert safetymod.in_posting_window(15, cfg) is True
    assert safetymod.in_posting_window(3, cfg) is False
    ranked = [(_post_action("057"), 9.0)]
    assert safetymod.gate(ranked, [], cfg, _now(hour=3)) == []


def test_dedupe_drops_already_done_targets():
    history = [{"tick_ts": "2026-06-16T15:00:00+00:00", "kind": "discussion_post",
                "target_id": "discussion_post:057", "status": "done"}]
    ranked = [(_post_action("057"), 9.0), (_post_action("058"), 8.0)]
    kept = safetymod.gate(ranked, history, _cfg(), _now())
    assert [a.target_id for a, _ in kept] == ["discussion_post:058"]


def test_daily_cap_limits_remaining_posts():
    today = "2026-06-17T14:00:00+00:00"
    history = [{"tick_ts": today, "kind": "discussion_post",
                "target_id": "discussion_post:055", "status": "done"}]
    cfg = _cfg(max_posts_per_day=2, max_posts_per_week=8)
    ranked = [(_post_action("057"), 9.0), (_post_action("058"), 8.0)]
    kept = safetymod.gate(ranked, history, cfg, _now())
    assert len(kept) == 1  # 1 already done today, cap 2 -> only 1 slot left


def test_failed_history_rows_do_not_consume_caps():
    today = "2026-06-17T14:00:00+00:00"
    history = [{"tick_ts": today, "kind": "discussion_post",
                "target_id": "discussion_post:055", "status": "failed"}]
    cfg = _cfg(max_posts_per_day=2)
    ranked = [(_post_action("057"), 9.0), (_post_action("058"), 8.0)]
    kept = safetymod.gate(ranked, history, cfg, _now())
    assert len(kept) == 2  # failed row doesn't count -> both slots free


# --- Task 5: feedback --------------------------------------------------------
from kaggle_portfolio.growth import feedback as fbmod


def _snap(notebook_votes):
    return {"categories": {"notebooks": {"total_votes": notebook_votes}}}


def test_weights_roundtrip(tmp_path):
    p = tmp_path / "flywheel_weights.json"
    fbmod.save_weights(p, {"discussion_post": 1.4})
    assert fbmod.load_weights(p) == {"discussion_post": 1.4}
    assert fbmod.load_weights(tmp_path / "missing.json") == {}


def test_attribute_credits_recent_action_kind_for_vote_gain():
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    history = [{"tick_ts": "2026-06-17T09:00:00+00:00", "kind": "forum_drop",
                "target_id": "forum_drop:nb-a:hull", "status": "done"}]
    updated = fbmod.attribute(history, _snap(50), _snap(56), {}, _cfg(), now)
    assert updated["forum_drop"] > 1.0
    assert "discussion_post" not in updated or updated["discussion_post"] <= 1.0


def test_attribute_decays_toward_one_when_no_gain():
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    history = [{"tick_ts": "2026-06-17T09:00:00+00:00", "kind": "forum_drop",
                "target_id": "forum_drop:nb-a:hull", "status": "done"}]
    updated = fbmod.attribute(history, _snap(50), _snap(50), {"forum_drop": 2.0}, _cfg(), now)
    assert updated["forum_drop"] < 2.0  # no gain -> EMA pulls the inflated weight down


# --- Task 6: conductor -------------------------------------------------------
import pytest
from kaggle_portfolio.growth import flywheel as fw


def test_tick_dispatches_highest_scored_safe_action(tmp_path, monkeypatch):
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    gs = _state([ItemState("nb-a", "notebook", 18, "NB A")])
    a_low = actmod.Action("discussion_post", "discussion_post:057", "low", {})
    a_high = actmod.Action("forum_drop", "forum_drop:nb-a:hull",
                           "high", {"competition": "hull"}, audience=4000, item_votes=18)
    monkeypatch.setattr(fw, "_load_state", lambda today: gs)
    monkeypatch.setattr(fw.actions, "enumerate_actions", lambda *a, **k: [a_low, a_high])
    monkeypatch.setattr(fw, "GROWTH_DIR", tmp_path)

    dispatched = []
    def fake_exec(action):
        dispatched.append(action.target_id)
        return fw.DispatchResult(ok=True, post_url="https://k/post/1")

    n = fw.tick(now=now, executor=fake_exec,
                cfg=_cfg(max_posts_per_day=1, max_forum_drops_per_comp_per_week=1))
    assert n == 2  # one of each kind fits the caps
    assert "forum_drop:nb-a:hull" in dispatched  # higher score acted
    hist = fw.load_history(tmp_path / "flywheel_history.jsonl")
    assert any(h["status"] == "done" for h in hist)


def test_tick_dry_run_posts_nothing(tmp_path, monkeypatch):
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fw, "_load_state", lambda today: _state())
    monkeypatch.setattr(fw.actions, "enumerate_actions",
                        lambda *a, **k: [actmod.Action("discussion_post", "discussion_post:057", "x", {})])
    monkeypatch.setattr(fw, "GROWTH_DIR", tmp_path)
    called = []
    n = fw.tick(now=now, dry_run=True, executor=lambda a: called.append(a) or fw.DispatchResult(ok=True))
    assert n == 0
    assert called == []  # executor never invoked in dry-run
    assert not (tmp_path / "flywheel_history.jsonl").exists()


def test_tick_failed_dispatch_logged_failed_and_no_cap_consumed(tmp_path, monkeypatch):
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fw, "_load_state", lambda today: _state())
    monkeypatch.setattr(fw.actions, "enumerate_actions",
                        lambda *a, **k: [actmod.Action("discussion_post", "discussion_post:057", "x", {})])
    monkeypatch.setattr(fw, "GROWTH_DIR", tmp_path)
    n = fw.tick(now=now, executor=lambda a: fw.DispatchResult(ok=False, error="captcha"), cfg=_cfg())
    assert n == 0
    hist = fw.load_history(tmp_path / "flywheel_history.jsonl")
    assert hist and hist[-1]["status"] == "failed"


def test_tick_wraps_executor_exception_as_failed(tmp_path, monkeypatch):
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fw, "_load_state", lambda today: _state())
    monkeypatch.setattr(fw.actions, "enumerate_actions",
                        lambda *a, **k: [actmod.Action("discussion_post", "discussion_post:057", "x", {})])
    monkeypatch.setattr(fw, "GROWTH_DIR", tmp_path)

    def boom(action):
        raise RuntimeError("playwright exploded")

    n = fw.tick(now=now, executor=boom, cfg=_cfg())
    assert n == 0
    hist = fw.load_history(tmp_path / "flywheel_history.jsonl")
    assert hist[-1]["status"] == "failed" and "playwright exploded" in hist[-1]["error"]


def test_default_executor_is_safe_stub():
    # Live posting is wired in rollout Phase 2; until then the default must fail
    # loudly (never silently "succeed") so a premature live tick is visible+safe.
    with pytest.raises(NotImplementedError):
        fw._default_executor(actmod.Action("discussion_post", "discussion_post:057", "x", {}))


def test_kill_switch_env_blocks_dispatch(tmp_path, monkeypatch):
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setenv("FLYWHEEL_DISABLED", "1")
    monkeypatch.setattr(fw, "_load_state", lambda today: _state())
    monkeypatch.setattr(fw.actions, "enumerate_actions",
                        lambda *a, **k: [actmod.Action("discussion_post", "discussion_post:057", "x", {})])
    monkeypatch.setattr(fw, "GROWTH_DIR", tmp_path)
    called = []
    n = fw.tick(now=now, executor=lambda a: called.append(a) or fw.DispatchResult(ok=True), cfg=_cfg())
    assert n == 0 and called == []


def test_main_status_runs_offline(monkeypatch, capsys):
    monkeypatch.setattr(fw, "_load_state", lambda today: _state([ItemState("nb-a", "notebook", 18, "A")]))
    rc = fw.main(["status"])
    assert rc == 0
    assert "Reach Score" in capsys.readouterr().out


# --- Task 7: command registration --------------------------------------------
from kaggle_portfolio import manage_commands


def test_flywheel_commands_registered():
    assert "flywheel-tick" in manage_commands.COMMAND_INDEX
    assert "flywheel-status" in manage_commands.COMMAND_INDEX
    assert manage_commands.COMMAND_INDEX["flywheel-tick"].requires_kaggle is True
