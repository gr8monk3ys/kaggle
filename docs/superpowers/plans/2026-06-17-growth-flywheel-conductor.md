# Growth Flywheel "Conductor" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous, self-tuning engagement engine (`kaggle_portfolio/growth/`) that scores and dispatches the highest-leverage Kaggle promotion action each cron tick, then attributes outcomes and reweights — converting already-published content into votes/followers/medals.

**Architecture:** A "Conductor" that adds only the two missing organs — a pure **Reach-Score scorer** and a closed **measure→attribute→reweight feedback loop** — on top of existing executors. Each tick: build read-only state → enumerate candidate actions from existing generators → score → safety-gate → dispatch top action through an injectable executor seam → log → attribute. All logic lives in the tested `kaggle_portfolio/` package; `pi-automation/` stays a thin cron caller.

**Tech Stack:** Python 3.9+ stdlib (`dataclasses`, `pathlib`, `json`, `datetime`, `argparse`), pytest with `monkeypatch`/`tmp_path` (offline, no real Kaggle calls). Reuses `kaggle_portfolio.ops.medal_ops`, `kaggle_portfolio.ops.metadata_tracker`, `kaggle_portfolio.notebooks.notebook_promoter`, `kaggle_portfolio.campaigns.campaign_execute`.

## Global Constraints

- **Package location:** all new code under `kaggle_portfolio/growth/`; the package is run via `PYTHONPATH` (not pip-installed) — no new top-level `*.py` at repo root (guardrail `test_repo_root_has_no_top_level_python_scripts`).
- **Tests are fully offline:** never call the real Kaggle CLI or Playwright; inject/monkeypatch. Run from repo root with `python -m pytest`.
- **No inauthentic actions, ever:** the action catalog contains only `discussion_post`, `forum_drop`, `cross_link`. No auto-follow, auto-upvote, or unsolicited replies — these must not appear anywhere in code.
- **Autonomy is gated:** dispatch must respect the kill switch (`FlywheelConfig.enabled` / `FLYWHEEL_DISABLED=1`), rate caps, posting window, and dedupe before any post.
- **State stores** live under `medal_ops/growth/` as JSON: `flywheel_history.jsonl`, `flywheel_weights.json`, `flywheel_config.json`, `followers.json`. `medal_ops/history/` is git-tracked (not ignored) — follow the same convention for `medal_ops/growth/`.
- **Medal vote cuts:** bronze=5, silver=20, gold=50 (from the tracker's Medal Thresholds table).
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|---|---|
| `kaggle_portfolio/growth/__init__.py` | Package marker (docstring only). |
| `kaggle_portfolio/growth/config.py` | Leaf: `FlywheelConfig` dataclass, `CUTS`, `TIER_VALUE`, `load_config()`. No intra-package imports. |
| `kaggle_portfolio/growth/scorer.py` | Pure scoring: `next_cut`, `vote_progress`, `reach_score`, `expected_lift`. Imports only `config`. |
| `kaggle_portfolio/growth/state.py` | `ItemState`, `GrowthState`, `build()` — read-only snapshot from `medal_ops` + `metadata_tracker` + `followers.json`. |
| `kaggle_portfolio/growth/actions.py` | `Action` dataclass + `enumerate_actions()` generators (discussion / forum-drop / cross-link). |
| `kaggle_portfolio/growth/safety.py` | The gate: `in_posting_window`, `recent_counts`, `gate()`. Pure over (history, config, now). |
| `kaggle_portfolio/growth/feedback.py` | `load_weights`, `save_weights`, `attribute()` — EMA reweight from snapshot deltas. |
| `kaggle_portfolio/growth/flywheel.py` | Conductor: history I/O, `DispatchResult`, `_default_executor`, `tick()`, `status()`, `main()` (argparse). |
| `kaggle_portfolio/manage_commands.py` | Register `flywheel-tick` + `flywheel-status` Commands (modify). |
| `pi-automation/crontab` | Phase-3 cutover: swap blind `discussion_post.py` for `flywheel-tick` (modify). |
| `tests/test_growth_flywheel.py` | All unit + integration tests (new). |

Dependency order (leaf → root): `config` → `scorer` → `state`/`actions`/`safety`/`feedback` → `flywheel` → `manage_commands`. Tasks follow this order so each builds only on already-tested code.

---

### Task 1: Scaffold + config + pure scorer

**Files:**
- Create: `kaggle_portfolio/growth/__init__.py`
- Create: `kaggle_portfolio/growth/config.py`
- Create: `kaggle_portfolio/growth/scorer.py`
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Produces:
  - `config.FlywheelConfig` (frozen dataclass) with fields: `w_followers: float=5.0`, `w_votes: float=1.0`, `w_discussion: float=0.5`, `near_bonus: float=3.0`, `near_window: int=3`, `max_posts_per_day: int=2`, `max_posts_per_week: int=8`, `max_forum_drops_per_comp_per_week: int=1`, `window_start_hour: int=13`, `window_end_hour: int=23`, `enabled: bool=True`, `ema_alpha: float=0.3`.
  - `config.CUTS: tuple[int,int,int] = (5, 20, 50)`
  - `config.TIER_VALUE: dict[int,float] = {5: 1.0, 20: 2.5, 50: 6.0}`
  - `config.load_config(path: Path) -> FlywheelConfig`
  - `scorer.next_cut(votes: int) -> int | None`
  - `scorer.vote_progress(votes: int, cfg: FlywheelConfig) -> float`
  - `scorer.reach_score(followers: int, item_votes: list[int], discussion_medals: int, cfg: FlywheelConfig) -> float`
  - `scorer.expected_lift(action_kind: str, audience: int, item_votes: int | None, cfg: FlywheelConfig, weight: float) -> float`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_growth_flywheel.py`:

```python
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
    near = scorer.vote_progress(18, cfg)     # 18 votes, next cut 20, gap 2 (near → bonus)
    assert near > far  # the 2-from-silver item must outrank the far one


def test_vote_progress_zero_when_maxed():
    assert scorer.vote_progress(50, _cfg()) == 0.0  # no next cut → no progress term


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_growth_flywheel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_portfolio.growth'`.

- [ ] **Step 3: Create the package marker**

Create `kaggle_portfolio/growth/__init__.py`:

```python
"""Growth flywheel: autonomous, self-tuning Kaggle engagement engine."""
```

- [ ] **Step 4: Implement `config.py`**

Create `kaggle_portfolio/growth/config.py`:

```python
"""Configuration leaf for the growth flywheel. No intra-package imports."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

# Medal vote cuts: bronze, silver, gold.
CUTS: tuple[int, int, int] = (5, 20, 50)
# Relative value of reaching each cut (gold worth most).
TIER_VALUE: dict[int, float] = {5: 1.0, 20: 2.5, 50: 6.0}


@dataclass(frozen=True)
class FlywheelConfig:
    w_followers: float = 5.0
    w_votes: float = 1.0
    w_discussion: float = 0.5
    near_bonus: float = 3.0
    near_window: int = 3
    max_posts_per_day: int = 2
    max_posts_per_week: int = 8
    max_forum_drops_per_comp_per_week: int = 1
    window_start_hour: int = 13  # inclusive UTC hour
    window_end_hour: int = 23    # inclusive UTC hour
    enabled: bool = True
    ema_alpha: float = 0.3


def load_config(path: Path) -> FlywheelConfig:
    """Load config from JSON, ignoring unknown keys; missing file → defaults."""
    if not path.exists():
        return FlywheelConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(FlywheelConfig)}
    return FlywheelConfig(**{k: v for k, v in raw.items() if k in known})
```

- [ ] **Step 5: Implement `scorer.py`**

Create `kaggle_portfolio/growth/scorer.py`:

```python
"""Pure Reach-Score scoring. Imports only `config`; no side effects."""
from __future__ import annotations

from .config import CUTS, TIER_VALUE, FlywheelConfig


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

    Heuristic: the marginal vote_progress an extra vote would add to the target
    item (huge near a threshold), scaled by an audience factor and the action
    type's learned effectiveness `weight`.
    """
    if item_votes is None:
        marginal = cfg.w_discussion  # a generic post: small, flat discussion value
    else:
        marginal = cfg.w_votes * (
            vote_progress(item_votes + 1, cfg) - vote_progress(item_votes, cfg)
        )
        # A brand-new vote on a near-threshold item can be negative-delta only
        # if it crosses the cut (progress resets to next tier); clamp to tier value.
        marginal = max(marginal, cfg.w_votes * TIER_VALUE[next_cut(item_votes) or 50])
    audience_factor = 1.0 + (audience / 1000.0)
    return weight * marginal * audience_factor
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_growth_flywheel.py -v`
Expected: PASS (7 tests).

- [ ] **Step 7: Commit**

```bash
git add kaggle_portfolio/growth/__init__.py kaggle_portfolio/growth/config.py kaggle_portfolio/growth/scorer.py tests/test_growth_flywheel.py
git commit -m "feat(growth): add config + pure Reach-Score scorer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `state.py` — read-only GrowthState

**Files:**
- Create: `kaggle_portfolio/growth/state.py`
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Consumes: `medal_ops.build_snapshot(content: str, today: date) -> dict`, `metadata_tracker.fetch_vote_counts() -> dict[str,int] | None`.
- Produces:
  - `state.ItemState` (frozen): `slug: str`, `kind: str`, `votes: int`, `title: str`
  - `state.GrowthState` (frozen): `followers: int`, `items: list[ItemState]`, `discussion_medals: int`, `discussion_total_posts: int`, `snapshot: dict`
  - `state.GROWTH_DIR: Path` (= repo_root / "medal_ops" / "growth")
  - `state.TRACKER_PATH: Path` (= repo_root / "docs" / "reports" / "grandmaster-tracker.md")
  - `state.build(today: date | None = None) -> GrowthState`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_growth_flywheel.py`:

```python
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
    # followers file present
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_growth_flywheel.py -k build -v`
Expected: FAIL — `AttributeError: module 'kaggle_portfolio.growth.state'` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement `state.py`**

Create `kaggle_portfolio/growth/state.py`:

```python
"""Read-only GrowthState: current votes/followers/medals from existing trackers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from kaggle_portfolio.ops import medal_ops
from kaggle_portfolio.ops import metadata_tracker

ROOT = Path(__file__).resolve().parents[2]
TRACKER_PATH = ROOT / "docs" / "reports" / "grandmaster-tracker.md"
GROWTH_DIR = ROOT / "medal_ops" / "growth"


@dataclass(frozen=True)
class ItemState:
    slug: str
    kind: str  # "notebook" | "dataset"
    votes: int
    title: str


@dataclass(frozen=True)
class GrowthState:
    followers: int
    items: list[ItemState]
    discussion_medals: int
    discussion_total_posts: int
    snapshot: dict = field(default_factory=dict)


def _read_tracker() -> str:
    return TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.exists() else ""


def _read_followers() -> int:
    path = GROWTH_DIR / "followers.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("followers", 0))
    except (ValueError, json.JSONDecodeError):
        return 0


def build(today: date | None = None) -> GrowthState:
    today = today or date.today()
    snapshot = medal_ops.build_snapshot(_read_tracker(), today)
    cats = snapshot.get("categories", {})
    discussion = cats.get("discussion", {})

    votes = metadata_tracker.fetch_vote_counts() or {}
    items = [
        ItemState(slug=slug, kind="notebook", votes=int(v), title=slug)
        for slug, v in sorted(votes.items())
    ]
    return GrowthState(
        followers=_read_followers(),
        items=items,
        discussion_medals=int(discussion.get("bronze", 0) or 0),
        discussion_total_posts=int(discussion.get("total_posts", 0) or 0),
        snapshot=snapshot,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_growth_flywheel.py -k build -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/growth/state.py tests/test_growth_flywheel.py
git commit -m "feat(growth): add read-only GrowthState builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `actions.py` — ToS-safe action catalog + generators

**Files:**
- Create: `kaggle_portfolio/growth/actions.py`
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Consumes: `state.GrowthState`, `state.ItemState`; `notebook_promoter.load_notebooks() -> tuple[list[dict], list[str]]`, `notebook_promoter.match_notebook_to_competitions(nb) -> list[str]`, `notebook_promoter.generate_promo_comment(nb, slug) -> str`, `notebook_promoter.notebook_url(nb) -> str`.
- Produces:
  - `actions.Action` (frozen): `kind: str`, `target_id: str`, `title: str`, `payload: dict`, `audience: int = 0`, `item_votes: int | None = None`
  - `actions.ALLOWED_KINDS: frozenset = frozenset({"discussion_post", "forum_drop", "cross_link"})`
  - `actions.enumerate_actions(gs: GrowthState, *, discussion_queue_path: Path, audience_by_comp: dict[str,int] | None = None) -> list[Action]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_growth_flywheel.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_growth_flywheel.py -k enumerate -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_portfolio.growth.actions'`.

- [ ] **Step 3: Implement `actions.py`**

Create `kaggle_portfolio/growth/actions.py`:

```python
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
```

(Cross-link generation is a Phase-3 extension; the `cross_link` kind is in `ALLOWED_KINDS` and the `Action` contract supports it, but no generator emits it in v1 — keep YAGNI until forum-drops + discussion are proven.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_growth_flywheel.py -k enumerate -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/growth/actions.py tests/test_growth_flywheel.py
git commit -m "feat(growth): add ToS-safe action catalog + generators

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `safety.py` — the dispatch gate

**Files:**
- Create: `kaggle_portfolio/growth/safety.py`
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Consumes: `actions.Action`, `config.FlywheelConfig`. History entries are dicts shaped like: `{"tick_ts": iso, "kind": str, "target_id": str, "competition": str | None, "status": "done" | "failed"}`.
- Produces:
  - `safety.in_posting_window(hour: int, cfg: FlywheelConfig) -> bool`
  - `safety.recent_counts(history: list[dict], now: datetime) -> dict` → `{"posts_today": int, "posts_week": int, "forum_drops_week": dict[str,int]}` (only `status=="done"` rows count)
  - `safety.gate(ranked: list[tuple[Action, float]], history: list[dict], cfg: FlywheelConfig, now: datetime) -> list[tuple[Action, float]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_growth_flywheel.py`:

```python
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
    assert len(kept) == 1  # 1 already done today, cap 2 → only 1 slot left


def test_failed_history_rows_do_not_consume_caps():
    today = "2026-06-17T14:00:00+00:00"
    history = [{"tick_ts": today, "kind": "discussion_post",
                "target_id": "discussion_post:055", "status": "failed"}]
    cfg = _cfg(max_posts_per_day=2)
    ranked = [(_post_action("057"), 9.0), (_post_action("058"), 8.0)]
    kept = safetymod.gate(ranked, history, cfg, _now())
    assert len(kept) == 2  # failed row doesn't count → both slots free
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_growth_flywheel.py -k "gate or window or dedupe or cap or failed_history" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_portfolio.growth.safety'`.

- [ ] **Step 3: Implement `safety.py`**

Create `kaggle_portfolio/growth/safety.py`:

```python
"""The dispatch gate: kill switch, posting window, dedupe, rate caps."""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from .actions import Action
from .config import FlywheelConfig


def in_posting_window(hour: int, cfg: FlywheelConfig) -> bool:
    return cfg.window_start_hour <= hour <= cfg.window_end_hour


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def recent_counts(history: list[dict], now: datetime) -> dict:
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    posts_today = posts_week = 0
    forum_drops_week: dict[str, int] = {}
    for row in history:
        if row.get("status") != "done":
            continue  # only confirmed posts consume caps
        ts = _parse(str(row.get("tick_ts", "")))
        if ts is None:
            continue
        kind = row.get("kind")
        if kind == "discussion_post":
            if ts >= day_ago:
                posts_today += 1
            if ts >= week_ago:
                posts_week += 1
        elif kind == "forum_drop" and ts >= week_ago:
            comp = str(row.get("competition", ""))
            forum_drops_week[comp] = forum_drops_week.get(comp, 0) + 1
    return {"posts_today": posts_today, "posts_week": posts_week,
            "forum_drops_week": forum_drops_week}


def gate(ranked, history, cfg: FlywheelConfig, now: datetime):
    """Filter (action, score) pairs down to the dispatchable, safe subset."""
    if not cfg.enabled or os.getenv("FLYWHEEL_DISABLED") == "1":
        return []
    if not in_posting_window(now.hour, cfg):
        return []

    done_targets = {r.get("target_id") for r in history if r.get("status") == "done"}
    counts = recent_counts(history, now)
    posts_left_day = cfg.max_posts_per_day - counts["posts_today"]
    posts_left_week = cfg.max_posts_per_week - counts["posts_week"]

    kept = []
    for action, score in ranked:
        if action.target_id in done_targets:
            continue  # dedupe
        if action.kind == "discussion_post":
            if posts_left_day <= 0 or posts_left_week <= 0:
                continue
            posts_left_day -= 1
            posts_left_week -= 1
        elif action.kind == "forum_drop":
            comp = str(action.payload.get("competition", ""))
            used = counts["forum_drops_week"].get(comp, 0)
            if used >= cfg.max_forum_drops_per_comp_per_week:
                continue
            counts["forum_drops_week"][comp] = used + 1
        kept.append((action, score))
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_growth_flywheel.py -k "gate or window or dedupe or cap or failed_history" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/growth/safety.py tests/test_growth_flywheel.py
git commit -m "feat(growth): add dispatch safety gate (caps, window, dedupe, kill switch)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `feedback.py` — closed attribution + EMA reweight

**Files:**
- Create: `kaggle_portfolio/growth/feedback.py`
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Consumes: `config.FlywheelConfig`, history rows (Task 4 shape). Snapshot dicts are `medal_ops.build_snapshot` outputs; total notebook votes live at `snapshot["categories"]["notebooks"]["total_votes"]`.
- Produces:
  - `feedback.load_weights(path: Path) -> dict[str,float]` (missing → `{}`)
  - `feedback.save_weights(path: Path, weights: dict[str,float]) -> None`
  - `feedback.attribute(history: list[dict], prev_snapshot: dict | None, cur_snapshot: dict, weights: dict[str,float], cfg: FlywheelConfig, now: datetime) -> dict[str,float]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_growth_flywheel.py`:

```python
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
    weights = {}  # start neutral
    updated = fbmod.attribute(history, _snap(50), _snap(56), weights, _cfg(), now)
    # +6 votes since last snapshot, only forum_drop acted → its weight rises above 1.0
    assert updated["forum_drop"] > 1.0
    assert "discussion_post" not in updated or updated["discussion_post"] <= 1.0


def test_attribute_decays_toward_one_when_no_gain():
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    history = [{"tick_ts": "2026-06-17T09:00:00+00:00", "kind": "forum_drop",
                "target_id": "forum_drop:nb-a:hull", "status": "done"}]
    updated = fbmod.attribute(history, _snap(50), _snap(50), {"forum_drop": 2.0}, _cfg(), now)
    assert updated["forum_drop"] < 2.0  # no gain → EMA pulls the inflated weight back down
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_growth_flywheel.py -k "weights or attribute" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_portfolio.growth.feedback'`.

- [ ] **Step 3: Implement `feedback.py`**

Create `kaggle_portfolio/growth/feedback.py`:

```python
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

    Effectiveness signal per acting kind = (vote_gain / n_acting_done) normalized
    around 1.0; kinds that acted but saw no gain decay back toward 1.0.
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
    # observed effectiveness target: 1.0 baseline + observed gain per action
    for kind in acting:
        observed = 1.0 + per_action_gain
        prior = updated.get(kind, 1.0)
        updated[kind] = (1 - alpha) * prior + alpha * observed
    return updated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_growth_flywheel.py -k "weights or attribute" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/growth/feedback.py tests/test_growth_flywheel.py
git commit -m "feat(growth): add closed feedback loop (attribution + EMA reweight)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `flywheel.py` — the Conductor + CLI

**Files:**
- Create: `kaggle_portfolio/growth/flywheel.py`
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Consumes: every prior module (`config`, `scorer`, `state`, `actions`, `safety`, `feedback`).
- Produces:
  - `flywheel.DispatchResult` (frozen): `ok: bool`, `post_url: str | None = None`, `error: str | None = None`
  - `flywheel.load_history(path: Path) -> list[dict]`
  - `flywheel.append_history(path: Path, entry: dict) -> None`
  - `flywheel.tick(*, now: datetime | None = None, dry_run: bool = False, executor=None, gs=None, cfg=None) -> int` (returns count dispatched)
  - `flywheel.status(*, gs=None, cfg=None) -> int`
  - `flywheel.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_growth_flywheel.py`:

```python
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

    n = fw.tick(now=now, executor=fake_exec, cfg=_cfg(max_posts_per_day=1, max_forum_drops_per_comp_per_week=1))
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
    assert not (tmp_path / "flywheel_history.jsonl").exists()  # no live history rows written


def test_tick_failed_dispatch_logged_failed_and_no_cap_consumed(tmp_path, monkeypatch):
    now = datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(fw, "_load_state", lambda today: _state())
    monkeypatch.setattr(fw.actions, "enumerate_actions",
                        lambda *a, **k: [actmod.Action("discussion_post", "discussion_post:057", "x", {})])
    monkeypatch.setattr(fw, "GROWTH_DIR", tmp_path)
    n = fw.tick(now=now, executor=lambda a: fw.DispatchResult(ok=False, error="captcha"),
                cfg=_cfg())
    assert n == 0
    hist = fw.load_history(tmp_path / "flywheel_history.jsonl")
    assert hist and hist[-1]["status"] == "failed"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_growth_flywheel.py -k "tick or kill_switch_env or main_status" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kaggle_portfolio.growth.flywheel'`.

- [ ] **Step 3: Implement `flywheel.py`**

Create `kaggle_portfolio/growth/flywheel.py`:

```python
"""The Conductor: orchestrate state→enumerate→score→gate→dispatch→log→attribute."""
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
        if line:
            rows.append(json.loads(line))
    return rows


def append_history(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _default_executor(action: actions.Action) -> DispatchResult:  # pragma: no cover - live only
    """Real dispatch via existing Playwright/CLI executors. Exercised only on
    Kaggle/live; unit tests inject a fake executor."""
    from kaggle_portfolio.campaigns import campaign_execute  # local import: Playwright optional
    try:
        # Route through the existing campaign queue executor path.
        post_url = campaign_execute.post_action_via_playwright(action.kind, action.payload)
        return DispatchResult(ok=True, post_url=post_url)
    except Exception as exc:  # captcha/login/etc. → caller trips health guard
        return DispatchResult(ok=False, error=str(exc))


def _ranked(gs, cfg: FlywheelConfig, weights: dict[str, float]):
    candidates = actions.enumerate_actions(
        gs,
        discussion_queue_path=Path(__file__).resolve().parents[2]
        / "pi-automation" / "data" / "discussion_queue.json",
        audience_by_comp=_audience_by_comp(gs),
    )
    scored = [
        (a, scorer.expected_lift(a.kind, a.audience, a.item_votes, cfg, weights.get(a.kind, 1.0)))
        for a in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _audience_by_comp(gs) -> dict[str, int]:
    out = {}
    for comp in gs.snapshot.get("active_competitions", []):
        name = str(comp.get("competition", "")).lower().replace(" ", "-")
        if comp.get("teams"):
            out[name] = int(comp["teams"])
    return out


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

    dispatched = 0
    for action, score in safe:
        result = executor(action)
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
            break  # health guard: stop on first failure (captcha/login) this tick

    new_weights = feedback.attribute(
        load_history(history_path), None, gs.snapshot, weights, cfg, now,
    )
    feedback.save_weights(GROWTH_DIR / WEIGHTS_NAME, new_weights)
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
        return 0 if tick(dry_run=args.dry_run) >= 0 else 1
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Note for the implementer:** `_default_executor` references `campaign_execute.post_action_via_playwright(kind, payload)`, a thin live-only helper. If that exact function does not yet exist in `kaggle_portfolio/campaigns/campaign_execute.py`, add it in this task as a small wrapper around the module's existing Playwright posting path (`require_playwright()` + `load_payload`/`claim_action`/`mark_done`), routing `discussion_post`/`forum_drop` through it. It is marked `# pragma: no cover` because it is exercised only on Kaggle; all unit tests inject a fake executor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_growth_flywheel.py -k "tick or kill_switch_env or main_status" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full growth suite**

Run: `python -m pytest tests/test_growth_flywheel.py -v`
Expected: PASS (all tests from Tasks 1–6).

- [ ] **Step 6: Commit**

```bash
git add kaggle_portfolio/growth/flywheel.py kaggle_portfolio/campaigns/campaign_execute.py tests/test_growth_flywheel.py
git commit -m "feat(growth): add Conductor tick + status CLI with injectable executor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Register `flywheel-tick` + `flywheel-status` commands

**Files:**
- Modify: `kaggle_portfolio/manage_commands.py` (the `COMMANDS = [...]` list, ~line 804–851)
- Test: `tests/test_growth_flywheel.py`

**Interfaces:**
- Consumes: `manage_commands.run_module`, `manage_commands.Command`, `manage_commands.COMMAND_INDEX`.
- Produces: two new commands invokable as `./manage.sh flywheel-tick [--dry-run]` and `./manage.sh flywheel-status`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_growth_flywheel.py`:

```python
from kaggle_portfolio import manage_commands


def test_flywheel_commands_registered():
    assert "flywheel-tick" in manage_commands.COMMAND_INDEX
    assert "flywheel-status" in manage_commands.COMMAND_INDEX
    assert manage_commands.COMMAND_INDEX["flywheel-tick"].requires_kaggle is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_growth_flywheel.py -k flywheel_commands_registered -v`
Expected: FAIL — `KeyError: 'flywheel-tick'` (assertion error).

- [ ] **Step 3: Add the commands**

In `kaggle_portfolio/manage_commands.py`, inside the `COMMANDS = [...]` list (next to the other `run_module` delegations, e.g. after the `scout` command near line 840), add:

```python
    Command("flywheel-status", "Print the growth-flywheel Reach-Score dashboard",
            lambda a: run_module("kaggle_portfolio.growth.flywheel", ["status", *a])),
    Command("flywheel-tick", "Run one growth-flywheel tick: score, gate, dispatch top safe actions",
            lambda a: run_module("kaggle_portfolio.growth.flywheel", ["tick", *a]),
            "[--dry-run]", True),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_growth_flywheel.py -k flywheel_commands_registered -v`
Expected: PASS.

- [ ] **Step 5: Verify the CLI lists them end-to-end**

Run: `./manage.sh help | grep flywheel`
Expected output (two lines):
```
  flywheel-status                    Print the growth-flywheel Reach-Score dashboard
  flywheel-tick                      Run one growth-flywheel tick: score, gate, dispatch top safe actions
```

Run: `./manage.sh flywheel-status`
Expected: prints a `Reach Score: …` dashboard (runs offline against the live tracker; followers default 0 if `medal_ops/growth/followers.json` is absent).

- [ ] **Step 6: Commit**

```bash
git add kaggle_portfolio/manage_commands.py tests/test_growth_flywheel.py
git commit -m "feat(growth): register flywheel-tick + flywheel-status commands

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Phase-3 cron cutover (gated on dry-run observation)

> **Do this task only after Phases 1–2 of the rollout** (observe `flywheel-status` + `flywheel-tick --dry-run` for several days, then run live discussion-only at low caps and confirm `flywheel_history.jsonl` shows clean `done` rows). It replaces the blind poster; do not land it before the engine has been observed.

**Files:**
- Modify: `pi-automation/crontab`
- Create: `medal_ops/growth/flywheel_config.json` (seed config; committed so the cron has explicit caps + kill switch)
- Test: `tests/test_growth_flywheel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_growth_flywheel.py`:

```python
def test_crontab_uses_flywheel_tick(repo_root):
    crontab = (repo_root / "pi-automation" / "crontab").read_text(encoding="utf-8")
    assert "flywheel" in crontab and "flywheel-tick" in crontab
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_growth_flywheel.py -k crontab_uses_flywheel -v`
Expected: FAIL — assertion (crontab still calls `discussion_post.py`).

- [ ] **Step 3: Seed the committed config**

Create `medal_ops/growth/flywheel_config.json` (start conservative; flip `enabled` to `true` only when ready to go live):

```json
{
  "enabled": false,
  "max_posts_per_day": 2,
  "max_posts_per_week": 8,
  "max_forum_drops_per_comp_per_week": 1,
  "window_start_hour": 13,
  "window_end_hour": 23
}
```

- [ ] **Step 4: Swap the cron line**

In `pi-automation/crontab`, replace the discussion-poster line:

```cron
# Discussion poster -- Tuesday and Friday 10:00 UTC
0 10 * * 2,5 python3 /scripts/discussion_post.py >> /var/log/cron.log 2>&1
```

with a growth-flywheel tick on the same cadence (the flywheel internally enforces the kill switch + caps, so a denser cron is safe — it self-throttles):

```cron
# Growth flywheel tick -- Tue/Thu/Sat 14:00 UTC (kill switch + caps enforced in-engine)
0 14 * * 2,4,6 cd /repo && ./manage.sh flywheel-tick >> /var/log/cron.log 2>&1
```

> Adjust `/repo` to the container's repo mount path used by the other `pi-automation` scripts. Confirm the mount by checking how `sync.sh`/`weekly_report.sh` locate the repo, and match that convention.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_growth_flywheel.py -k crontab_uses_flywheel -v`
Expected: PASS.

- [ ] **Step 6: Run the full repo suite (no regressions)**

Run: `python -m pytest -q`
Expected: PASS (existing suite + new `test_growth_flywheel.py`).

- [ ] **Step 7: Commit**

```bash
git add pi-automation/crontab medal_ops/growth/flywheel_config.json tests/test_growth_flywheel.py
git commit -m "feat(growth): cut cron over to flywheel-tick (kill switch defaults off)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| `state.py` (GrowthState) | Task 2 |
| `scorer.py` (Reach Score, near-threshold, expected_lift) | Task 1 |
| `actions.py` (catalog + generators; no inauthentic kinds) | Task 3 |
| `safety.py` (caps, window, dedupe, kill switch) | Task 4 |
| `feedback.py` (attribution + EMA reweight) | Task 5 |
| `flywheel.py` (conductor + CLI + injectable executor) | Task 6 |
| State stores under `medal_ops/growth/` (history/weights/config/followers) | Tasks 2, 5, 6, 8 |
| CLI surface (`flywheel-status`, `flywheel-tick [--dry-run]`) | Tasks 6, 7 |
| Error handling (failed ≠ cap consumed; stop on failure) | Tasks 4, 6 |
| Testing (offline, monkeypatched) | every task |
| Rollout phases (dry-run → discussion-only → full + cron swap) | Task 8 (gated) + flywheel `--dry-run` |
| Account-health guard | Task 6 (`break` on first failed dispatch) |

Account-health *Telegram alert* and per-item *dataset* votes are intentionally deferred (noted in spec risks / Task 3 YAGNI note) — the engine degrades safely without them (it under-posts and uses notebook votes, the dominant near-threshold pool). No other spec requirement is unimplemented.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code and test step contains complete content. The single forward-reference (`campaign_execute.post_action_via_playwright`) is explicitly specified in Task 6's implementer note with how to build it.

**3. Type consistency:** `FlywheelConfig`, `Action`, `GrowthState`/`ItemState`, `DispatchResult` field names are identical across every task that references them. History-row keys (`tick_ts`, `kind`, `target_id`, `competition`, `status`) match between `safety.recent_counts`/`gate` (Task 4), `feedback.attribute` (Task 5), and `flywheel.tick`'s `append_history` (Task 6). `scorer.expected_lift(action_kind, audience, item_votes, cfg, weight)` is called with exactly those positional args in `flywheel._ranked`.

---

## Execution Handoff

(Provided after the plan by the writing-plans skill.)
