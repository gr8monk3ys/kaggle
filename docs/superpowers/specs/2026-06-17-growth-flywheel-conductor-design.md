# Growth Flywheel ("Conductor") — Design Spec

**Date:** 2026-06-17
**Status:** Approved (design)
**Owner:** Lorenzo
**Topic:** An autonomous, self-tuning engagement engine that converts already-published Kaggle content into votes, followers, and medals.

---

## Context — why this exists

The portfolio has a **distribution problem, not a production problem.** Live tracker numbers (2026-06):

| Track | Already published | Result today | Grandmaster bar |
|---|---|---|---|
| Notebooks | ~70 | ~50–67 votes total, 3 bronze, **0 gold** (most at 0 votes) | 15 gold (≥50 votes each) |
| Datasets | 12 (1,750 downloads) | 54 votes, 1 silver, rest ~0 | 5 gold (≥50 votes) |
| Discussion | 50 drafts written | **0 posts, 0 medals** | 50 gold + 500 total |
| Competitions | 12 entered | 0 medals (Novice) | 5 gold (1 solo) |

Enormous content surplus, near-zero engagement. The user's stated goal — "Grandmaster + badges + enough exposure to the materials I have made" — collapses to one problem: **get eyes, votes, and follows onto content that already exists.** Moving that needle lifts Notebooks, Datasets, Discussion, *and* badges together (the competition track is out of scope here; it is the separate long-pole lever).

The building blocks already exist but are not wired into a loop:

- `kaggle_portfolio/ops/medal_ops.py` — `top_actions()` ranks actions, but emits **human-readable text**, not machine-scored executable actions.
- `kaggle_portfolio/campaigns/campaign_dispatcher.py` — a real queue with a `planned → in_progress → done` state machine (`pi-automation/data/promotion_campaign_queue.json`).
- `kaggle_portfolio/campaigns/campaign_execute.py`, `notebooks/notebook_promoter.py`, `ops/discussion_scheduler.py` — **executors** that post via Playwright / Kaggle CLI.
- `kaggle_portfolio/ops/metadata_tracker.py` — already computes **`vote_delta` per item between snapshots** (raw material for a feedback loop).
- `pi-automation/crontab` — already cron-posts discussions Tue/Fri 10:00 UTC, but **blindly** (fixed job, not "the highest-leverage safe action right now").

**Three things are missing, and they *are* the flywheel:** (1) a unified scorer, (2) a closed measure→attribute→reweight feedback loop, (3) autonomous cross-track action selection.

## Goals

- Add only the **missing scorer + feedback loop**; reuse every existing executor and the `vote_delta` data already captured.
- Keep all logic in the tested `kaggle_portfolio/` package (`pi-automation/` stays a thin Playwright executor — repo convention).
- Run **fully automated on cron**, but behind non-negotiable safety rails and a dry-run-first rollout.
- Optimize a single **followers-weighted composite "Reach Score."**

## Non-goals

- No competition modeling/submissions (separate lever).
- No external social syndication (X/LinkedIn) — Kaggle-only for v1.
- **No inauthentic engagement**: no auto-follow, no auto-upvote of strangers, no mass unsolicited replies. These are permanently out of the action catalog, not merely gated.

## Decisions locked in brainstorming

| Decision | Choice |
|---|---|
| Lever | Growth flywheel (distribution), not competitions |
| Surface | Kaggle-only, extend existing modules |
| Autonomy | Fully automated cron (with mandated safety rails) |
| North-star | Followers-weighted composite "Reach Score" |
| Architecture | **A — Conductor**: new scorer + feedback modules dispatching through the existing campaign queue + executors |

---

## Architecture — `kaggle_portfolio/growth/`

Six focused, independently testable modules:

| Module | Single job | Reuses |
|---|---|---|
| `state.py` | Build a read-only `GrowthState`: per-item votes, gap-to-next-threshold, follower count, discussion totals | `medal_ops` snapshot, `metadata_tracker` deltas |
| `scorer.py` | **Pure function**: `reach_score(state)` and `expected_lift(state, action)`; no side effects | — |
| `actions.py` | The ToS-safe action catalog + generators enumerating concrete candidate actions | `discussion_scheduler`, `notebook_promoter`, `campaign_pack` |
| `safety.py` | The gate: `gate(ranked, state, history, config)` applies rate caps, posting window, dedupe, kill switch; plus the account-health guard | `flywheel_history.jsonl`, `flywheel_config.json` |
| `feedback.py` | Closed loop: attribute vote/follower deltas to recent actions, update per-action-type EMA weights | `metadata_tracker` deltas, `flywheel_history.jsonl` |
| `flywheel.py` | The conductor + CLI entrypoints (`flywheel-tick`, `flywheel-status`, `flywheel-dry-run`): orchestrates state→enumerate→score→gate→dispatch→log→attribute | the campaign queue + executors |

Registered in `kaggle_portfolio/manage_commands.py` as new `Command`s via `run_module`. The blind `pi-automation` Tue/Fri `discussion_post.py` cron is replaced by `flywheel-tick`.

### Data flow (one tick)

```
cron → ./manage.sh flywheel-tick
 1. state.build()            medal_ops snapshot + metadata_tracker deltas + follower count → GrowthState
 2. actions.enumerate(state) candidates: due discussion drafts · notebook_promoter forum-matches
                             (only comps ENTERED) · notebook↔dataset cross-link gaps
 3. rank = expected_lift(state, a) × feedback.weight(a.type) × audience_factor(a)
 4. safety.gate(ranked)      drop if over daily/weekly cap · kill-switch on · duplicate (vs history)
                             · outside posting window
 5. dispatch top-K           via campaign_execute / discussion_scheduler (Playwright/CLI)
 6. history.append(...)      {tick_ts, action, target, pre_state}
 7. feedback.attribute()     Δvotes/Δfollowers since last tick → credit recent actions → update weights
```

### State stores — JSON under `medal_ops/growth/` (committable, like `medal_ops/history/`)

- `flywheel_history.jsonl` — every dispatched action + pre-state snapshot (audit + attribution source).
- `flywheel_weights.json` — per-action-type EMA effectiveness weights.
- `flywheel_config.json` — caps, posting windows, kill switch, Reach-Score weights (CLI flags override).

---

## Reach Score (north-star)

```
reach_score(state) = w_f·followers
                   + w_v·Σ_items vote_progress(item)
                   + w_d·discussion_medals

vote_progress(item) = tier_value(next_cut) · min(votes, next_cut)/next_cut
                      × NEAR_BONUS   if (next_cut − votes) ∈ {1,2,3}
```

- `next_cut` ∈ {5, 20, 50} (bronze/silver/gold); `tier_value` weights gold > silver > bronze.
- `NEAR_BONUS` (default ≈3×) implements **near-threshold-first**: an item 1–3 votes below a cut is top priority — tiny push, whole tier gained.
- `expected_lift(state, action)` = scorer's estimate of `Δreach_score` if the action lands, multiplied by `feedback.weight(type)` (historical effectiveness) and `audience_factor` (e.g. competition forum team-count, normalized).
- Default weights: `w_f` weighted high (user's stated #1), `w_v` mid, `w_d` low-but-nonzero. All in `flywheel_config.json`.

Worked intuition: a notebook at 18 votes (2 from silver) cross-linked into a 4,000-team competition forum outranks posting a cold discussion draft — exactly the ROI ordering intended.

---

## Action catalog & safety

### Allowed actions (own-content only)
1. **DiscussionPost** — post the next-due draft from the discussion queue (`discussion_scheduler` / `campaign_execute`).
2. **CompetitionForumDrop** — post a starter/EDA notebook link in the forum of a competition **you have entered**, matched by `notebook_promoter` tags/`competition_sources`.
3. **CrossLink** — add notebook↔dataset cross-references / strengthen descriptions (`campaign_pack` + metadata).

### Permanently excluded (never in the catalog)
- Auto-follow, auto-upvote of others, mass unsolicited replies on strangers' threads.

### Safety rails (non-negotiable — the system is autonomous)
- **Rate caps + jitter**: default ≤2 discussion posts/day, ≤8/week, ≤1 forum-drop per competition per week; randomized timing inside an allowed posting window.
- **Kill switch**: `enabled:false` in `flywheel_config.json` or `FLYWHEEL_DISABLED=1` halts all dispatch; the tick still logs what it *would* have done (full dry-run trace).
- **Dedupe**: never repost the same draft/target (checked against `flywheel_history.jsonl`).
- **Account-health guard**: Playwright captcha/login failure, or an anomalous vote *drop* from `metadata_tracker`, auto-pauses dispatch and alerts via Telegram (reuse `pi-automation` notify).

---

## Error handling

- Every executor call is wrapped. A failure logs `status=failed` and **does not consume a rate-cap slot**, so it is retried next tick.
- An action is marked `done` only on a **confirmed** post (idempotent against `flywheel_history.jsonl`).
- Captcha / login failure trips the account-health guard (pause + alert) instead of silently retrying.

---

## CLI surface (`manage_commands.py`)

| Command | Behavior |
|---|---|
| `./manage.sh flywheel-status` | Print the Reach Score dashboard: current score, per-track gaps, near-threshold items, action-type weights. Read-only. |
| `./manage.sh flywheel-tick --dry-run` | Run a full tick, score + rank + safety-gate, print what it *would* dispatch. No posting. |
| `./manage.sh flywheel-tick` | Live tick: dispatch top-K safe actions, log, attribute. |

---

## Testing (all offline, matches existing `tests/` convention — executors monkeypatched)

- `scorer` (pure): `reach_score`, near-threshold bonus, `expected_lift` ranking.
- `feedback`: attribution window + EMA weight update on synthetic history.
- `actions`: generators against fixture `GrowthState` (reuse `conftest` fixtures).
- `safety.gate`: caps, kill switch, dedupe, posting window.
- `flywheel` tick (integration): executors monkeypatched → asserts it picks the highest-scored **safe** action and respects caps; asserts a failed dispatch does not consume a cap slot.
- New file: `tests/test_growth_flywheel.py`.

---

## Rollout (de-risks the "fully automated" choice)

1. **Observe** — ship `flywheel-status` + `flywheel-tick --dry-run` only. Zero posting; watch the scoring for a few days.
2. **Discussion-only** — enable live dispatch for own drafts at low caps (≤2/day).
3. **Full** — add forum-drops + cross-links once attribution shows what lands; then swap the `pi-automation` cron from `discussion_post.py` to `flywheel-tick`.

---

## Risks & open notes

- **Attribution is heuristic.** Kaggle exposes no per-referrer analytics, so `feedback.py` approximates causality by time-correlating deltas to recent actions. Expect noise; the EMA smooths it. This is acceptable because the *ranking* (near-threshold-first) is robust even if attribution is weak.
- **Anti-spam is the dominant failure mode.** Caps, jitter, the allowlist, and the health guard exist specifically to protect account standing; if in doubt, the system under-posts rather than over-posts.
- **Follower attribution is the weakest signal** (lowest frequency); treat `w_f` movement as a slow trend, not a per-tick metric.
