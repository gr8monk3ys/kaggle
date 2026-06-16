# Grandmaster Roadmap (Balanced 4-Category) + Follower Growth — Design

**Date:** 2026-06-16
**Status:** Draft for review
**Relates to:** [`2026-06-14-grandmaster-program-phase-0-1-design.md`](2026-06-14-grandmaster-program-phase-0-1-design.md) (rails + measurement, done), [`docs/discussions/engagement-strategy.md`](../../discussions/engagement-strategy.md), [`docs/reports/grandmaster-tracker.md`](../../reports/grandmaster-tracker.md), [`docs/reports/competition-scout-report.md`](../../reports/competition-scout-report.md)

## Decisions captured (this session)

- **Priority:** *Balanced across all four categories* — steady parallel progress, not single-category sprint.
- **Follower growth:** *Conservative / ToS-safe* — curated, human-paced follows + genuine engagement; no bulk automation.
- **Session scope:** *Plan + safe setup* — write this roadmap and do reversible prep only; **no live posting/following** without explicit go-ahead.

## Reality check (important)

1. **Grandmaster tiers are medal-based, not follower-based.** None of the four GM tiers count followers. Followers amplify *visibility*, which indirectly lifts notebook/dataset/discussion votes — a second-order lever, not a tier requirement.
2. **Kaggle has no official "follow" API.** Following is browser-automation only (`pi-automation/scripts/follow_users.py`). Bulk automated following is a **ToS gray area** that can flag/suspend an account. "Unlimited follows without breaking guidelines" is not accurate — *pace and genuineness* are the guardrails.
3. **Two known prerequisites gate live cadence** (both outside this repo's code):
   - **GitHub Actions billing is failing account-wide** → scheduled telemetry/health/posting will not run until resolved in *Settings → Billing & plans*.
   - **No always-on host** for browser posting/following (the `pi-automation` stack is not deployed). Live discussion posting and following must currently be **run manually** (or a host stood up — deferred Phase-2 decision).

## Current standing (from the tracker, 2026-06-11 snapshot)

| Category | Tier | Medals (B/S/G) | Next tier | Gap to next tier |
|----------|------|----------------|-----------|------------------|
| Competitions | Novice | 0 / 0 / 0 (12 entered) | Expert | 2 bronze (top 40%, 100+ teams) |
| Notebooks | Novice | 3 / 0 / 0 (~50 votes, 59 nb) | Expert | 2 more bronze (need 5 total) |
| Datasets | Novice | 1 / 1 / 0 (54 votes, 1750 dl) | Expert | 1 more bronze (need 3 total) |
| Discussion | Novice | 0 / 0 / 0 (**0 posted**, 50 drafts ready) | Expert | 50 bronze (≥1 net vote each) |

**Read:** the cheapest tier-ups are **Datasets** (1 medal away from Expert) and **Notebooks** (2 away). **Discussion** has the most upside per unit effort because 50 drafts are written but **none are posted** — the entire pipeline is built and idle. **Competitions** is the longest road (GM = 5 gold incl. 1 solo) and is treated as a slow continuous grind, not a near-term tier-up.

## Strategy: balanced, with effort weighted by ROI

"Balanced" does not mean equal hours — it means *no category goes dark*. Weekly effort splits roughly:

- **Discussion (≈40%)** — highest ROI right now. Convert the idle 50-draft pipeline into posted bronze medals at a **conservative 2–3 posts/week** (matches `engagement-strategy.md`). Each genuine, useful post needs only ≥1 net vote for bronze.
- **Notebooks (≈25%)** — ship/refresh 1 high-quality notebook per 1–2 weeks aimed at the active boards in the scout report; cross-link from discussion posts to gather the 2 bronze needed for Expert, then push toward silver (20 votes).
- **Datasets (≈15%)** — one focused push: 1 dataset over the bronze line (≥5 votes) → Datasets Expert. Convert the existing 1,750 downloads into votes via better usability + a companion EDA notebook + a discussion post.
- **Competitions (≈20%)** — pick **one** active board from the scout report and submit weekly. Target a *single bronze* first (top 40% on a 100+ team board) to break out of Novice; GM is a multi-season effort.
- **Followers (woven in, ~paced)** — see below; a low-effort, low-risk visibility multiplier, not a category.

### Per-category concrete next actions

**Discussion → Expert (50 bronze).** Pipeline already exists (`discussion_scheduler`, 50 drafts, queue of 6). Next:
- Post via `./manage.sh next-post` (surfaces the next ready draft to post **manually**) at 2–3/week.
- Keep the queue fed: `./manage.sh draft-set <id> --status ready` to promote drafts; `./manage.sh post-discussion --schedule-weeks N` to rebuild the window.
- This is **Phase 2 (engagement)** of the existing program — this doc is its first concrete plan. The remaining work (always-on host vs. manual cadence) is the open Phase-2 decision.

**Datasets → Expert (3 bronze; currently 2 medaled).** One more dataset over ≥5 votes:
- `./manage.sh quality`/usability already keeps datasets ≥85; pick the highest-download, lowest-vote dataset and add a companion EDA notebook + a "dataset spotlight" discussion post to drive votes.

**Notebooks → Expert (5 bronze; currently 3).** Two more bronze:
- The 5 notebooks rebuilt this session (now 100/100 quality) are publishable assets. Publish/refresh on Kaggle (`./manage.sh push <dir>`), then cross-promote each from a discussion post.

**Competitions → Expert (2 bronze).** From the scout report, the workable boards with healthy team counts are e.g. `playground-series-s6e6` (forgiving, tabular) and `hull-tactical-market-prediction`. Pick one, submit a baseline weekly, iterate. The `local_competition_lab` benchmarks (Phase 3) support this.

## Follower growth — conservative / ToS-safe

**Principle:** followers follow *value and presence*, not follow-spam. Order of leverage:

1. **Content + engagement (primary).** Every posted notebook/dataset/discussion is a discovery surface. Reply substantively in threads on boards you compete in. This is the durable, zero-risk growth engine and it doubles as medal progress.
2. **Targeted reciprocal following (secondary, paced).** Follow *peers* — people who recently engaged with your content or compete on the same boards — who are plausibly reciprocal. Not Grandmasters (won't follow back), not random bulk.
3. **Never** bulk-automate follows. It risks the account that the entire medal effort depends on.

**Guardrails (encoded in `follow_targets.json`):**
- ≤ **10 follows per session** (the `follow_users.py --limit` default), ≤ **2–3 sessions/week**.
- Human-paced (the script already inserts per-action delays); run **manually / headed**, never on a tight loop.
- Curate `users` by the selection criteria below; verify each is a real, active, relevant profile before adding.

**How to populate targets safely (no scraping):**
- Co-competitors visible on the public leaderboards of boards you've entered.
- Authors of notebooks/datasets you genuinely upvoted.
- Active participants in discussion threads you join.

This session sets up the *structure and guardrails* of `follow_targets.json` but intentionally leaves `users` empty — seeding it with unverified handles would be noise. The user (or a future go-ahead) curates real peers per the criteria.

## Weekly cadence (steady-state)

| Day | Action | Tool |
|-----|--------|------|
| Mon | Post discussion draft #1; reply in 2 threads | `manage.sh next-post` |
| Tue | Competition submission/iteration | `local_competition_lab` |
| Wed | Post discussion draft #2; 1 follow session (≤10, manual) | `next-post`, `follow-users` |
| Thu | Notebook ship/refresh + cross-post | `manage.sh push` |
| Fri | Post discussion draft #3; dataset vote-driving | `next-post` |
| (daily) | Telemetry digest review (deltas, pace, nearest deadline) | telemetry workflow → Telegram |

## Measurement (reuse existing loop)

No new measurement infra. The Phase-1 telemetry workflow (`medal_ops sync` → tracker + `medal_ops/history/snapshot-*.json` → `pace`/`digest`) is the source of truth once billing is restored. Until then, run `./manage.sh sync` + `./manage.sh scorecard` manually to refresh counts. Success = monotonically rising medal counts and a non-`n/a` `pace` velocity.

## Dependencies / risks

- **Actions billing** (account-level) — blocks scheduled telemetry, health, and any future scheduled posting. *User action required.*
- **Always-on host** — blocks unattended discussion posting/following; until decided, cadence is manual. (Open Phase-2 decision.)
- **ToS** — automated following at volume risks the account; the conservative guardrails above are deliberate.
- **Auto-committed tracker** — the telemetry loop auto-updates the tracker; bad data would auto-commit. The Phase-0 zero-vote/guard fixes mitigate this.

## Safe setup performed this session (reversible, local)

1. `pi-automation/data/follow_targets.json` — added `selection_criteria`, `guardrails`, and `candidate_sources`; `users` left empty by design (await curated, verified peers).
2. Verified the discussion `next-post` safe-assist surfaces a ready draft for **manual** posting.

No live posts, follows, or Kaggle mutations were performed.

## Not done (needs your go-ahead)

- Posting any discussion drafts live.
- Following any users live.
- Entering/submitting to competitions.
- Resolving the Actions billing block (account-only).
