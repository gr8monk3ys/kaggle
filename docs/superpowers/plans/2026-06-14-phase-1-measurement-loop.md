# Phase 1 — Measurement Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Grandmaster progress measurable by persisting a durable daily snapshot, a one-message digest, and a GitHub Actions cadence that commits telemetry into the repo — so `pace`/ETA stop reporting "1 snapshot".

**Architecture:** Un-gitignore `medal_ops/history/` so snapshots survive in git; add a pure `generate_digest()` + a `medal_ops digest` subcommand that prints a Markdown digest to stdout (keeping `medal_ops` decoupled from Telegram); wire `digest` into `manage.sh`; add a `telemetry.yml` workflow that runs a real (non-`--dry-run`) live `sync`, composes the digest, sends it via `notify.py`, and commits the refreshed tracker + new snapshot.

**Tech Stack:** Python 3.14 / pytest 9, the repo `.venv`, GitHub Actions, conventional commits.

**Reference spec:** [`docs/superpowers/specs/2026-06-14-grandmaster-program-phase-0-1-design.md`](../specs/2026-06-14-grandmaster-program-phase-0-1-design.md)

**Test runner:** `.venv/bin/python -m pytest`.

**Integration facts (verified against the code — use these exactly):**
- Snapshot dict keys: `generated_on`, `tracker_last_updated`, `tracker_stale_days`, `categories.{competitions,notebooks,datasets,discussion}` (each with `gold/silver/bronze/tier` plus `entered` (competitions), `total_votes` (notebooks/datasets), `total_posts` (discussion), and `*_goal`/`*_gap`), and `active_competitions[]` (each with `competition`, `days_to_deadline: int|None`, `deadline_date`, `teams`, ...).
- Reuse existing module-level helpers in `medal_ops.py`: `delta(current, previous, path)` → `int|None`; `top_actions(snapshot)` → `list[str]`; `load_all_snapshots(history_dir)` → chronological list.
- Draft-queue health: `build_ops_summary(load_queue())` from `discussion_scheduler` → dict with `ready_now`, `days_until_next_post`, `overdue_scheduled`.
- Subcommands register via `subparsers.add_parser(...)` + `add_shared_cli_args(parser, is_subparser=True)`; dispatch in `main()` after `snapshot = build_snapshot(content, today)`, where `history_dir`, `output_root`, `tracker_path` are in scope.
- `notify.py` is NOT package-importable; call it as `python pi-automation/scripts/notify.py "msg"`.

---

### Task 1: Persist snapshots in git (un-gitignore `medal_ops/history/`)

Snapshots currently land in a gitignored dir, so they never accumulate. Un-ignore `history/` (keep `reports/` and `sync_inputs/` ignored).

**Files:**
- Modify: `.gitignore`
- Test: `tests/test_repo_guardrails.py` (append one function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_repo_guardrails.py`:

```python
def test_medal_ops_history_is_tracked_not_ignored():
    """Daily snapshots must be committable so pace history accumulates."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "medal_ops/history/snapshot-sample.json"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    # returncode 1 == NOT ignored (what we want); 0 == ignored (fail)
    assert result.returncode == 1, "medal_ops/history/ must NOT be gitignored"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_repo_guardrails.py::test_medal_ops_history_is_tracked_not_ignored -v`
Expected: FAIL (returncode 0 — `medal_ops/history/` is currently ignored).

- [ ] **Step 3: Remove the history ignore rule**

In `.gitignore`, the `# Medal ops generated artifacts` block currently reads:

```gitignore
# Medal ops generated artifacts
medal_ops/history/
medal_ops/reports/
medal_ops/sync_inputs/
.competition_lab/
```

Delete the `medal_ops/history/` line so it reads:

```gitignore
# Medal ops generated artifacts (history is committed for pace tracking; reports/sync_inputs stay local)
medal_ops/reports/
medal_ops/sync_inputs/
.competition_lab/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_repo_guardrails.py::test_medal_ops_history_is_tracked_not_ignored -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/test_repo_guardrails.py
git commit -m "feat: track medal_ops/history so daily snapshots accumulate for pace"
```

---

### Task 2: Add `generate_digest()` + `medal_ops digest` subcommand

A pure function composes a one-message Markdown digest from snapshot history + draft-queue health; the `digest` subcommand prints it to stdout.

**Files:**
- Modify: `kaggle_portfolio/ops/medal_ops.py`
- Test: `tests/test_medal_ops.py` (append a `TestDigest` class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_medal_ops.py` (the module imports `medal_ops` — match however it is aliased at the top of that file; below uses `medal_ops`):

```python
class TestDigest:
    @staticmethod
    def _snap(generated_on, *, entered, nb_votes, ds_votes, posts, stale_days=0, comps=None):
        return {
            "generated_on": generated_on,
            "tracker_last_updated": generated_on,
            "tracker_stale_days": stale_days,
            "categories": {
                "competitions": {"gold": 0, "silver": 0, "bronze": 0, "entered": entered,
                                  "tier": "Novice", "gold_goal": 5, "gold_gap": 5,
                                  "expert_bronze_goal": 1, "expert_bronze_gap": 1},
                "notebooks": {"gold": 0, "silver": 0, "bronze": 0, "total_notebooks": 10,
                              "total_votes": nb_votes, "tier": "Novice", "gold_goal": 15,
                              "gold_gap": 15, "expert_bronze_goal": 1, "expert_bronze_gap": 1},
                "datasets": {"gold": 0, "silver": 0, "bronze": 0, "total_datasets": 5,
                             "total_votes": ds_votes, "tier": "Novice", "gold_goal": 5,
                             "gold_gap": 5, "expert_bronze_goal": 1, "expert_bronze_gap": 1},
                "discussion": {"gold": 0, "silver": 0, "bronze": 0, "total_posts": posts,
                               "tier": "Novice", "gold_goal": 50, "gold_gap": 50,
                               "total_goal": 500, "total_gap": 500,
                               "expert_bronze_goal": 50, "expert_bronze_gap": 50},
            },
            "active_competitions": comps or [],
        }

    def test_digest_with_two_snapshots_shows_deltas_deadline_and_action(self):
        comps = [
            {"competition": "Orbit Wars", "days_to_deadline": 12, "deadline_date": "2026-06-26"},
            {"competition": "Hull Tactical", "days_to_deadline": 3, "deadline_date": "2026-06-17"},
        ]
        s1 = self._snap("2026-06-13", entered=10, nb_votes=60, ds_votes=54, posts=0)
        s2 = self._snap("2026-06-14", entered=11, nb_votes=68, ds_votes=54, posts=2, comps=comps)
        health = {"ready_now": 2, "days_until_next_post": 4, "overdue_scheduled": 0}

        out = medal_ops.generate_digest([s1, s2], health)

        assert "2026-06-14" in out
        assert "+8" in out                 # notebook votes 60 -> 68
        assert "Hull Tactical" in out and "3" in out   # nearest deadline (not Orbit Wars at 12)
        assert "Orbit Wars" not in out
        assert "ready" in out.lower()
        # top action is non-empty (discussion has a bronze gap)
        assert "Top action" in out

    def test_digest_first_snapshot_has_no_deltas(self):
        s1 = self._snap("2026-06-14", entered=10, nb_votes=60, ds_votes=54, posts=0)
        out = medal_ops.generate_digest([s1], {})
        assert "First snapshot" in out

    def test_digest_no_snapshots(self):
        assert "No snapshots" in medal_ops.generate_digest([], {})

    def test_digest_tolerates_missing_queue_health(self):
        s1 = self._snap("2026-06-13", entered=10, nb_votes=60, ds_votes=54, posts=0)
        s2 = self._snap("2026-06-14", entered=10, nb_votes=60, ds_votes=54, posts=0)
        out = medal_ops.generate_digest([s1, s2], {})   # empty health must not crash
        assert "2026-06-14" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_medal_ops.py::TestDigest -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'generate_digest'`.

- [ ] **Step 3: Implement `generate_digest`, `_load_queue_health`, and the subcommand**

In `kaggle_portfolio/ops/medal_ops.py`, add these two functions immediately AFTER `generate_pace_markdown` (they rely on the existing module-level `delta` and `top_actions`):

```python
def _fmt_delta(value: int | None) -> str:
    if value is None:
        return "—"
    return f"+{value}" if value > 0 else str(value)


def generate_digest(snapshots: list[dict[str, Any]], queue_health: dict[str, Any]) -> str:
    """Compose a one-message daily Grandmaster digest (Markdown) from snapshot history.

    Pure function: takes the chronological snapshot list and a draft-queue health
    dict (may be empty), returns a Telegram-ready Markdown string.
    """
    if not snapshots:
        return "No snapshots available yet — run `medal_ops sync` first."

    current = snapshots[-1]
    previous = snapshots[-2] if len(snapshots) >= 2 else None
    lines = [f"\U0001F4CA *Grandmaster digest — {current.get('generated_on', '?')}*", ""]

    if previous is None:
        lines.append("_First snapshot recorded — deltas start next run._")
    else:
        movers = [
            ("Comp entered", ("categories", "competitions", "entered")),
            ("Notebook votes", ("categories", "notebooks", "total_votes")),
            ("Dataset votes", ("categories", "datasets", "total_votes")),
            ("Discussion posts", ("categories", "discussion", "total_posts")),
        ]
        bits = []
        for label, path in movers:
            d = delta(current, previous, path)
            if d:
                bits.append(f"{label} {_fmt_delta(d)}")
        lines.append("*Since last snapshot:* " + (", ".join(bits) if bits else "no change"))

    upcoming = [
        c for c in current.get("active_competitions", [])
        if isinstance(c.get("days_to_deadline"), int) and c["days_to_deadline"] >= 0
    ]
    if upcoming:
        nearest = min(upcoming, key=lambda c: c["days_to_deadline"])
        lines.append(f"*Nearest deadline:* {nearest.get('competition', '?')} "
                     f"in {nearest['days_to_deadline']}d")
    else:
        lines.append("*Nearest deadline:* none tracked")

    if queue_health:
        nd = queue_health.get("days_until_next_post")
        nd_str = f"{nd}d" if isinstance(nd, int) else "n/a"
        lines.append(
            f"*Draft queue:* {queue_health.get('ready_now', 0)} ready, "
            f"next post in {nd_str}, {queue_health.get('overdue_scheduled', 0)} overdue"
        )

    actions = top_actions(current)
    if actions:
        lines.append(f"*Top action today:* {actions[0]}")

    return "\n".join(lines)


def _load_queue_health() -> dict[str, Any]:
    """Best-effort draft-queue health for the digest; empty dict if unavailable."""
    try:
        from kaggle_portfolio.ops.discussion_scheduler import build_ops_summary, load_queue
        return build_ops_summary(load_queue())
    except Exception:
        return {}
```

Register the subcommand. Find the block where other subparsers are created (near `scorecard_parser = subparsers.add_parser(...)`, ~line 1568) and add:

```python
    digest_parser = subparsers.add_parser("digest", help="Print a one-message daily Grandmaster digest to stdout.")
    add_shared_cli_args(digest_parser, is_subparser=True)
```

Dispatch it in `main()`. After the line `snapshot = build_snapshot(content, today)` and alongside the other `if args.command == "..."` branches, add:

```python
    if args.command == "digest":
        snapshots = load_all_snapshots(history_dir)
        if not snapshots:
            snapshots = [snapshot]
        print(generate_digest(snapshots, _load_queue_health()))
        return 0
```

(`Any` is already imported at the top of `medal_ops.py` via `from typing import Any`; confirm and use it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_medal_ops.py -q`
Expected: PASS (the new `TestDigest` class plus the existing suite).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/ops/medal_ops.py tests/test_medal_ops.py
git commit -m "feat: add medal_ops digest subcommand and generate_digest composer"
```

---

### Task 3: Wire `digest` into `manage.sh`

Expose `./manage.sh digest` so the command is discoverable alongside `scorecard`, `weekly-plan`, etc.

**Files:**
- Modify: `kaggle_portfolio/manage_commands.py`
- Test: `tests/test_manage_commands.py` (append one function)

- [ ] **Step 1: Read the existing pattern, then write the failing test**

First READ `kaggle_portfolio/manage_commands.py` to see the exact `Command(...)` constructor signature and how the existing medal_ops subcommands (`scorecard`, `weekly-plan`, `pace`, `badge-plan`) are registered in the command table. Match that pattern exactly.

Append to `tests/test_manage_commands.py` (match the module's existing import of the commands table / dispatch — adapt the accessor name to whatever the file already uses to look up a command):

```python
def test_digest_command_is_registered_and_targets_medal_ops():
    from kaggle_portfolio import manage_commands
    names = manage_commands.command_names() if hasattr(manage_commands, "command_names") \
        else [c.name for c in manage_commands.COMMANDS]
    assert "digest" in names
```

If `test_manage_commands.py` already has a helper that lists or looks up commands, use that instead of the inline fallback above — keep it consistent with the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manage_commands.py::test_digest_command_is_registered_and_targets_medal_ops -v`
Expected: FAIL — `"digest"` is not a registered command.

- [ ] **Step 3: Register the command**

In `kaggle_portfolio/manage_commands.py`, add a `digest` entry to the command table next to the other `medal_ops` commands, following the EXACT same `Command(...)` form already used for `scorecard`/`weekly-plan` (same dispatch helper, same metadata fields). It must invoke the `medal_ops` module with the `digest` subcommand, e.g. (adapt to the real constructor/dispatch helper in the file):

```python
    Command(
        "digest",
        "Print a one-message daily Grandmaster digest",
        lambda args: run_module("kaggle_portfolio.ops.medal_ops", ["digest", *args]),
    ),
```

Use whatever `run_module`/dispatch helper and `Command` field order the surrounding entries use — do not invent a new shape.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manage_commands.py -q`
Expected: PASS (new test + existing suite).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/manage_commands.py tests/test_manage_commands.py
git commit -m "feat: expose digest via manage.sh"
```

---

### Task 4: Add the daily telemetry GitHub Actions workflow

A new, separate workflow runs a real live `sync` (writes tracker + snapshot), composes + sends the digest, and commits the telemetry back to the repo.

**Files:**
- Create: `.github/workflows/telemetry.yml`
- Test: `tests/test_repo_guardrails.py` (append one function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_repo_guardrails.py` (the file already imports `yaml`/reads workflows for other guardrail tests — match its existing YAML-loading helper if present; otherwise use the inline load below):

```python
def test_telemetry_workflow_records_and_commits_snapshots():
    import yaml
    wf_path = ROOT / ".github" / "workflows" / "telemetry.yml"
    assert wf_path.exists(), "telemetry.yml must exist"
    wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))

    # scheduled + manually dispatchable
    on = wf.get(True, wf.get("on"))   # PyYAML may parse bare `on:` as boolean True
    assert "schedule" in on
    assert "workflow_dispatch" in on

    # needs write access to commit telemetry
    assert wf.get("permissions", {}).get("contents") == "write"

    body = wf_path.read_text(encoding="utf-8")
    # runs a REAL sync (not dry-run) and composes the digest
    assert "medal_ops sync" in body
    assert "--dry-run" not in body
    assert "medal_ops digest" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_repo_guardrails.py::test_telemetry_workflow_records_and_commits_snapshots -v`
Expected: FAIL (`telemetry.yml` does not exist).

- [ ] **Step 3: Create the workflow**

Create `.github/workflows/telemetry.yml`. IMPORTANT: for `actions/checkout` and `actions/setup-python`, use the EXACT pinned commit SHAs + version comments already used in `.github/workflows/medal-ops-health.yml` (read that file and copy the refs verbatim so this workflow matches the repo's SHA-pinning baseline). Use that file's "Install Kaggle CLI" and "Configure Kaggle credentials" steps as the model for the two credential steps below.

```yaml
name: Telemetry Snapshot

on:
  schedule:
    - cron: "30 9 * * *"   # daily, offset from medal-ops-health (10 9)
  workflow_dispatch:

permissions:
  contents: write

jobs:
  snapshot:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@<PINNED_SHA_FROM_MEDAL_OPS_HEALTH>   # vX.Y.Z

      - name: Setup Python
        uses: actions/setup-python@<PINNED_SHA_FROM_MEDAL_OPS_HEALTH>   # vX.Y.Z
        with:
          python-version: "3.11"

      - name: Install Kaggle CLI + deps (live mode only)
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          set -euo pipefail
          if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
            echo "No Kaggle credentials configured; telemetry needs live data. Exiting cleanly."
            echo "SKIP=1" >> "$GITHUB_ENV"
            exit 0
          fi
          python -m pip install --upgrade pip
          python -m pip install kaggle requests

      - name: Configure Kaggle credentials (live mode only)
        if: env.SKIP != '1'
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          set -euo pipefail
          mkdir -p ~/.kaggle
          printf '{"username":"%s","key":"%s"}' "${KAGGLE_USERNAME}" "${KAGGLE_KEY}" > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      - name: Sync tracker + record snapshot
        if: env.SKIP != '1'
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          set -euo pipefail
          python -m kaggle_portfolio.ops.medal_ops sync
          python -m kaggle_portfolio.notebooks.competition_scout --update || echo "scout refresh skipped"

      - name: Compose + send digest
        if: env.SKIP != '1'
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          set -euo pipefail
          MSG="$(python -m kaggle_portfolio.ops.medal_ops digest || echo 'digest generation failed')"
          echo "$MSG"
          if [[ -n "${TELEGRAM_BOT_TOKEN:-}" ]]; then
            python pi-automation/scripts/notify.py "$MSG" || echo "notify failed (non-fatal)"
          else
            echo "No Telegram token; digest printed to log only."
          fi

      - name: Commit telemetry
        if: env.SKIP != '1'
        run: |
          set -euo pipefail
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add docs/reports/grandmaster-tracker.md medal_ops/history/ || true
          git add docs/reports/competition-scout-report.md 2>/dev/null || true
          if git diff --cached --quiet; then
            echo "No telemetry changes to commit."
          else
            git commit -m "chore(telemetry): daily snapshot $(date -u +%Y-%m-%d) [skip ci]"
            git push
          fi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_repo_guardrails.py::test_telemetry_workflow_records_and_commits_snapshots -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/telemetry.yml tests/test_repo_guardrails.py
git commit -m "feat: add daily telemetry workflow (sync + snapshot + digest + commit)"
```

---

### Task 5: Full-suite verification

- [ ] **Step 1: Run both suites**

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest pi-automation/tests -q
```
Expected: both green; `tests/` gains the digest + guardrail tests, 0 failures.

- [ ] **Step 2: Smoke-test the digest CLI against current history**

```bash
.venv/bin/python -m kaggle_portfolio.ops.medal_ops digest
```
Expected: prints a Markdown digest (either real deltas, "First snapshot…", or "No snapshots available yet" depending on local history) without error.

- [ ] **Step 3: Confirm branch diff is coherent**

```bash
git diff --stat main..HEAD
```
Expected: exactly `.gitignore`, `kaggle_portfolio/ops/medal_ops.py`, `kaggle_portfolio/manage_commands.py`, `.github/workflows/telemetry.yml`, the matching test files, and this plan doc. No unrelated changes.

---

## Self-Review

**Spec coverage:** durable snapshots → Task 1; digest composer + subcommand → Task 2; manage.sh exposure → Task 3; GitHub Actions cadence that syncs/commits/digests → Task 4. Decisions A (un-gitignore history) and the digest-to-stdout decoupling are implemented. Decision B (auto-commit tracker) is realized by the telemetry commit step.

**Placeholder scan:** the only deliberate read-from-repo instructions are the `Command(...)` shape in Task 3 and the pinned action SHAs in Task 4 — both are concrete values that live in the repo and must be copied verbatim, not invented. All code steps contain complete code.

**Type/name consistency:** `generate_digest(snapshots, queue_health)` and `_load_queue_health()` defined in Task 2 are used by the `digest` dispatch in the same task; `delta`/`top_actions`/`load_all_snapshots` are existing module functions; the workflow in Task 4 calls the `digest`/`sync` subcommands defined/confirmed in Tasks 2 and the existing code.

**Known external gate:** landing `telemetry.yml` on origin requires the pushing token to have the `workflow` scope (`gh auth refresh -h github.com -s workflow`) — same constraint as PRs #32/#33. Building/committing locally is unaffected; only the push of the workflow file is gated.
