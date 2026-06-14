# Phase 0 — Safety Rails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the five review fixes that gate the rest of the automation (credential leak, token leak, draft misrouting, phantom-zero vote corruption, unbounded auth probes) before any cadence is switched on.

**Architecture:** Five independent, test-first fixes across `.gitignore`, `pi-automation/scripts/notify.py`, and three `kaggle_portfolio/ops/` modules. Each fix is its own task: failing test → minimal change → green → commit. No fix depends on another, so tasks may be done in any order.

**Tech Stack:** Python 3.14 / pytest 9, the repo `.venv`, conventional-commit messages (release-please).

**Reference spec:** [`docs/superpowers/specs/2026-06-14-grandmaster-program-phase-0-1-design.md`](../specs/2026-06-14-grandmaster-program-phase-0-1-design.md)

**Test runner:** all commands use the repo venv: `.venv/bin/python -m pytest`.

---

### Task 1: Gitignore the Kaggle session-cookie file (security)

The Playwright login persists `kaggle_storage_state.json` (live session cookies) into the git-tracked `pi-automation/data/` directory. Add an ignore rule so it can never be staged.

**Files:**
- Modify: `.gitignore`
- Test: `tests/test_repo_guardrails.py` (append one function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_repo_guardrails.py`:

```python
def test_kaggle_session_cookie_is_gitignored():
    """The Playwright session-cookie file must never be committable."""
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "-q", "pi-automation/data/kaggle_storage_state.json"],
        cwd=repo_root,
    )
    assert result.returncode == 0, "kaggle_storage_state.json must be gitignored"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_repo_guardrails.py::test_kaggle_session_cookie_is_gitignored -v`
Expected: FAIL (returncode 1 — the path is not currently ignored).

- [ ] **Step 3: Add the ignore rule**

In `.gitignore`, under the `# Secrets` block, add the `kaggle_storage_state.json` line so it reads:

```gitignore
# Secrets
kaggle.json
**/kaggle.json
**/kaggle_storage_state.json
.kaggle/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_repo_guardrails.py::test_kaggle_session_cookie_is_gitignored -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/test_repo_guardrails.py
git commit -m "security: gitignore Kaggle session-cookie storage_state file"
```

---

### Task 2: Redact the Telegram bot token from notify.py error logs (security)

`notify.send()` builds the bot URL with the token embedded, and a `requests.RequestException` string contains that URL. Cron redirects stderr into a persisted log volume, so a transient network error writes the token in cleartext. Redact it.

**Files:**
- Modify: `pi-automation/scripts/notify.py`
- Test: `pi-automation/tests/test_notify.py` (append one function)

- [ ] **Step 1: Write the failing test**

Append to `pi-automation/tests/test_notify.py`:

```python
def test_send_redacts_token_in_request_exception(monkeypatch, capsys):
    """A request failure must not leak the bot token into stderr."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

    exc = requests.RequestException(
        "HTTPSConnectionPool: url=https://api.telegram.org/botsecret-token-123/sendMessage failed"
    )
    with patch("notify.requests.post", side_effect=exc):
        import importlib
        import notify
        importlib.reload(notify)
        notify.send("leaky")

    captured = capsys.readouterr()
    assert "secret-token-123" not in captured.err
    assert "***" in captured.err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest pi-automation/tests/test_notify.py::test_send_redacts_token_in_request_exception -v`
Expected: FAIL — the token appears in stderr (assert `"secret-token-123" not in captured.err` fails).

- [ ] **Step 3: Implement redaction**

In `pi-automation/scripts/notify.py`, add a redaction helper above `send()` and use it in both error branches. The function body becomes:

```python
def _redact(text: str, token: str) -> str:
    """Replace the bot token with '***' so it never reaches logs."""
    return text.replace(token, "***") if token else text


def send(message: str) -> None:
    """Send a Telegram message via Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    Raises EnvironmentError if token is missing.
    Prints to stderr on API failure (does not raise).
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN environment variable is not set")

    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"Telegram API request failed: {_redact(str(exc), token)}", file=sys.stderr)
        return

    if response.status_code != 200:
        print(
            f"Telegram API error {response.status_code}: {_redact(response.text, token)}",
            file=sys.stderr,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest pi-automation/tests/test_notify.py -v`
Expected: PASS (the new test plus the four existing `notify` tests).

- [ ] **Step 5: Commit**

```bash
git add pi-automation/scripts/notify.py pi-automation/tests/test_notify.py
git commit -m "security: redact Telegram bot token from notify.py error logs"
```

---

### Task 3: Resolve discussion forums longest-key-first (fix NLP misroute)

`parse_drafts` resolves the target forum with a first-match substring scan over `FORUM_MAP`. Because `"getting started"` is inserted before and is a substring of `"nlp getting started"`, an `NLP Getting Started` draft routes to the generic board and `infer_priority` then drops it from high to medium. Match the longest key first.

**Files:**
- Modify: `kaggle_portfolio/ops/discussion_scheduler.py:127-146` (extract a `resolve_forum` helper; call it from `parse_drafts`)
- Test: `tests/test_discussion_scheduler.py` (append two functions)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_discussion_scheduler.py`:

```python
def test_resolve_forum_prefers_longest_matching_key():
    resolve = discussion_scheduler.resolve_forum
    assert resolve("nlp getting started") == \
        "https://www.kaggle.com/competitions/nlp-getting-started/discussion"
    assert resolve("getting started") == \
        "https://www.kaggle.com/discussions/getting-started"
    assert resolve("deep past akkadian") == \
        "https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion"
    assert resolve("something unmapped") == discussion_scheduler.DEFAULT_FORUM


def test_parse_drafts_routes_nlp_getting_started_to_competition(tmp_path):
    drafts_path = tmp_path / "discussion-drafts.md"
    drafts_path.write_text(
        "\n".join(
            [
                "## Draft 1: NLP Tips",
                "**Target forum:** NLP Getting Started",
                "",
                "### NLP Tips",
                "",
                "Body.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    drafts = discussion_scheduler.parse_drafts(drafts_path)

    assert drafts[0]["forum_url"] == \
        "https://www.kaggle.com/competitions/nlp-getting-started/discussion"
    assert drafts[0]["priority"] == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_discussion_scheduler.py::test_resolve_forum_prefers_longest_matching_key -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'resolve_forum'`.

- [ ] **Step 3: Add `resolve_forum` and call it from `parse_drafts`**

In `kaggle_portfolio/ops/discussion_scheduler.py`, add this helper immediately above `def parse_drafts` (after `infer_priority`):

```python
def resolve_forum(forum_key: str) -> str:
    """Map a parsed 'Target forum' label to a forum URL.

    Matches the longest FORUM_MAP key contained in the label first, so specific
    boards (e.g. 'nlp getting started') win over generic substrings
    ('getting started').
    """
    for key in sorted(FORUM_MAP, key=len, reverse=True):
        if key in forum_key:
            return FORUM_MAP[key]
    return DEFAULT_FORUM
```

Then replace the inline `next(...)` resolution inside `parse_drafts` (currently lines 144-146):

```python
        forum_url = next(
            (v for k, v in FORUM_MAP.items() if k in forum_key), DEFAULT_FORUM
        )
```

with:

```python
        forum_url = resolve_forum(forum_key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_discussion_scheduler.py -v`
Expected: PASS (the two new tests plus the existing suite).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/ops/discussion_scheduler.py tests/test_discussion_scheduler.py
git commit -m "fix: resolve discussion forum by longest-key-first to stop NLP misroute"
```

---

### Task 4: Stop recording phantom zero votes on Kaggle CLI failure

`fetch_vote_counts()` returns `{}` on any failure, so `cmd_snapshot` records `votes=0` for every notebook and writes the snapshot unconditionally — injecting phantom drop/recover deltas into the correlation log. Make a failed fetch return `None`, record votes as unknown (`None`), and have the report skip deltas for unknown votes.

**Files:**
- Modify: `kaggle_portfolio/ops/metadata_tracker.py` — `fetch_vote_counts` (97-127), `_merge_votes` (130-136), `cmd_snapshot` (143-170), `cmd_report` (215-228)
- Test: `tests/test_metadata_tracker.py` (append four functions; add `import subprocess` and `from types import SimpleNamespace` at the top if absent)

- [ ] **Step 1: Write the failing tests**

Ensure the test module has these imports near the top (add any that are missing):

```python
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
```

Append to `tests/test_metadata_tracker.py`:

```python
class TestVoteFetchFailure:
    def test_fetch_returns_none_on_cli_failure(self, monkeypatch):
        monkeypatch.setattr(tracker, "kaggle_command", lambda: ["kaggle"])
        monkeypatch.setattr(
            tracker.subprocess, "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        assert tracker.fetch_vote_counts() is None

    def test_merge_records_none_when_votes_unavailable(self, mock_root):
        meta = tracker.collect_metadata()
        merged = tracker._merge_votes(meta, None)
        assert all(entry["votes"] is None for entry in merged.values())

    def test_snapshot_records_unknown_votes_on_fetch_failure(self, mock_root):
        with patch.object(tracker, "fetch_vote_counts", return_value=None):
            rc = tracker.cmd_snapshot()
        assert rc == 0
        log = tracker._load_log()
        assert log[-1]["votes_available"] is False
        assert all(e["votes"] is None for e in log[-1]["notebooks"].values())

    def test_report_skips_phantom_delta_when_votes_unknown(self, mock_root, capsys):
        tracker.cmd_snapshot(votes={"feature-engineering": 10, "attention-guide": 5})
        with patch.object(tracker, "fetch_vote_counts", return_value=None):
            tracker.cmd_snapshot()
        rc = tracker.cmd_report()
        captured = capsys.readouterr()
        assert rc == 0
        # The phantom "votes dropped to 0" delta (-10) must not appear.
        assert "-10" not in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metadata_tracker.py::TestVoteFetchFailure -v`
Expected: FAIL (`fetch_vote_counts` returns `{}` not `None`; `_merge_votes` sets `0` not `None`; snapshot has no `votes_available` key).

- [ ] **Step 3: Implement the fix**

Replace `fetch_vote_counts` (lines 97-127) with:

```python
def fetch_vote_counts() -> dict[str, int] | None:
    """Fetch vote counts from Kaggle CLI for all owned kernels.

    Returns a dict mapping kernel slug to votes on success (possibly empty), or
    None if the Kaggle CLI call fails — so callers can distinguish 'fetch
    failed' from 'genuinely zero votes' instead of silently recording zeros.
    """
    votes: dict[str, int] = {}
    try:
        cli = kaggle_command()
        result = subprocess.run(
            [*cli, "kernels", "list", "--mine", "--csv", "--page-size", "50"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                f"{YELLOW}Vote fetch failed{RESET}: "
                f"{summarize_subprocess_error(result.stdout, result.stderr)}",
                file=sys.stderr,
            )
            return None

        reader = csv.DictReader(io.StringIO(result.stdout))
        for row in reader:
            ref = row.get("ref", "")
            slug = ref.split("/")[-1] if "/" in ref else ref
            vote_col = next(
                (k for k in row if "vote" in k.lower() or "upvote" in k.lower()),
                None,
            )
            if vote_col:
                try:
                    votes[slug] = int(row[vote_col] or 0)
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        print(f"{YELLOW}Vote fetch failed{RESET}: {exc}", file=sys.stderr)
        return None
    return votes
```

Replace `_merge_votes` (lines 130-136) with:

```python
def _merge_votes(metadata: dict[str, dict], votes: dict[str, int] | None) -> dict[str, dict]:
    """Merge vote counts into metadata entries, matching by slug.

    When ``votes`` is None (a failed fetch), record votes as None rather than 0
    so downstream reporting can distinguish 'unknown' from 'genuinely zero'.
    """
    for dir_name, entry in metadata.items():
        if votes is None:
            entry["votes"] = None
            continue
        kernel_id = entry.get("id", "")
        slug = kernel_id.split("/")[-1] if "/" in kernel_id else dir_name
        entry["votes"] = votes.get(slug, 0)
    return metadata
```

Replace `cmd_snapshot` (lines 143-170) with:

```python
def cmd_snapshot(dry_run: bool = False, votes: dict[str, int] | None = None) -> int:
    """Take a snapshot of all metadata + votes and append to the log."""
    metadata = collect_metadata()
    votes_unavailable = False
    if votes is None:
        votes = fetch_vote_counts()
        if votes is None:
            votes_unavailable = True

    metadata = _merge_votes(metadata, votes)

    snapshot = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "notebooks": metadata,
        "annotation": None,
        "votes_available": not votes_unavailable,
    }

    if dry_run:
        print(f"{YELLOW}DRY RUN{RESET} — would write snapshot with "
              f"{len(metadata)} notebooks")
        for name, entry in sorted(metadata.items()):
            print(f"  {name}: votes={entry.get('votes', '?')} "
                  f"title={entry.get('title', '?')[:50]}")
        return 0

    if votes_unavailable:
        print(f"{YELLOW}Warning{RESET}: vote counts unavailable (Kaggle CLI "
              "fetch failed); recording votes as unknown for this snapshot.",
              file=sys.stderr)

    log = _load_log()
    log.append(snapshot)
    _save_log(log)
    total_votes = sum((e.get("votes") or 0) for e in metadata.values())
    print(f"{GREEN}Snapshot saved{RESET} — {len(metadata)} notebooks, "
          f"{total_votes} total votes")
    return 0
```

In `cmd_report`, replace the delta block (lines 215-228) with:

```python
            c_votes = c.get("votes")
            p_votes = p.get("votes")
            votes_known = c_votes is not None and p_votes is not None
            vote_delta = (c_votes - p_votes) if votes_known else 0
            title_changed = p.get("title") != c.get("title") and p.get("title")
            keywords_changed = (
                set(p.get("keywords", [])) != set(c.get("keywords", []))
                and p.get("keywords") is not None
            )

            if (votes_known and vote_delta != 0) or title_changed or keywords_changed:
                entry = {
                    "timestamp": ts,
                    "notebook": name,
                    "vote_delta": vote_delta if votes_known else None,
                    "votes_now": c_votes,
                }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_metadata_tracker.py -v`
Expected: PASS (the four new tests plus the existing suite — existing tests pass explicit int vote dicts and are unaffected).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/ops/metadata_tracker.py tests/test_metadata_tracker.py
git commit -m "fix: stop recording phantom zero votes on Kaggle CLI failure"
```

---

### Task 5: Honor `--timeout` in kaggle_auth_doctor network probes

`--timeout` is wired to `probe_blob_upload_auth` but never used, and `probe_public_listing` runs `subprocess.run` with no `timeout=`, so a stalled connection can hang the preflight gate. Bound the subprocess probe and apply a socket-level timeout to the SDK probe.

**Files:**
- Modify: `kaggle_portfolio/ops/kaggle_auth_doctor.py` — add `import socket`; `probe_public_listing` (107-114); `probe_blob_upload_auth` (117-150); `main` call site (203)
- Test: `tests/test_kaggle_auth_doctor.py` (append two functions; add `import subprocess`)

- [ ] **Step 1: Write the failing tests**

Add `import subprocess` to the top of `tests/test_kaggle_auth_doctor.py`, then append:

```python
def test_probe_public_listing_passes_timeout_to_subprocess(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ref\nuser/x\n", stderr="")

    monkeypatch.setattr(kaggle_auth_doctor, "kaggle_command", lambda: ["kaggle"])
    monkeypatch.setattr(kaggle_auth_doctor.subprocess, "run", fake_run)

    ok, _ = kaggle_auth_doctor.probe_public_listing("owner", timeout=13)
    assert ok is True
    assert captured.get("timeout") == 13


def test_probe_public_listing_reports_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(kaggle_auth_doctor, "kaggle_command", lambda: ["kaggle"])
    monkeypatch.setattr(kaggle_auth_doctor.subprocess, "run", fake_run)

    ok, msg = kaggle_auth_doctor.probe_public_listing("owner", timeout=2)
    assert ok is False
    assert "timed out" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kaggle_auth_doctor.py::test_probe_public_listing_passes_timeout_to_subprocess tests/test_kaggle_auth_doctor.py::test_probe_public_listing_reports_timeout -v`
Expected: FAIL — `probe_public_listing` currently takes only `owner` (TypeError on the `timeout=` kwarg) and passes no `timeout` to `subprocess.run`.

- [ ] **Step 3: Implement the fix**

Add `import socket` to the imports at the top of `kaggle_portfolio/ops/kaggle_auth_doctor.py`.

Replace `probe_public_listing` (lines 107-114) with:

```python
def probe_public_listing(owner: str, timeout: int) -> tuple[bool, str]:
    cmd = [*kaggle_command(), "datasets", "list", "-s", owner, "--csv"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"public listing probe timed out after {timeout}s"
    if result.returncode != 0:
        return False, summarize_subprocess_error(result.stdout, result.stderr)
    reader = csv.DictReader(io.StringIO(result.stdout))
    count = sum(1 for _ in reader)
    return True, f"retrieved {count} public dataset rows"
```

Wrap the body of `probe_blob_upload_auth` (lines 117-150) so the socket default timeout is applied and restored. The function becomes:

```python
def probe_blob_upload_auth(timeout: int) -> tuple[bool, str]:
    """Check whether Kaggle's official upload-start flow accepts the local credentials."""
    temp_path: str | None = None
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        with tempfile.NamedTemporaryFile("wb", prefix="kaggle-auth-doctor-", suffix=".txt", delete=False) as handle:
            handle.write(b"auth-doctor upload probe\n")
            temp_path = handle.name

        api = KaggleApi()
        api.authenticate()

        request = ApiStartBlobUploadRequest()
        request.type = ApiBlobType.DATASET
        request.name = Path(temp_path).name
        request.content_length = os.path.getsize(temp_path)
        request.last_modified_epoch_seconds = int(os.path.getmtime(temp_path))

        with api.build_kaggle_client() as kaggle:
            response = kaggle.blobs.blob_api_client.start_blob_upload(request)

        create_url = str(getattr(response, "create_url", "") or "")
        token = str(getattr(response, "token", "") or "")
        if create_url and token:
            return True, "official upload-start probe succeeded"
        return False, "official upload-start probe returned no create_url/token"
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        if any(marker in lowered for marker in ("401", "403", "unauthenticated", "unauthorized")):
            return False, f"official upload-start probe rejected credentials ({message})"
        return False, f"official upload-start probe failed: {message}"
    finally:
        socket.setdefaulttimeout(previous_timeout)
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
```

Update the `main` call site (line 203) from:

```python
    listing_ok, listing_msg = probe_public_listing(expected_owner)
```

to:

```python
    listing_ok, listing_msg = probe_public_listing(expected_owner, timeout=args.timeout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kaggle_auth_doctor.py -v`
Expected: PASS (the two new tests plus the existing `test_probe_blob_upload_auth_status_classification`, which still passes — `setdefaulttimeout(1)`/restore is harmless).

- [ ] **Step 5: Commit**

```bash
git add kaggle_portfolio/ops/kaggle_auth_doctor.py tests/test_kaggle_auth_doctor.py
git commit -m "fix: honor --timeout in kaggle_auth_doctor network probes"
```

---

### Task 6: Full-suite verification

Confirm the whole suite is green after all five fixes (no regressions in either test root).

- [ ] **Step 1: Run both test suites**

Run:
```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest pi-automation/tests -q
```
Expected: both green — `tests` ≥ 703 passing + the new tests; `pi-automation/tests` ≥ 78 passing + the new notify test. 0 failures.

- [ ] **Step 2: Confirm clean tree**

Run: `git status`
Expected: working tree clean, 5 fix commits on `feat/grandmaster-phase-0-1`.

---

## Self-Review

**Spec coverage** (Phase 0 rails table): cookie-leak → Task 1; FORUM_MAP longest-key-first → Task 3; zero-vote→`None` + log error → Task 4; `--timeout` honored → Task 5; notify.py token redaction → Task 2. All five gating fixes covered. (Phase 1 measurement loop is a separate plan, per the scope split.)

**Placeholder scan:** every step contains the actual code or the exact command + expected result. No TBD/TODO/"handle edge cases".

**Type/name consistency:** `resolve_forum(forum_key: str) -> str` defined in Task 3 and called in `parse_drafts`. `fetch_vote_counts() -> dict | None`, `_merge_votes(metadata, votes | None)`, and the `votes_available` snapshot key defined in Task 4 are used consistently within that task's tests. `probe_public_listing(owner, timeout)` signature in Task 5 matches the new `main` call site and both new tests.
