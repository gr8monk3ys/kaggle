from __future__ import annotations

import sys
from pathlib import Path

import pytest

import dataset_publish_pipeline as pipeline


def _candidate(
    rel: str,
    *,
    ref: str | None,
    score: int,
    live_state: str,
    eligible: bool,
) -> pipeline.PublishCandidate:
    return pipeline.PublishCandidate(
        rel_path=rel,
        dir_path=Path(rel),
        dataset_ref=ref,
        score=score,
        score_10=max(0, min(10, (score + 9) // 10)),
        tier="Good",
        live_state=live_state,
        eligible=eligible,
        blocked_reasons=[] if eligible else ["blocked"],
    )


def test_parse_live_refs_csv_extracts_refs():
    raw = "ref,title\nowner/a,A\nowner/b,B\n"
    refs = pipeline.parse_live_refs_csv(raw)
    assert refs == {"owner/a", "owner/b"}
def test_parse_live_refs_csv_ignores_preamble_warning():
    raw = "Warning: outdated API\nref,title\nowner/a,A\n"
    refs = pipeline.parse_live_refs_csv(raw)
    assert refs == {"owner/a"}
def test_summarize_subprocess_error_prefers_real_error_line():
    message = pipeline.summarize_subprocess_error(
        "warning one\nwarning two\n",
        "401 Client Error: Unauthorized for url: https://www.kaggle.com/api/v1/blobs/upload\n"
        "  warnings.warn(\n",
    )
    assert message.startswith("401 Client Error: Unauthorized")


def test_classify_live_state_handles_missing_lookup():
    assert pipeline.classify_live_state("owner/a", None) == "unknown"
    assert pipeline.classify_live_state(None, {"owner/a"}) == "unknown"
    assert pipeline.classify_live_state("owner/a", {"owner/a"}) == "live"
    assert pipeline.classify_live_state("owner/b", {"owner/a"}) == "draft"


def test_select_targets_respects_draft_mode_and_max_items():
    candidates = [
        _candidate("datasets/a", ref="owner/a", score=90, live_state="draft", eligible=True),
        _candidate("datasets/b", ref="owner/b", score=90, live_state="live", eligible=True),
        _candidate("datasets/c", ref="owner/c", score=90, live_state="draft", eligible=False),
    ]

    draft_only = pipeline.select_targets(candidates, draft_only=True, max_items=0)
    assert [item.rel_path for item in draft_only] == ["datasets/a"]

    all_mode = pipeline.select_targets(candidates, draft_only=False, max_items=1)
    assert [item.rel_path for item in all_mode] == ["datasets/a"]


def test_infer_owner_uses_majority_dataset_ref_owner():
    candidates = [
        _candidate("datasets/a", ref="owner-one/a", score=90, live_state="draft", eligible=True),
        _candidate("datasets/b", ref="owner-one/b", score=90, live_state="draft", eligible=True),
        _candidate("datasets/c", ref="owner-two/c", score=90, live_state="draft", eligible=True),
    ]

    assert pipeline.infer_owner(candidates) == "owner-one"


def test_fetch_live_refs_combines_mine_and_search(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        if args == ["--mine", "--csv"]:
            return {"owner/a", "owner/b"}, None
        if args == ["-s", "owner", "--csv"]:
            return {"owner/c"}, None
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(pipeline, "_run_dataset_list", fake_run)

    refs, err = pipeline.fetch_live_refs("owner")

    assert err is None
    assert refs == {"owner/a", "owner/b", "owner/c"}
    assert calls == [["--mine", "--csv"], ["-s", "owner", "--csv"]]


def test_fetch_live_refs_filters_non_owner_refs(monkeypatch):
    def fake_run(args):
        if args == ["--mine", "--csv"]:
            return {"owner/a", "someone-else/z"}, None
        if args == ["-s", "owner", "--csv"]:
            return {"owner/b", "another/w"}, None
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(pipeline, "_run_dataset_list", fake_run)

    refs, err = pipeline.fetch_live_refs("owner")

    assert err is None
    assert refs == {"owner/a", "owner/b"}


def test_build_ui_sync_command_includes_refs_and_flags():
    cmd = pipeline.build_ui_sync_command(
        ["owner/a", "owner/b"],
        headed=True,
        timeout_ms=12345,
        manual_login=False,
    )

    assert cmd[0] == sys.executable
    assert cmd[1].endswith("pi-automation/scripts/dataset_metadata_sync.py")
    assert "--apply" in cmd
    assert "--headed" in cmd
    assert "--no-manual-login" in cmd
    assert "--timeout-ms" in cmd
    assert "12345" in cmd
    assert cmd.count("--dataset-ref") == 2
    assert "owner/a" in cmd and "owner/b" in cmd


def test_main_rejects_sync_ui_without_apply(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dataset_publish_pipeline.py", "--sync-ui-metadata"])
    with pytest.raises(SystemExit, match="requires --apply"):
        pipeline.main()
