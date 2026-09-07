from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kaggle_portfolio.datasets import dataset_publish_pipeline as pipeline
from kaggle_portfolio.shared.kaggle_client import (
    Dataset,
    FakeKaggleClient,
    KaggleError,
    parse_csv,
)
from kaggle_portfolio.shared.errors import CommandError


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


def _client(csv_text: str = "", **kwargs) -> FakeKaggleClient:
    datasets = (
        [Dataset.from_row(row) for row in parse_csv(csv_text)] if csv_text else []
    )
    return FakeKaggleClient(datasets=datasets, **kwargs)


def test_fetch_live_refs_unions_private_and_public_listings():
    client = _client("ref,title\nowner/a,A\nowner/b,B\nowner/c,C\n")
    refs, err = pipeline.fetch_live_refs(client, "owner")
    assert err is None
    assert refs == {"owner/a", "owner/b", "owner/c"}


def test_fetch_live_refs_filters_non_owner_refs():
    client = _client("ref,title\nowner/a,A\nsomeone-else/z,Z\nowner/b,B\nanother/w,W\n")
    refs, err = pipeline.fetch_live_refs(client, "owner")
    assert err is None
    assert refs == {"owner/a", "owner/b"}


def test_fetch_live_refs_returns_empty_set_for_owner_with_no_datasets():
    refs, err = pipeline.fetch_live_refs(_client(), "owner")
    assert err is None
    assert refs == set()


def test_fetch_live_refs_reports_kaggle_failure():
    client = FakeKaggleClient(fail_with=KaggleError("403 Forbidden"))
    refs, err = pipeline.fetch_live_refs(client, "owner")
    assert refs is None
    assert "403 Forbidden" in err


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
    monkeypatch.setattr(
        sys, "argv", ["dataset_publish_pipeline.py", "--sync-ui-metadata"]
    )
    with pytest.raises(CommandError, match="requires --apply"):
        pipeline.main()


def test_classify_live_state_handles_missing_lookup():
    assert pipeline.classify_live_state("owner/a", None) == "unknown"
    assert pipeline.classify_live_state(None, {"owner/a"}) == "unknown"
    assert pipeline.classify_live_state("owner/a", {"owner/a"}) == "live"
    assert pipeline.classify_live_state("owner/b", {"owner/a"}) == "draft"


def test_infer_owner_uses_majority_dataset_ref_owner():
    candidates = [
        _candidate(
            "datasets/a", ref="owner-one/a", score=90, live_state="draft", eligible=True
        ),
        _candidate(
            "datasets/b", ref="owner-one/b", score=90, live_state="draft", eligible=True
        ),
        _candidate(
            "datasets/c", ref="owner-two/c", score=90, live_state="draft", eligible=True
        ),
    ]

    assert pipeline.infer_owner(candidates) == "owner-one"


def test_select_targets_respects_draft_mode_and_max_items():
    candidates = [
        _candidate(
            "datasets/a", ref="owner/a", score=90, live_state="draft", eligible=True
        ),
        _candidate(
            "datasets/b", ref="owner/b", score=90, live_state="live", eligible=True
        ),
        _candidate(
            "datasets/c", ref="owner/c", score=90, live_state="draft", eligible=False
        ),
    ]

    draft_only = pipeline.select_targets(candidates, draft_only=True, max_items=0)
    assert [item.rel_path for item in draft_only] == ["datasets/a"]

    all_mode = pipeline.select_targets(candidates, draft_only=False, max_items=1)
    assert [item.rel_path for item in all_mode] == ["datasets/a"]


def test_summarize_output_prefers_real_error_line():
    message = pipeline.summarize_output(
        "warning one\nwarning two\n",
        "401 Client Error: Unauthorized for url: https://www.kaggle.com/api/v1/blobs/upload\n"
        "  warnings.warn(\n",
    )
    assert message.startswith("401 Client Error: Unauthorized")
