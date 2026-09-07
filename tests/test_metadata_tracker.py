"""Tests for metadata_tracker.py."""

import json
from unittest.mock import patch

import pytest

from kaggle_portfolio.ops import metadata_tracker as tracker
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import FakeKaggleClient, Kernel, KaggleError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deps(tmp_path):
    """A temporary repo with two notebooks, and a fake Kaggle behind it."""
    for name, title, keywords, gpu in (
        ("feature-engineering", "Feature Engineering Guide", ["eda", "tabular"], False),
        ("attention-guide", "Attention Mechanism Guide", ["nlp", "transformers"], True),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": f"user/{name}",
                    "title": title,
                    "keywords": keywords,
                    "enable_gpu": gpu,
                    "dataset_sources": [],
                    "competition_sources": [],
                }
            ),
            encoding="utf-8",
        )
    return Deps.for_test(tmp_path, client=FakeKaggleClient())


@pytest.fixture
def dry_deps(deps):
    """The same repo, with effects disabled."""
    return Deps.for_test(
        deps.layout.root, client=FakeKaggleClient(effects=False), effects=False
    )


class TestCollectMetadata:
    def test_finds_kernel_metadata_files(self, deps):
        meta = tracker.collect_metadata(deps)
        assert len(meta) == 2
        assert "feature-engineering" in meta
        assert "attention-guide" in meta

    def test_extracts_fields(self, deps):
        meta = tracker.collect_metadata(deps)
        fe = meta["feature-engineering"]
        assert fe["title"] == "Feature Engineering Guide"
        assert fe["keywords"] == ["eda", "tabular"]
        assert fe["enable_gpu"] is False

    def test_skips_hidden_directories(self, deps):
        hidden = deps.layout.root / ".hidden" / "bad"
        hidden.mkdir(parents=True)
        (hidden / "kernel-metadata.json").write_text('{"id": "x/y"}')
        meta = tracker.collect_metadata(deps)
        assert ".hidden/bad" not in meta


# ---------------------------------------------------------------------------
# Vote merging
# ---------------------------------------------------------------------------


class TestMergeVotes:
    def test_merges_matching_slugs(self, deps):
        meta = tracker.collect_metadata(deps)
        votes = {"feature-engineering": 15, "attention-guide": 8}
        merged = tracker._merge_votes(meta, votes)
        assert merged["feature-engineering"]["votes"] == 15
        assert merged["attention-guide"]["votes"] == 8

    def test_missing_votes_default_zero(self, deps):
        meta = tracker.collect_metadata(deps)
        votes = {"feature-engineering": 10}
        merged = tracker._merge_votes(meta, votes)
        assert merged["attention-guide"]["votes"] == 0


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_dry_run_does_not_write(self, deps, dry_deps):
        votes = {"feature-engineering": 5}
        rc = tracker.cmd_snapshot(dry_deps, votes=votes)
        assert rc == 0
        assert not tracker.log_path(deps).exists()

    def test_snapshot_writes_log(self, deps):
        votes = {"feature-engineering": 10, "attention-guide": 3}
        rc = tracker.cmd_snapshot(deps, votes=votes)
        assert rc == 0
        assert tracker.log_path(deps).exists()

        log = json.loads(tracker.log_path(deps).read_text(encoding="utf-8"))
        assert len(log) == 1
        assert "timestamp" in log[0]
        assert len(log[0]["notebooks"]) == 2

    def test_multiple_snapshots_append(self, deps):
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 8})

        log = json.loads(tracker.log_path(deps).read_text(encoding="utf-8"))
        assert len(log) == 2


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_annotate_requires_snapshot(self, deps):
        rc = tracker.cmd_annotate(deps, "feature-engineering", "test note")
        assert rc == 1  # no snapshots

    def test_annotate_adds_note(self, deps):
        tracker.cmd_snapshot(deps, votes={})
        rc = tracker.cmd_annotate(deps, "feature-engineering", "Changed title for SEO")
        assert rc == 0

        log = json.loads(tracker.log_path(deps).read_text(encoding="utf-8"))
        assert log[-1]["annotation"]["feature-engineering"] == "Changed title for SEO"

    def test_multiple_annotations(self, deps):
        tracker.cmd_snapshot(deps, votes={})
        tracker.cmd_annotate(deps, "feature-engineering", "Title change")
        tracker.cmd_annotate(deps, "attention-guide", "Added keywords")

        log = json.loads(tracker.log_path(deps).read_text(encoding="utf-8"))
        ann = log[-1]["annotation"]
        assert "feature-engineering" in ann
        assert "attention-guide" in ann


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_needs_two_snapshots(self, deps, capsys):
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 5})
        rc = tracker.cmd_report(deps)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Need at least 2" in captured.out

    def test_report_detects_vote_changes(self, deps, capsys):
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 8})
        rc = tracker.cmd_report(deps)
        assert rc == 0
        captured = capsys.readouterr()
        assert "+3" in captured.out or "3" in captured.out

    def test_report_json_output(self, deps, capsys):
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 10})
        # Clear captured output from snapshot commands
        capsys.readouterr()
        rc = tracker.cmd_report(deps, as_json=True)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert any(ch["vote_delta"] == 5 for ch in data)

    def test_report_no_changes(self, deps, capsys):
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(deps, votes={"feature-engineering": 5})
        rc = tracker.cmd_report(deps)
        assert rc == 0
        captured = capsys.readouterr()
        assert "No metadata or vote changes" in captured.out

    def test_empty_log_handled(self, deps, capsys):
        rc = tracker.cmd_report(deps)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Need at least 2" in captured.out


# ---------------------------------------------------------------------------
# Vote fetch failure
# ---------------------------------------------------------------------------


class TestVoteFetchFailure:
    def test_fetch_returns_none_when_kaggle_fails(self):
        client = FakeKaggleClient(fail_with=KaggleError("boom"))
        assert tracker.fetch_vote_counts(client) is None

    def test_fetch_maps_slug_to_votes(self):
        client = FakeKaggleClient(kernels=[Kernel("me/feature-engineering", "FE", 12)])
        assert tracker.fetch_vote_counts(client) == {"feature-engineering": 12}

    def test_merge_records_none_when_votes_unavailable(self, deps):
        meta = tracker.collect_metadata(deps)
        merged = tracker._merge_votes(meta, None)
        assert all(entry["votes"] is None for entry in merged.values())

    def test_snapshot_records_unknown_votes_on_fetch_failure(self, deps):
        with patch.object(tracker, "fetch_vote_counts", return_value=None):
            rc = tracker.cmd_snapshot(deps)
        assert rc == 0
        log = tracker._load_log(deps)
        assert log[-1]["votes_available"] is False
        assert all(e["votes"] is None for e in log[-1]["notebooks"].values())

    def test_report_skips_phantom_delta_when_votes_unknown(self, deps, capsys):
        tracker.cmd_snapshot(
            deps, votes={"feature-engineering": 10, "attention-guide": 5}
        )
        with patch.object(tracker, "fetch_vote_counts", return_value=None):
            tracker.cmd_snapshot(deps)
        rc = tracker.cmd_report(deps)
        captured = capsys.readouterr()
        assert rc == 0
        # The phantom "votes dropped to 0" delta (-10) must not appear.
        assert "-10" not in captured.out

    def test_report_handles_title_change_with_unknown_votes(self, deps):
        tracker.cmd_snapshot(
            deps, votes={"feature-engineering": 10, "attention-guide": 5}
        )
        meta_path = deps.layout.root / "feature-engineering" / "kernel-metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["title"] = "New SEO Title"
        meta_path.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(tracker, "fetch_vote_counts", return_value=None):
            tracker.cmd_snapshot(deps)
        rc = tracker.cmd_report(deps)  # must not raise
        assert rc == 0
