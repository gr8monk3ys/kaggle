"""Tests for metadata_tracker.py."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kaggle_portfolio.ops import metadata_tracker as tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_root(tmp_path):
    """Set up a temporary root with kernel-metadata.json files."""
    # Create two notebook directories
    nb1 = tmp_path / "feature-engineering"
    nb1.mkdir()
    (nb1 / "kernel-metadata.json").write_text(json.dumps({
        "id": "user/feature-engineering",
        "title": "Feature Engineering Guide",
        "keywords": ["eda", "tabular"],
        "enable_gpu": False,
        "dataset_sources": [],
        "competition_sources": [],
    }), encoding="utf-8")

    nb2 = tmp_path / "attention-guide"
    nb2.mkdir()
    (nb2 / "kernel-metadata.json").write_text(json.dumps({
        "id": "user/attention-guide",
        "title": "Attention Mechanism Guide",
        "keywords": ["nlp", "transformers"],
        "enable_gpu": True,
        "dataset_sources": [],
        "competition_sources": [],
    }), encoding="utf-8")

    medal_ops = tmp_path / "medal_ops"
    medal_ops.mkdir()

    with patch.object(tracker, "ROOT", tmp_path), \
         patch.object(tracker, "MEDAL_OPS_DIR", medal_ops), \
         patch.object(tracker, "LOG_PATH", medal_ops / "metadata_ab_log.json"):
        yield tmp_path


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------

class TestCollectMetadata:
    def test_finds_kernel_metadata_files(self, mock_root):
        meta = tracker.collect_metadata()
        assert len(meta) == 2
        assert "feature-engineering" in meta
        assert "attention-guide" in meta

    def test_extracts_fields(self, mock_root):
        meta = tracker.collect_metadata()
        fe = meta["feature-engineering"]
        assert fe["title"] == "Feature Engineering Guide"
        assert fe["keywords"] == ["eda", "tabular"]
        assert fe["enable_gpu"] is False

    def test_skips_hidden_directories(self, mock_root):
        hidden = mock_root / ".hidden" / "bad"
        hidden.mkdir(parents=True)
        (hidden / "kernel-metadata.json").write_text('{"id": "x/y"}')
        meta = tracker.collect_metadata()
        assert ".hidden/bad" not in meta


# ---------------------------------------------------------------------------
# Vote merging
# ---------------------------------------------------------------------------

class TestMergeVotes:
    def test_merges_matching_slugs(self, mock_root):
        meta = tracker.collect_metadata()
        votes = {"feature-engineering": 15, "attention-guide": 8}
        merged = tracker._merge_votes(meta, votes)
        assert merged["feature-engineering"]["votes"] == 15
        assert merged["attention-guide"]["votes"] == 8

    def test_missing_votes_default_zero(self, mock_root):
        meta = tracker.collect_metadata()
        votes = {"feature-engineering": 10}
        merged = tracker._merge_votes(meta, votes)
        assert merged["attention-guide"]["votes"] == 0


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_dry_run_does_not_write(self, mock_root):
        votes = {"feature-engineering": 5}
        rc = tracker.cmd_snapshot(dry_run=True, votes=votes)
        assert rc == 0
        assert not tracker.LOG_PATH.exists()

    def test_snapshot_writes_log(self, mock_root):
        votes = {"feature-engineering": 10, "attention-guide": 3}
        rc = tracker.cmd_snapshot(dry_run=False, votes=votes)
        assert rc == 0
        assert tracker.LOG_PATH.exists()

        log = json.loads(tracker.LOG_PATH.read_text(encoding="utf-8"))
        assert len(log) == 1
        assert "timestamp" in log[0]
        assert len(log[0]["notebooks"]) == 2

    def test_multiple_snapshots_append(self, mock_root):
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 8})

        log = json.loads(tracker.LOG_PATH.read_text(encoding="utf-8"))
        assert len(log) == 2


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class TestAnnotate:
    def test_annotate_requires_snapshot(self, mock_root):
        rc = tracker.cmd_annotate("feature-engineering", "test note")
        assert rc == 1  # no snapshots

    def test_annotate_adds_note(self, mock_root):
        tracker.cmd_snapshot(dry_run=False, votes={})
        rc = tracker.cmd_annotate("feature-engineering", "Changed title for SEO")
        assert rc == 0

        log = json.loads(tracker.LOG_PATH.read_text(encoding="utf-8"))
        assert log[-1]["annotation"]["feature-engineering"] == "Changed title for SEO"

    def test_multiple_annotations(self, mock_root):
        tracker.cmd_snapshot(dry_run=False, votes={})
        tracker.cmd_annotate("feature-engineering", "Title change")
        tracker.cmd_annotate("attention-guide", "Added keywords")

        log = json.loads(tracker.LOG_PATH.read_text(encoding="utf-8"))
        ann = log[-1]["annotation"]
        assert "feature-engineering" in ann
        assert "attention-guide" in ann


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_needs_two_snapshots(self, mock_root, capsys):
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 5})
        rc = tracker.cmd_report()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Need at least 2" in captured.out

    def test_report_detects_vote_changes(self, mock_root, capsys):
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 8})
        rc = tracker.cmd_report()
        assert rc == 0
        captured = capsys.readouterr()
        assert "+3" in captured.out or "3" in captured.out

    def test_report_json_output(self, mock_root, capsys):
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 10})
        # Clear captured output from snapshot commands
        capsys.readouterr()
        rc = tracker.cmd_report(as_json=True)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert any(ch["vote_delta"] == 5 for ch in data)

    def test_report_no_changes(self, mock_root, capsys):
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 5})
        tracker.cmd_snapshot(dry_run=False, votes={"feature-engineering": 5})
        rc = tracker.cmd_report()
        assert rc == 0
        captured = capsys.readouterr()
        assert "No metadata or vote changes" in captured.out

    def test_empty_log_handled(self, mock_root, capsys):
        rc = tracker.cmd_report()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Need at least 2" in captured.out
