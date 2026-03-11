"""Tests for the shared kaggle_browser module."""

import json
import re
from pathlib import Path

import pytest

import kaggle_browser as kb


# ---------------------------------------------------------------------------
# Fake Playwright objects (mirrors existing FakeLocator pattern)
# ---------------------------------------------------------------------------

class _FakeLocator:
    def __init__(self, count: int = 0):
        self._count = count

    @property
    def first(self):
        return self

    def count(self) -> int:
        return self._count

    def inner_text(self, timeout=None) -> str:
        return ""


class _ClickableLocator(_FakeLocator):
    def __init__(self, count: int = 0):
        super().__init__(count=count)
        self.clicked = False
        self.filled_value = None

    def click(self, timeout=None):
        self.clicked = True

    def fill(self, value: str, timeout=None):
        self.filled_value = value

    def get_attribute(self, name: str):
        return None


class _FakePage:
    def __init__(
        self,
        *,
        url: str = "https://www.kaggle.com/datasets",
        signed_out: bool = False,
        title_text: str = "",
        body_text: str = "",
    ):
        self.url = url
        self.signed_out = signed_out
        self._title_text = title_text
        self._body_text = body_text

    def title(self) -> str:
        return self._title_text

    def get_by_role(self, role: str, name=None):
        pattern = getattr(name, "pattern", "") if name is not None else ""
        if not self.signed_out:
            return _FakeLocator(0)
        if role in {"link", "button"} and "sign in" in str(pattern).lower():
            return _FakeLocator(1)
        if role == "link" and "register" in str(pattern).lower():
            return _FakeLocator(1)
        return _FakeLocator(0)

    def locator(self, selector: str):
        if selector == "body":
            locator = _FakeLocator(1)
            locator.inner_text = lambda timeout=None: self._body_text
            return locator
        return _FakeLocator(0)


# ---------------------------------------------------------------------------
# locator_count tests
# ---------------------------------------------------------------------------

class TestLocatorCount:
    def test_returns_count(self):
        assert kb.locator_count(_FakeLocator(3)) == 3

    def test_returns_zero_on_exception(self):
        class _Broken:
            def count(self):
                raise RuntimeError("broken")
        assert kb.locator_count(_Broken()) == 0


# ---------------------------------------------------------------------------
# first_available tests
# ---------------------------------------------------------------------------

class TestFirstAvailable:
    def test_returns_first_with_count(self):
        a = _FakeLocator(0)
        b = _FakeLocator(1)
        c = _FakeLocator(2)
        assert kb.first_available(a, b, c) is b

    def test_returns_none_when_all_empty(self):
        assert kb.first_available(_FakeLocator(0), _FakeLocator(0)) is None

    def test_skips_none(self):
        b = _FakeLocator(1)
        assert kb.first_available(None, b) is b


# ---------------------------------------------------------------------------
# is_authenticated tests
# ---------------------------------------------------------------------------

class TestIsAuthenticated:
    def test_authenticated_page(self):
        page = _FakePage(signed_out=False)
        assert kb.is_authenticated(page) is True

    def test_signed_out_page(self):
        page = _FakePage(signed_out=True)
        assert kb.is_authenticated(page) is False

    def test_login_url_is_not_authenticated(self):
        page = _FakePage(url="https://www.kaggle.com/account/login", signed_out=False)
        assert kb.is_authenticated(page) is False


class TestBrowserChallenge:
    def test_detects_challenge_from_title(self):
        page = _FakePage(title_text="Checking your browser - reCAPTCHA")
        assert kb.is_browser_challenge(page) is True

    def test_detects_challenge_from_body(self):
        page = _FakePage(body_text="Checking your browser before accessing www.kaggle.com ...")
        assert kb.is_browser_challenge(page) is True

    def test_non_challenge_page_returns_false(self):
        page = _FakePage(title_text="Datasets", body_text="Normal Kaggle page")
        assert kb.is_browser_challenge(page) is False


# ---------------------------------------------------------------------------
# TrackerFile tests
# ---------------------------------------------------------------------------

class TestTrackerFile:
    def test_empty_tracker(self, tmp_path):
        tracker = kb.TrackerFile(tmp_path / "tracker.json")
        assert tracker.has("key") is False
        assert tracker.completed == {}

    def test_mark_and_has(self, tmp_path):
        tracker = kb.TrackerFile(tmp_path / "tracker.json")
        tracker.mark("user1", "followed user1")
        assert tracker.has("user1") is True
        assert tracker.has("user2") is False

    def test_persistence(self, tmp_path):
        path = tmp_path / "tracker.json"
        t1 = kb.TrackerFile(path)
        t1.mark("a", "done")
        t1.save()

        t2 = kb.TrackerFile(path)
        assert t2.has("a") is True

    def test_corrupted_file_resets(self, tmp_path):
        path = tmp_path / "tracker.json"
        path.write_text("not json!!!", encoding="utf-8")
        tracker = kb.TrackerFile(path)
        assert tracker.completed == {}

    def test_mark_records_timestamp(self, tmp_path):
        tracker = kb.TrackerFile(tmp_path / "tracker.json")
        tracker.mark("k", "detail")
        entry = tracker.completed["k"]
        assert "at" in entry
        assert entry["detail"] == "detail"


# ---------------------------------------------------------------------------
# add_common_browser_args tests
# ---------------------------------------------------------------------------

class TestCommonBrowserArgs:
    def test_defaults(self):
        import argparse
        parser = argparse.ArgumentParser()
        kb.add_common_browser_args(parser)
        args = parser.parse_args([])
        assert args.headed is False
        assert args.dry_run is False
        assert args.timeout_ms == kb.DEFAULT_TIMEOUT_MS
        assert args.manual_login is False

    def test_flags(self):
        import argparse
        parser = argparse.ArgumentParser()
        kb.add_common_browser_args(parser)
        args = parser.parse_args(["--headed", "--dry-run", "--timeout-ms", "5000"])
        assert args.headed is True
        assert args.dry_run is True
        assert args.timeout_ms == 5000


# ---------------------------------------------------------------------------
# Cover image upload tests
# ---------------------------------------------------------------------------

class TestCoverImageDiscovery:
    def test_discovers_datasets_with_cover(self, tmp_path):
        import cover_image_upload as ciu
        ds = tmp_path / "datasets" / "test-ds"
        ds.mkdir(parents=True)
        (ds / "cover.png").write_bytes(b"fake png")
        (ds / "dataset-metadata.json").write_text(
            json.dumps({"id": "owner/test-ds"}), encoding="utf-8"
        )
        results = ciu.discover_cover_datasets(tmp_path / "datasets")
        assert len(results) == 1
        assert results[0][0] == "test-ds"
        assert results[0][1] == "owner/test-ds"

    def test_skips_without_cover(self, tmp_path):
        import cover_image_upload as ciu
        ds = tmp_path / "datasets" / "no-cover"
        ds.mkdir(parents=True)
        (ds / "dataset-metadata.json").write_text(
            json.dumps({"id": "owner/no-cover"}), encoding="utf-8"
        )
        results = ciu.discover_cover_datasets(tmp_path / "datasets")
        assert len(results) == 0

    def test_only_filter(self, tmp_path):
        import cover_image_upload as ciu
        for name in ("ds-a", "ds-b"):
            ds = tmp_path / "datasets" / name
            ds.mkdir(parents=True)
            (ds / "cover.png").write_bytes(b"fake")
            (ds / "dataset-metadata.json").write_text(
                json.dumps({"id": f"owner/{name}"}), encoding="utf-8"
            )
        results = ciu.discover_cover_datasets(tmp_path / "datasets", only="ds-b")
        assert len(results) == 1
        assert results[0][0] == "ds-b"


# ---------------------------------------------------------------------------
# Follow users tests
# ---------------------------------------------------------------------------

class TestFollowTargets:
    def test_load_targets(self, tmp_path):
        import follow_users as fu
        path = tmp_path / "targets.json"
        path.write_text(json.dumps({"users": ["alice", "bob", ""]}), encoding="utf-8")
        targets = fu.load_targets(path)
        assert targets == ["alice", "bob"]

    def test_load_missing_file(self, tmp_path):
        import follow_users as fu
        targets = fu.load_targets(tmp_path / "nope.json")
        assert targets == []

    def test_load_corrupted_file(self, tmp_path):
        import follow_users as fu
        path = tmp_path / "bad.json"
        path.write_text("not json!", encoding="utf-8")
        targets = fu.load_targets(path)
        assert targets == []


# ---------------------------------------------------------------------------
# Upvote content tests
# ---------------------------------------------------------------------------

class TestUpvoteUrlNormalization:
    def test_full_url_passthrough(self):
        import upvote_content as uc
        url = "https://www.kaggle.com/code/someone/notebook"
        assert uc.normalize_url(url) == url

    def test_notebook_slug(self):
        import upvote_content as uc
        assert uc.normalize_url("owner/nb", "notebook") == "https://www.kaggle.com/code/owner/nb"

    def test_dataset_slug(self):
        import upvote_content as uc
        assert uc.normalize_url("owner/ds", "dataset") == "https://www.kaggle.com/datasets/owner/ds"

    def test_discussion_slug(self):
        import upvote_content as uc
        assert uc.normalize_url("12345", "discussion") == "https://www.kaggle.com/discussions/12345"

    def test_tracker_key_strips_protocol(self):
        import upvote_content as uc
        key = uc.tracker_key("https://www.kaggle.com/code/owner/nb/")
        assert key == "www.kaggle.com/code/owner/nb"


class TestUpvoteQueue:
    def test_load_queue(self, tmp_path):
        import upvote_content as uc
        path = tmp_path / "queue.json"
        path.write_text(json.dumps({
            "items": [
                {"url": "https://kaggle.com/code/a/b", "type": "notebook"},
                {"type": "notebook"},  # missing url → filtered
            ]
        }), encoding="utf-8")
        items = uc.load_queue(path)
        assert len(items) == 1

    def test_load_missing_queue(self, tmp_path):
        import upvote_content as uc
        assert uc.load_queue(tmp_path / "nope.json") == []


# ---------------------------------------------------------------------------
# Comment thread tests
# ---------------------------------------------------------------------------

class TestCommentQueue:
    def test_load_queue(self, tmp_path):
        import comment_thread as ct
        path = tmp_path / "queue.json"
        path.write_text(json.dumps({
            "comments": [
                {"url": "https://kaggle.com/discussions/123", "body": "Great work!", "id": "c1"},
                {"url": "https://kaggle.com/discussions/456"},  # missing body → filtered
            ]
        }), encoding="utf-8")
        items = ct.load_comment_queue(path)
        assert len(items) == 1

    def test_comment_key_with_id(self):
        import comment_thread as ct
        key = ct.comment_key({"id": "c1", "url": "https://example.com", "body": "text"})
        assert key == "c1"

    def test_comment_key_fallback(self):
        import comment_thread as ct
        key = ct.comment_key({"url": "https://example.com", "body": "some text here"})
        assert "example.com" in key
        assert "some text" in key

    def test_load_missing_queue(self, tmp_path):
        import comment_thread as ct
        assert ct.load_comment_queue(tmp_path / "nope.json") == []


# ---------------------------------------------------------------------------
# human_delay sanity check
# ---------------------------------------------------------------------------

class TestHumanDelay:
    def test_runs_without_error(self):
        """Just verify it doesn't crash with minimal delay."""
        kb.human_delay(base=0.01, jitter=0.01)
