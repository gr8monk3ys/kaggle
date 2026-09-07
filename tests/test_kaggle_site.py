"""Tests for the Kaggle site seam.

Two layers, on purpose:

- ``FakeSite`` proves the callers, with no browser at all.
- ``PlaywrightSite`` is driven against a **static local HTML fixture**, which
  proves the selectors and the outcome shape without touching kaggle.com.
  Upvoting and following are irreversible and public; the account is the
  operator's, so no test may reach the real site.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_portfolio.shared.kaggle_site import (
    FakeSite,
    PlaywrightSite,
    SiteError,
    SiteOutcome,
)

FIXTURES = Path(__file__).parent / "fixtures" / "site"


class TestFakeSite:
    def test_upvote_is_idempotent(self):
        site = FakeSite()
        assert site.upvote("https://k/n").detail == "upvoted https://k/n"
        assert site.upvote("https://k/n").detail == "already upvoted https://k/n"

    def test_follow_is_idempotent(self):
        site = FakeSite()
        assert "followed" in site.follow("someone").detail
        assert "already following" in site.follow("someone").detail

    def test_every_action_is_recorded(self):
        site = FakeSite()
        site.upvote("u")
        site.follow("f")
        site.comment("t", "b")
        site.create_dataset_topic("o/d", "t", "b")
        assert [name for name, _ in site.calls] == [
            "upvote",
            "follow",
            "comment",
            "create_dataset_topic",
        ]

    def test_effects_off_performs_nothing(self):
        site = FakeSite(effects=False)
        outcome = site.follow("someone")
        assert outcome.skipped and outcome.ok
        assert site.already_following == set(), "a gated run must not change state"

    def test_topic_creation_returns_the_post_url(self):
        site = FakeSite(posted_url="https://www.kaggle.com/discussion/42")
        assert site.create_forum_topic("f", "t", "b").url.endswith("/42")

    def test_a_seeded_failure_propagates(self):
        site = FakeSite(fail_with=SiteError("challenge"))
        with pytest.raises(SiteError):
            site.upvote("u")


@pytest.fixture(scope="module")
def page():
    """A real Chromium page.

    The launch is guarded, not the yield: wrapping the yield would swallow a
    genuine test failure into a skip, which is how a browser test quietly stops
    testing anything.
    """
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - no browser binaries
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield browser.new_page()
        finally:
            browser.close()


class TestPlaywrightSiteAgainstAFixture:
    """The selectors, proven against local HTML — never against kaggle.com."""

    def test_upvote_finds_the_button_and_reports_the_outcome(self, page, monkeypatch):
        site = PlaywrightSite(page, timeout_ms=5000)
        monkeypatch.setattr(site, "_is_authenticated", lambda: True)
        monkeypatch.setattr(site, "_is_challenge", lambda: False)
        outcome = site.upvote((FIXTURES / "notebook.html").as_uri())
        assert isinstance(outcome, SiteOutcome)
        assert outcome.ok and outcome.detail.startswith("upvoted")

    def test_follow_finds_the_button(self, page, monkeypatch):
        # base_url points at the fixture directory; follow() builds
        # "{base}/{username}", so the username is the fixture filename.
        site = PlaywrightSite(page, timeout_ms=5000, base_url=FIXTURES.as_uri())
        monkeypatch.setattr(site, "_is_authenticated", lambda: True)
        monkeypatch.setattr(site, "_is_challenge", lambda: False)
        outcome = site.follow("profile.html")
        assert outcome.ok and outcome.detail.startswith("followed")

    def test_a_missing_button_is_a_SiteError_not_a_crash(self, page, monkeypatch):
        site = PlaywrightSite(page, timeout_ms=2000)
        monkeypatch.setattr(site, "_is_authenticated", lambda: True)
        monkeypatch.setattr(site, "_is_challenge", lambda: False)
        with pytest.raises(SiteError, match="Upvote button not found"):
            site.upvote((FIXTURES / "profile.html").as_uri())

    def test_effects_off_never_opens_the_page(self, page):
        site = PlaywrightSite(page, effects=False)
        outcome = site.upvote("https://www.kaggle.com/should-never-be-fetched")
        assert outcome.skipped
        assert site.skipped_effects == [
            "upvote https://www.kaggle.com/should-never-be-fetched"
        ]
