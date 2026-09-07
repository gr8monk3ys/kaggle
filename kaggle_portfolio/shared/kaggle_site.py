#!/usr/bin/env python3
"""The Kaggle *site* seam: one interface over the browser-driven path.

ADR-0001 put the Kaggle **CLI** behind one adapter. The browser path was never
covered and was still in the state the CLI was in before it: ``kaggle_browser``
looked like the adapter, but its interface handed callers a Playwright ``page``,
so every caller owned selectors, waits and clicks — and three other modules
copied its auth helpers rather than importing them.

Callers ask for Kaggle *actions* and get outcomes. Selectors, waits, session
state and challenge handling live behind the interface. See
``docs/adr/0005-kaggle-ui-behind-one-adapter.md``.

Playwright is imported lazily inside the adapter, the same guard
``kaggle_client`` uses for the Kaggle SDK: importing this module must not require
a browser to be installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class SiteError(RuntimeError):
    """An action against kaggle.com could not be completed."""


class NotAuthenticated(SiteError):
    """The session is not signed in."""


class BrowserChallenge(SiteError):
    """Kaggle served a Cloudflare / reCAPTCHA interstitial."""


@dataclass(frozen=True)
class SiteOutcome:
    """What an action did. ``url`` is set when the action produced a page."""

    ok: bool
    detail: str = ""
    url: str | None = None
    skipped: bool = False  # effects were disabled

    def __bool__(self) -> bool:
        return self.ok


class KaggleSite(Protocol):
    """What callers may know about kaggle.com through a browser.

    Every method here mutates something public and irreversible, which is why the
    adapter takes ``effects``: before this, ``--dry-run`` covered the Kaggle CLI
    and report writes but not a single browser action, so a dry run could still
    follow a user.
    """

    def upvote(self, url: str) -> SiteOutcome: ...
    def follow(self, username: str) -> SiteOutcome: ...
    def comment(self, thread_url: str, body: str) -> SiteOutcome: ...
    def upload_cover(self, ref: str, image: Path) -> SiteOutcome: ...
    def create_dataset_topic(self, ref: str, title: str, body: str) -> SiteOutcome: ...
    def create_forum_topic(
        self, forum_url: str, title: str, body: str
    ) -> SiteOutcome: ...


# ---------------------------------------------------------------------------
# Production adapter
# ---------------------------------------------------------------------------


class PlaywrightSite:
    """Drives kaggle.com in a real browser.

    Constructed with an open Playwright ``page``; ``kaggle_browser`` still owns
    launching the browser and establishing the session, because that machinery is
    sound — what was missing is an interface above it that speaks in actions.
    """

    def __init__(
        self,
        page: Any,
        *,
        effects: bool = True,
        timeout_ms: int = 20_000,
        base_url: str = "https://www.kaggle.com",
    ) -> None:
        self._page = page
        self._effects = effects
        self._timeout = timeout_ms
        # Injectable so the selectors can be exercised against a local fixture.
        # With the URL hardcoded, every action that builds its own address was
        # untestable without reaching kaggle.com.
        self._base = base_url.rstrip("/")
        self.skipped_effects: list[str] = []

    # -- plumbing -----------------------------------------------------------

    def _guard(self, description: str) -> SiteOutcome | None:
        if self._effects:
            return None
        self.skipped_effects.append(description)
        return SiteOutcome(
            True, f"skipped (effects disabled): {description}", skipped=True
        )

    def _open(self, url: str, settle_ms: int = 1500) -> None:
        self._page.goto(url, wait_until="domcontentloaded", timeout=self._timeout)
        self._page.wait_for_timeout(settle_ms)
        if self._is_challenge():
            raise BrowserChallenge(
                "Kaggle browser challenge detected. Clear the Cloudflare/reCAPTCHA "
                "check in a headed browser and retry with --manual-login."
            )
        if not self._is_authenticated():
            raise NotAuthenticated("Not authenticated")

    def _is_authenticated(self) -> bool:
        from kaggle_portfolio.shared import _browser_session as session

        return session.is_authenticated(self._page)

    def _is_challenge(self) -> bool:
        from kaggle_portfolio.shared import _browser_session as session

        return session.is_browser_challenge(self._page)

    def _first(self, *locators):
        from kaggle_portfolio.shared import _browser_session as session

        return session.first_available(*locators)

    # -- actions ------------------------------------------------------------

    def upvote(self, url: str) -> SiteOutcome:
        skipped = self._guard(f"upvote {url}")
        if skipped:
            return skipped
        self._open(url)
        button = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"upvote", re.IGNORECASE)
            ).first,
            self._page.locator('button[aria-label*="upvote" i]').first,
            self._page.locator('button[data-testid="upvote"]').first,
            self._page.locator('button[aria-label*="vote" i]').first,
        )
        if button is None:
            raise SiteError(f"Upvote button not found on {url}")
        if (button.get_attribute("aria-pressed") or "").lower() == "true":
            return SiteOutcome(True, f"already upvoted {url}", url=url)
        button.click(timeout=self._timeout)
        self._page.wait_for_timeout(1000)
        return SiteOutcome(True, f"upvoted {url}", url=url)

    def follow(self, username: str) -> SiteOutcome:
        skipped = self._guard(f"follow {username}")
        if skipped:
            return skipped
        profile = f"{self._base}/{username}"
        self._open(profile, settle_ms=1000)
        already = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"^following$", re.IGNORECASE)
            ).first,
            self._page.get_by_role(
                "button", name=re.compile(r"^unfollow$", re.IGNORECASE)
            ).first,
        )
        if already is not None:
            return SiteOutcome(True, f"already following {username}", url=profile)
        button = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"^follow$", re.IGNORECASE)
            ).first,
            self._page.locator('button[aria-label*="follow" i]').first,
        )
        if button is None:
            raise SiteError(f"Follow button not found for {username}")
        button.click(timeout=self._timeout)
        self._page.wait_for_timeout(1000)
        return SiteOutcome(True, f"followed {username}", url=profile)

    def comment(self, thread_url: str, body: str) -> SiteOutcome:
        skipped = self._guard(f"comment on {thread_url}")
        if skipped:
            return skipped
        self._open(thread_url, settle_ms=2000)
        box = self._first(
            self._page.get_by_role(
                "textbox", name=re.compile(r"comment|reply", re.IGNORECASE)
            ).first,
            self._page.locator('div[contenteditable="true"]').first,
            self._page.locator("textarea").first,
        )
        if box is None:
            raise SiteError(f"Reply box not found on {thread_url}")
        box.click(timeout=self._timeout)
        box.fill(body) if hasattr(box, "fill") else box.type(body)
        submit = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"^(comment|reply|post)$", re.IGNORECASE)
            ).first,
            self._page.locator('button[type="submit"]').first,
        )
        if submit is None:
            raise SiteError(f"Submit button not found on {thread_url}")
        submit.click(timeout=self._timeout)
        self._page.wait_for_timeout(2000)
        return SiteOutcome(True, f"commented on {thread_url}", url=thread_url)

    def upload_cover(self, ref: str, image: Path) -> SiteOutcome:
        skipped = self._guard(f"upload cover for {ref}")
        if skipped:
            return skipped
        self._open(f"{self._base}/datasets/{ref}/settings", settle_ms=2000)
        edit = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"edit image", re.IGNORECASE)
            ).first
        )
        if edit is None:
            raise SiteError(f"Edit Image button not found for {ref}")
        edit.click(timeout=self._timeout)
        self._page.wait_for_timeout(1500)
        file_input = self._page.locator('input[type="file"]').first
        if file_input.count() == 0:
            raise SiteError("File input not found in image modal")
        file_input.set_input_files(str(image))
        self._page.wait_for_timeout(2000)
        save = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"^save$", re.IGNORECASE)
            ).first
        )
        if save is None:
            raise SiteError("Save button not found in image modal")
        save.click(timeout=self._timeout)
        self._page.wait_for_timeout(2000)
        return SiteOutcome(True, f"uploaded {image.name} to {ref}")

    def create_dataset_topic(self, ref: str, title: str, body: str) -> SiteOutcome:
        skipped = self._guard(f"create dataset topic on {ref}")
        if skipped:
            return skipped
        self._open(f"{self._base}/datasets/{ref}/discussion", settle_ms=2000)
        return self._compose_topic(title, body)

    def create_forum_topic(self, forum_url: str, title: str, body: str) -> SiteOutcome:
        skipped = self._guard(f"create forum topic on {forum_url}")
        if skipped:
            return skipped
        self._open(forum_url, settle_ms=2000)
        return self._compose_topic(title, body)

    def _compose_topic(self, title: str, body: str) -> SiteOutcome:
        new_topic = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"new topic", re.IGNORECASE)
            ).first
        )
        if new_topic is None:
            raise SiteError("New Topic button not found")
        if new_topic.is_disabled():
            raise SiteError("New Topic button is disabled")
        new_topic.click(timeout=self._timeout)
        self._page.wait_for_timeout(1500)
        title_box = self._first(
            self._page.get_by_role(
                "textbox", name=re.compile(r"title", re.IGNORECASE)
            ).first
        )
        content_box = self._first(
            self._page.get_by_role(
                "textbox", name=re.compile(r"body|content", re.IGNORECASE)
            ).first,
            self._page.locator('div[contenteditable="true"]').first,
        )
        if title_box is None or content_box is None:
            raise SiteError("Discussion editor controls not found")
        title_box.fill(title)
        content_box.click(timeout=self._timeout)
        content_box.fill(body) if hasattr(content_box, "fill") else content_box.type(
            body
        )
        publish = self._first(
            self._page.get_by_role(
                "button", name=re.compile(r"^(publish|post)$", re.IGNORECASE)
            ).first
        )
        if publish is None:
            raise SiteError("Publish button not found")
        publish.click(timeout=self._timeout)
        self._page.wait_for_timeout(3000)
        url = str(self._page.url)
        if not re.search(r"/discussion/\d+", url):
            return SiteOutcome(False, f"topic did not publish; still at {url}", url=url)
        return SiteOutcome(True, f"posted {url}", url=url)


# ---------------------------------------------------------------------------
# Test / dry-run adapter
# ---------------------------------------------------------------------------


@dataclass
class FakeSite:
    """An in-memory kaggle.com. Two adapters are what make this a real seam.

    Ships in the package rather than in ``tests/`` so ``pi-automation``'s suite
    and every ``--dry-run`` path can reach it.
    """

    already_upvoted: set[str] = field(default_factory=set)
    already_following: set[str] = field(default_factory=set)
    fail_with: SiteError | None = None
    effects: bool = True
    posted_url: str = "https://www.kaggle.com/discussion/1"
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    skipped_effects: list[str] = field(default_factory=list)

    def _record(self, name: str, *args: Any) -> SiteOutcome | None:
        self.calls.append((name, args))
        if self.fail_with is not None:
            raise self.fail_with
        if not self.effects:
            self.skipped_effects.append(name)
            return SiteOutcome(
                True, f"skipped (effects disabled): {name}", skipped=True
            )
        return None

    def upvote(self, url: str) -> SiteOutcome:
        skipped = self._record("upvote", url)
        if skipped:
            return skipped
        if url in self.already_upvoted:
            return SiteOutcome(True, f"already upvoted {url}", url=url)
        self.already_upvoted.add(url)
        return SiteOutcome(True, f"upvoted {url}", url=url)

    def follow(self, username: str) -> SiteOutcome:
        skipped = self._record("follow", username)
        if skipped:
            return skipped
        if username in self.already_following:
            return SiteOutcome(True, f"already following {username}")
        self.already_following.add(username)
        return SiteOutcome(True, f"followed {username}")

    def comment(self, thread_url: str, body: str) -> SiteOutcome:
        skipped = self._record("comment", thread_url, body)
        return skipped or SiteOutcome(
            True, f"commented on {thread_url}", url=thread_url
        )

    def upload_cover(self, ref: str, image: Path) -> SiteOutcome:
        skipped = self._record("upload_cover", ref, image)
        return skipped or SiteOutcome(True, f"uploaded {Path(image).name} to {ref}")

    def create_dataset_topic(self, ref: str, title: str, body: str) -> SiteOutcome:
        skipped = self._record("create_dataset_topic", ref, title, body)
        return skipped or SiteOutcome(
            True, f"posted {self.posted_url}", url=self.posted_url
        )

    def create_forum_topic(self, forum_url: str, title: str, body: str) -> SiteOutcome:
        skipped = self._record("create_forum_topic", forum_url, title, body)
        return skipped or SiteOutcome(
            True, f"posted {self.posted_url}", url=self.posted_url
        )
