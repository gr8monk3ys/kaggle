#!/usr/bin/env python3
"""Everything ambient a command needs, in one object.

Built once at the CLI edge and passed down, so what a command can touch is
visible in its signature rather than hidden in import-time globals.

See ``docs/adr/0002-dependencies-travel-as-one-object.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kaggle_portfolio.shared.clock import Clock
from kaggle_portfolio.shared.kaggle_client import CliKaggleClient, KaggleClient
from kaggle_portfolio.shared.layout import RepoLayout


@dataclass
class Deps:
    """The seam between a command and the world outside it.

    ``effects`` is the single answer to "may this run change anything?". The
    Kaggle client is constructed with it, so a dry run cannot push, publish,
    submit or download even if a caller forgets to check.
    """

    layout: RepoLayout
    clock: Clock
    client: KaggleClient
    effects: bool = True

    @classmethod
    def resolve(
        cls,
        *,
        root: Path | str | None = None,
        output_root: Path | str | None = None,
        today: str | None = None,
        effects: bool = True,
        client: KaggleClient | None = None,
        timeout: int | None = None,
    ) -> "Deps":
        """Build production dependencies.

        Every ``main()`` calls this when it is handed no ``deps``, which is what
        keeps ``python -m kaggle_portfolio.<module>`` working — the container's
        cron jobs and health checks invoke modules that way.
        """
        return cls(
            layout=RepoLayout.resolve(root, output_root),
            clock=Clock.resolve(today),
            client=client
            if client is not None
            else CliKaggleClient(effects=effects, timeout=timeout),
            effects=effects,
        )

    @classmethod
    def for_test(
        cls,
        root: Path | str,
        *,
        today: str = "2026-01-01",
        client: KaggleClient | None = None,
        effects: bool = True,
        output_root: Path | str | None = None,
    ) -> "Deps":
        """Build dependencies against a temporary repo, with a fake Kaggle."""
        from kaggle_portfolio.shared.kaggle_client import FakeKaggleClient

        return cls(
            layout=RepoLayout.resolve(root, output_root),
            clock=Clock.fixed(today),
            client=client if client is not None else FakeKaggleClient(effects=effects),
            effects=effects,
        )

    # -- convenience --------------------------------------------------------

    @property
    def today(self):
        return self.clock.today

    def with_output_root(self, output_root) -> "Deps":
        """Same dependencies, writing reports elsewhere. Used by preflight."""
        return Deps(
            layout=self.layout.with_output_root(output_root),
            clock=self.clock,
            client=self.client,
            effects=self.effects,
        )

    def with_effects(self, effects: bool) -> "Deps":
        """Return a copy with effects toggled, rebuilding the client to match."""
        if effects == self.effects:
            return self
        client = self.client
        if isinstance(client, CliKaggleClient):
            client = CliKaggleClient(effects=effects)
        return Deps(
            layout=self.layout, clock=self.clock, client=client, effects=effects
        )
