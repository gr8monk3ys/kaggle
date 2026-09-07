#!/usr/bin/env python3
"""Report emission: one place that knows how a report is named and written.

Six modules hand-rolled a private ``write_text``/``write_json``, and five
hand-rolled the dated-plus-latest pair — ``medal_ops.main`` six times,
``notebook_quality.main`` eight. The filenames were an implicit cross-module
contract: ``campaign_pack`` read ``latest-dataset-usability.json`` by string
literal, written by ``dataset_usability`` by a different string literal.

The registry below makes that contract a symbol. The emitter makes writing an
effect that can be turned off at the seam rather than by an ``if`` each caller
has to remember — see ``docs/adr/0002-dependencies-travel-as-one-object.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol


class Report(str):
    """A report's registered name. The dated and latest filenames derive from it."""

    __slots__ = ()


# --- the registry -----------------------------------------------------------
# Anything reading another module's output should import the name from here
# rather than retyping the string.

SCORECARD = Report("scorecard")
BADGE_PLAN = Report("badge-plan")
WEEKLY_PLAN = Report("weekly-plan")
PACE = Report("pace")
DIGEST = Report("digest")
SYNC = Report("sync")
DOCTOR = Report("doctor")
NOTEBOOK_QUALITY = Report("notebook-quality")
NOTEBOOK_QUALITY_FIXES = Report("notebook-quality-fixes")
DATASET_USABILITY = Report("dataset-usability")
DATASET_USABILITY_TRACKER = Report("dataset-usability-tracker")
USABILITY_BENCHMARK = Report("usability-benchmark")
PROMOTION_CAMPAIGN = Report("promotion-campaign")
CAMPAIGN_RUNBOOK = Report("campaign-runbook")
STALE_CONTENT = Report("stale-content")
LIVE_RATINGS = Report("live-ratings")

ALL_REPORTS = (
    SCORECARD,
    BADGE_PLAN,
    WEEKLY_PLAN,
    PACE,
    DIGEST,
    SYNC,
    DOCTOR,
    NOTEBOOK_QUALITY,
    NOTEBOOK_QUALITY_FIXES,
    DATASET_USABILITY,
    DATASET_USABILITY_TRACKER,
    USABILITY_BENCHMARK,
    PROMOTION_CAMPAIGN,
    CAMPAIGN_RUNBOOK,
    STALE_CONTENT,
    LIVE_RATINGS,
)


def dated_name(report: str, on: date, ext: str = "md") -> str:
    return f"{report}-{on.isoformat()}.{ext}"


def latest_name(report: str, ext: str = "md") -> str:
    return f"latest-{report}.{ext}"


# --- the seam ---------------------------------------------------------------


class ReportEmitter(Protocol):
    """Where a generated report goes."""

    reports_dir: Path

    def emit(
        self, report: str, content: Any, *, ext: str = ..., on: date = ...
    ) -> list[Path]: ...
    def latest_path(self, report: str, *, ext: str = ...) -> Path: ...


def _render(content: Any) -> str:
    if isinstance(content, str):
        return content if content.endswith("\n") else content + "\n"
    return json.dumps(content, indent=2) + "\n"


@dataclass
class WritingEmitter:
    """Writes the dated file and the latest pointer, and says where they went."""

    reports_dir: Path
    today: date
    quiet: bool = False

    def emit(
        self, report: str, content: Any, *, ext: str = "md", on: date | None = None
    ) -> list[Path]:
        rendered = _render(content)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        dated = self.reports_dir / dated_name(report, on or self.today, ext)
        latest = self.reports_dir / latest_name(report, ext)
        for path in (dated, latest):
            path.write_text(rendered, encoding="utf-8")
        if not self.quiet:
            print(f"{report} written: {dated}")
            print(f"Latest {report}: {latest}")
        return [dated, latest]

    def latest_path(self, report: str, *, ext: str = "md") -> Path:
        return self.reports_dir / latest_name(report, ext)


@dataclass
class RecordingEmitter:
    """Writes nothing and remembers what it was asked to write.

    Used by tests and by every ``--dry-run``. Before this, "dry run" was six
    hand-written conditionals and none of them covered report writing, so
    ``medal_ops sync --dry-run`` still wrote two files.
    """

    reports_dir: Path
    today: date
    quiet: bool = True
    emitted: list[tuple[str, str, Any]] = field(default_factory=list)

    def emit(
        self, report: str, content: Any, *, ext: str = "md", on: date | None = None
    ) -> list[Path]:
        self.emitted.append((report, ext, content))
        if not self.quiet:
            print(f"Would write {report} ({ext}) to {self.reports_dir}")
        return [
            self.reports_dir / dated_name(report, on or self.today, ext),
            self.reports_dir / latest_name(report, ext),
        ]

    def latest_path(self, report: str, *, ext: str = "md") -> Path:
        return self.reports_dir / latest_name(report, ext)

    def names(self) -> list[str]:
        return [name for name, _, _ in self.emitted]
