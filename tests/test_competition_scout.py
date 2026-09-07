from __future__ import annotations

from datetime import datetime, timezone

from kaggle_portfolio.notebooks import competition_scout
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import (
    Competition,
    FakeKaggleClient,
    parse_csv,
)

# Shaped like real `kaggle competitions list --csv` output, banner included.
LISTING = (
    "Warning: Looks like you're using an outdated API Version\n"
    "ref,title,category,teamCount,deadline,userHasEntered\n"
    "https://www.kaggle.com/competitions/march-machine-learning-mania-2026,"
    "March Mania,Featured,772,2026-03-19T16:00:00Z,True\n"
    "https://www.kaggle.com/competitions/gan-getting-started,"
    "GAN Intro,Getting Started,21,2030-07-01T23:59:00Z,False\n"
)

NOW = datetime(2026, 3, 10, tzinfo=timezone.utc)


def _competitions() -> list[Competition]:
    return [Competition.from_row(row) for row in parse_csv(LISTING)]


def test_parse_deadline_datetime_normalizes_naive_values_to_utc():
    parsed = competition_scout.parse_deadline_datetime("2026-03-01T00:00:00")
    assert parsed.tzinfo == timezone.utc


def test_competition_slug_strips_a_full_url():
    featured, _ = _competitions()
    assert featured.slug == "march-machine-learning-mania-2026"


def test_score_competition_prefers_featured_active_board_over_getting_started():
    featured, evergreen = _competitions()
    assert competition_scout.score_competition(
        featured, NOW
    ) > competition_scout.score_competition(evergreen, NOW)


def test_fetch_competitions_deduplicates_and_skips_uncovered_categories():
    client = FakeKaggleClient(competitions=_competitions())
    found = competition_scout.fetch_competitions(client)
    # Only the four SCOUT_CATEGORIES are requested, so Getting Started is absent.
    assert [c.slug for c in found] == ["march-machine-learning-mania-2026"]


def test_main_writes_a_report_anchored_to_the_repo_layout(tmp_path):
    deps = Deps.for_test(
        tmp_path,
        today="2026-03-10",
        client=FakeKaggleClient(competitions=_competitions()),
    )
    assert competition_scout.main(["--update"], deps=deps) == 0
    report = deps.layout.scout_report
    assert report.exists()
    assert "| 1 | march-machine-learning-mania-2026 | 772 |" in report.read_text()


def test_main_reports_failure_when_kaggle_returns_nothing(tmp_path):
    deps = Deps.for_test(tmp_path, client=FakeKaggleClient(competitions=[]))
    assert competition_scout.main([], deps=deps) == 1
