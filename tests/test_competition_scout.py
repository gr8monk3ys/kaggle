from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kaggle_portfolio.notebooks import competition_scout as scout
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import (
    Competition,
    FakeKaggleClient,
    parse_csv,
)

NOW = datetime(2026, 3, 10, tzinfo=timezone.utc)


def _comp(slug, category, teams, deadline, entered="False", title=None):
    row = parse_csv(
        "ref,title,category,teamCount,deadline,userHasEntered\n"
        f"https://www.kaggle.com/competitions/{slug},{title or slug},"
        f"{category},{teams},{deadline},{entered}\n"
    )[0]
    return Competition.from_row(row)


# Shaped like real `kaggle competitions list --csv` output, banner included.
LISTING = (
    "Warning: Looks like you're using an outdated API Version\n"
    "ref,title,category,teamCount,deadline,userHasEntered\n"
    "https://www.kaggle.com/competitions/march-machine-learning-mania-2026,"
    "March Mania,Featured,772,2026-05-19T16:00:00Z,True\n"
    "https://www.kaggle.com/competitions/gan-getting-started,"
    "GAN Intro,Getting Started,21,2030-07-01T23:59:00Z,False\n"
)


def _listing() -> list[Competition]:
    return [Competition.from_row(row) for row in parse_csv(LISTING)]


def test_parse_deadline_datetime_normalizes_naive_values_to_utc():
    parsed = scout.parse_deadline_datetime("2026-03-01T00:00:00")
    assert parsed.tzinfo == timezone.utc


def test_competition_slug_strips_a_full_url():
    featured, _ = _listing()
    assert featured.slug == "march-machine-learning-mania-2026"


@pytest.mark.parametrize(
    "category, medals",
    [
        ("Featured", True),
        ("Research", True),
        ("Playground", False),
        ("Getting Started", False),
    ],
)
def test_only_featured_and_research_boards_award_medals(category, medals):
    comp = _comp("c", category, 500, "2026-05-01T00:00:00Z")
    assert scout.awards_medals(comp) is medals
    assert (scout.medal_score(comp, NOW) is not None) is medals


@pytest.mark.parametrize(
    "teams, cutoff",
    [(100, 40), (249, 99), (250, 100), (999, 100), (1000, 100), (4000, 400)],
)
def test_bronze_cutoff_follows_kaggles_table(teams, cutoff):
    assert scout.bronze_cutoff(teams) == cutoff


def test_a_playground_board_still_ranks_for_notebook_votes():
    playground = _comp(
        "playground-series-s6e3", "Playground", 3000, "2026-05-01T00:00:00Z"
    )
    assert scout.medal_score(playground, NOW) is None
    assert scout.notebook_score(playground, NOW) is not None


def test_medal_score_prefers_runway_over_a_final_week_sprint():
    sprint = _comp("sprint", "Featured", 150, "2026-03-15T00:00:00Z")
    runway = _comp("runway", "Featured", 150, "2026-05-01T00:00:00Z")
    assert scout.medal_score(runway, NOW) > scout.medal_score(sprint, NOW)


def test_medal_score_prefers_a_generous_bronze_cutoff():
    small = _comp("small", "Research", 200, "2026-05-01T00:00:00Z")  # top 40%
    huge = _comp("huge", "Research", 5000, "2026-05-01T00:00:00Z")  # top 10%
    assert scout.medal_score(small, NOW) > scout.medal_score(huge, NOW)


def test_notebook_score_prefers_posting_early():
    early = _comp("early", "Playground", 2000, "2026-05-20T00:00:00Z")
    late = _comp("late", "Playground", 2000, "2026-03-15T00:00:00Z")
    assert scout.notebook_score(early, NOW) > scout.notebook_score(late, NOW)


def test_ended_boards_are_dropped_from_both_rankings():
    ended = _comp("ended", "Featured", 500, "2026-03-01T00:00:00Z")
    assert scout.medal_score(ended, NOW) is None
    assert scout.notebook_score(ended, NOW) is None


def test_fetch_competitions_deduplicates_and_skips_uncovered_categories():
    client = FakeKaggleClient(competitions=_listing())
    found = scout.fetch_competitions(client)
    # Getting Started is not a scouted category.
    assert [c.slug for c in found] == ["march-machine-learning-mania-2026"]


def test_main_writes_both_rankings_to_the_report(tmp_path):
    deps = Deps.for_test(
        tmp_path,
        today="2026-03-10",
        client=FakeKaggleClient(competitions=_listing()),
    )
    assert scout.main(["--update"], deps=deps) == 0
    text = deps.layout.scout_report.read_text()
    assert "## Competition medals (Featured and Research only)" in text
    assert "## Notebook votes (any active board)" in text
    assert "| 1 | march-machine-learning-mania-2026 | 772 | 70 | 100 (13%) |" in text


def test_main_reports_failure_when_kaggle_returns_nothing(tmp_path):
    deps = Deps.for_test(tmp_path, client=FakeKaggleClient(competitions=[]))
    assert scout.main([], deps=deps) == 1
