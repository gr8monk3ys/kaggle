from __future__ import annotations

from datetime import timezone
from datetime import datetime

from kaggle_portfolio.notebooks import competition_scout
from kaggle_portfolio.shared import kaggle_utils


def test_kaggle_command_falls_back_to_module_cli(monkeypatch):
    monkeypatch.setattr(kaggle_utils, "kaggle_cli_path", lambda: None)
    monkeypatch.setattr(
        kaggle_utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "kaggle.cli" else None,
    )
    cmd = kaggle_utils.kaggle_command()
    assert cmd[1:] == ["-m", "kaggle.cli"]


def test_parse_csv_handles_standard_output():
    raw = "ref,title,teamCount,deadline\ncomp/sample,Sample,42,2026-03-01T00:00:00Z\n"
    rows = competition_scout._parse_csv(raw)
    assert len(rows) == 1
    assert rows[0]["ref"] == "comp/sample"
    assert rows[0]["teamCount"] == "42"


def test_parse_deadline_datetime_normalizes_naive_values_to_utc():
    parsed = competition_scout.parse_deadline_datetime("2026-03-01T00:00:00")
    assert parsed.tzinfo == timezone.utc


def test_normalize_competition_ref_handles_full_url():
    assert (
        competition_scout.normalize_competition_ref(
            "https://www.kaggle.com/competitions/march-machine-learning-mania-2026"
        )
        == "march-machine-learning-mania-2026"
    )


def test_score_competition_prefers_featured_active_board_over_getting_started():
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    featured = {
        "ref": "https://www.kaggle.com/competitions/march-machine-learning-mania-2026",
        "deadline": "2026-03-19T16:00:00Z",
        "category": "Featured",
        "teamCount": "772",
        "userHasEntered": "True",
    }
    evergreen = {
        "ref": "https://www.kaggle.com/competitions/gan-getting-started",
        "deadline": "2030-07-01T23:59:00Z",
        "category": "Getting Started",
        "teamCount": "21",
        "userHasEntered": "False",
    }

    assert competition_scout.score_competition(featured, now) > competition_scout.score_competition(evergreen, now)
