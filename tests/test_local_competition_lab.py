from __future__ import annotations

import pandas as pd

from kaggle_portfolio.notebooks import local_competition_lab as lab


def test_march_seed_number_parses_seed_codes():
    assert lab._march_seed_number("W01") == 1.0
    assert lab._march_seed_number("X16b") == 16.0


def test_march_submission_pairs_parses_stage_ids():
    sample = pd.DataFrame({"ID": ["2026_1101_1102", "2026_2101_2102"]})

    parsed = lab._march_submission_pairs(sample)

    assert parsed.to_dict("records") == [
        {"ID": "2026_1101_1102", "Season": 2026, "Team1": 1101, "Team2": 1102},
        {"ID": "2026_2101_2102", "Season": 2026, "Team1": 2101, "Team2": 2102},
    ]


def test_march_matchups_uses_sorted_team_ids_and_binary_target():
    features = pd.DataFrame(
        [
            {"Season": 2026, "TeamID": 1101, "games": 30, "win_pct": 0.8, "avg_score": 75, "avg_allowed": 60, "avg_margin": 15, "recent_win_pct": 0.9, "recent_margin": 18, "elo": 1600, "seed": 1},
            {"Season": 2026, "TeamID": 1102, "games": 30, "win_pct": 0.7, "avg_score": 72, "avg_allowed": 64, "avg_margin": 8, "recent_win_pct": 0.7, "recent_margin": 9, "elo": 1540, "seed": 4},
        ]
    )
    games = pd.DataFrame(
        [
            {"Season": 2026, "WTeamID": 1102, "LTeamID": 1101},
        ]
    )

    matchups = lab._march_matchups(games, features, include_target=True)

    assert len(matchups) == 1
    row = matchups.iloc[0]
    assert row["Team1"] == 1101
    assert row["Team2"] == 1102
    assert row["target"] == 0
    assert row["elo_diff"] == 60
