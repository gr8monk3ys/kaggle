from __future__ import annotations

import pandas as pd

from kaggle_portfolio.notebooks import local_competition_lab as lab


def test_deep_past_display_name_candidates_strip_cuneiform_prefix():
    row = pd.Series(
        {
            "label": "Cuneiform Tablet Kt 92/k 221 (AKT 5 1)",
            "aliases": "Kt 92/k 221 | AKT 5 1",
        }
    )

    candidates = lab._deep_past_display_name_candidates(row)

    assert "Kt 92/k 221 (AKT 5 1)" in candidates
    assert "AKT 5 1" in candidates


def test_deep_past_assign_sentences_uses_half_open_row_boundaries():
    test = pd.DataFrame(
        [
            {"id": 0, "line_start": 1, "line_end": 7},
            {"id": 1, "line_start": 7, "line_end": 14},
            {"id": 2, "line_start": 14, "line_end": 24},
            {"id": 3, "line_start": 25, "line_end": 30},
        ]
    )
    sentence_rows = pd.DataFrame(
        [
            {"line_number": 1, "translation": "a"},
            {"line_number": 6, "translation": "b"},
            {"line_number": 7, "translation": "c"},
            {"line_number": 8, "translation": "d"},
            {"line_number": 14, "translation": "e"},
            {"line_number": 25, "translation": "f"},
            {"line_number": 28, "translation": "g"},
        ]
    )

    predictions = lab._deep_past_assign_sentences_to_rows(test, sentence_rows)

    assert predictions == ["a b", "c d", "e", "f g"]


def test_deep_past_sentence_rows_match_stripped_display_name():
    sentences = pd.DataFrame(
        [
            {"display_name": "Kt 92/k 221 (AKT 5 1)", "line_number": 7, "translation": "Line 7"},
            {"display_name": "Kt 92/k 221 (AKT 5 1)", "line_number": 1, "translation": "Line 1"},
        ]
    )
    row = pd.Series({"label": "Cuneiform Tablet Kt 92/k 221 (AKT 5 1)", "aliases": "Kt 92/k 221 | AKT 5 1"})

    matched = lab._deep_past_sentence_rows(sentences, row)

    assert matched["translation"].tolist() == ["Line 1", "Line 7"]


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
