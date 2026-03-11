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


def test_deep_past_split_translation_handles_missing_line_end():
    test = pd.DataFrame(
        [
            {"id": 0, "line_start": 1, "line_end": 3},
            {"id": 1, "line_start": 4, "line_end": None},
        ]
    )

    predictions = lab._deep_past_split_translation_by_rows("alpha beta gamma delta", test)

    assert predictions == ["alpha beta gamma", "delta"]


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


def test_benchmark_deep_past_falls_back_without_auxiliary_files(tmp_path):
    pd.DataFrame(
        [
            {"transliteration": "a na", "translation": "go to"},
            {"transliteration": "dingir lugal", "translation": "the god king"},
        ]
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        [
            {"id": 1, "line_start": 1, "line_end": 1, "transliteration": "a"},
            {"id": 2, "line_start": 2, "line_end": 2, "transliteration": "na"},
        ]
    ).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame(
        [
            {"id": 1, "translation": "broken text"},
            {"id": 2, "translation": "broken text"},
        ]
    ).to_csv(tmp_path / "sample_submission.csv", index=False)

    result = lab.benchmark_deep_past(tmp_path, _folds=0, write_submission=False)

    assert result.best_model == "train_retrieval"
    assert any(
        row["model"] == "published_sentence_match" and row["available"] is False
        for row in result.benchmark_rows
    )


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


def test_benchmarks_include_new_entered_competitions():
    assert "playground-series-s6e3" in lab.BENCHMARKS
    assert "house-prices-advanced-regression-techniques" in lab.BENCHMARKS
    assert "store-sales-time-series-forecasting" in lab.BENCHMARKS


def test_playground_prepare_features_adds_telco_derivatives():
    train = pd.DataFrame(
        [
            {
                "id": 1,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 80.0,
                "TotalCharges": "960.0",
                "Churn": "No",
            }
        ]
    )
    test = train.drop(columns=["Churn"]).copy()

    train_x, test_x = lab._playground_prepare_features(train, test)

    assert "ChargesPerTenure" in train_x.columns
    assert "HasAutoPay" in train_x.columns
    assert train_x.loc[0, "HasFiber"] == 1
    assert train_x.loc[0, "HasAutoPay"] == 1
    assert test_x.loc[0, "HasStreaming"] == 1


def test_house_prepare_features_adds_core_engineering_columns():
    train = pd.DataFrame(
        [
            {
                "Id": 1,
                "MSSubClass": 20,
                "Neighborhood": "NAmes",
                "LotFrontage": 80.0,
                "TotalBsmtSF": 900,
                "1stFlrSF": 1000,
                "2ndFlrSF": 400,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "YrSold": 2010,
                "YearBuilt": 2000,
                "YearRemodAdd": 2005,
                "GarageArea": 500,
                "PoolArea": 0,
                "Fireplaces": 1,
                "WoodDeckSF": 10,
                "OpenPorchSF": 20,
                "EnclosedPorch": 0,
                "3SsnPorch": 0,
                "ScreenPorch": 30,
                "OverallQual": 7,
                "OverallCond": 5,
                "GrLivArea": 1400,
                "MasVnrArea": 100,
                "SalePrice": 200000,
            }
        ]
    )
    test = train.drop(columns=["SalePrice"]).copy()

    train_x, _test_x = lab._house_prepare_features(train, test)

    assert train_x.loc[0, "TotalSF"] == 2300
    assert train_x.loc[0, "TotalBath"] == 3.5
    assert train_x.loc[0, "QualSF"] == 9800
    assert train_x.loc[0, "TotalPorchSF"] == 60


def test_store_sales_prediction_frame_produces_complete_predictions():
    history = pd.DataFrame(
        [
            {"date": "2024-01-01", "store_nbr": 1, "family": "A", "onpromotion": 0, "sales": 10.0},
            {"date": "2024-01-08", "store_nbr": 1, "family": "A", "onpromotion": 0, "sales": 12.0},
            {"date": "2024-01-15", "store_nbr": 1, "family": "A", "onpromotion": 1, "sales": 15.0},
            {"date": "2024-01-22", "store_nbr": 1, "family": "A", "onpromotion": 1, "sales": 16.0},
        ]
    )
    target = pd.DataFrame(
        [
            {"date": "2024-01-29", "store_nbr": 1, "family": "A", "onpromotion": 1},
            {"date": "2024-01-30", "store_nbr": 1, "family": "A", "onpromotion": 0},
        ]
    )

    frame = lab._store_sales_prediction_frame(history, target)

    assert frame["recent_dow_promo_mean"].notna().all()
    assert frame["recent_28_mean"].notna().all()
    assert frame["hybrid_mean"].notna().all()
