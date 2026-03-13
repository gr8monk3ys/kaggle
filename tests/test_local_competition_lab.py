from __future__ import annotations

import sys
import types

import numpy as np
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


def test_benchmark_playground_prefers_advanced_model_when_available(tmp_path, monkeypatch):
    train = pd.DataFrame(
        [
            {
                "id": idx,
                "gender": "Male" if idx % 2 == 0 else "Female",
                "SeniorCitizen": idx % 2,
                "Partner": "Yes" if idx % 3 else "No",
                "Dependents": "No" if idx % 2 else "Yes",
                "tenure": 10 + idx,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if idx % 2 else "DSL",
                "OnlineSecurity": "Yes" if idx % 2 else "No",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes" if idx % 2 else "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month" if idx % 2 else "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 60.0 + idx,
                "TotalCharges": 600.0 + 20 * idx,
                "Churn": "Yes" if idx % 2 else "No",
            }
            for idx in range(6)
        ]
    )
    test = train.drop(columns=["Churn"]).head(2).copy()
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)

    monkeypatch.setattr(lab, "_playground_original_path", lambda _data_dir: tmp_path / "orig.csv")
    monkeypatch.setattr(
        lab,
        "_playground_model_result",
        lambda _model, _train_x, _test_x, _y, _cv: (0.91001, np.array([0.1] * 6), np.array([0.4, 0.6])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_lightgbm_result",
        lambda _train, _test, _orig, _folds: (0.9188, np.array([0.18] * 6), np.array([0.3, 0.7])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_xgboost_result",
        lambda _train, _test, _orig, _folds: (0.91999, np.array([0.2] * 6), np.array([0.25, 0.75])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_xgboost_pseudo_result",
        lambda _train, _test, _orig, _folds: (0.9192, np.array([0.19] * 6), np.array([0.2, 0.8])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_catboost_result",
        lambda _train, _test, _orig, _folds: (0.9185, np.array([0.15] * 6), np.array([0.35, 0.65])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_best_blend",
        lambda _predictions, _y, step=0.05: None,
    )
    pd.DataFrame({"dummy": [1]}).to_csv(tmp_path / "orig.csv", index=False)

    result = lab.benchmark_playground_telco(tmp_path, folds=3, write_submission=True)

    assert result.best_model == "xgboost_te"
    assert any(row["model"] == "xgboost_te" and row["score"] == 0.91999 for row in result.benchmark_rows)
    submission = pd.read_csv(result.submission_path)
    assert submission["Churn"].tolist() == [0.25, 0.75]


def test_benchmark_playground_prefers_blend_when_it_wins(tmp_path, monkeypatch):
    train = pd.DataFrame(
        [
            {
                "id": idx,
                "gender": "Male" if idx % 2 == 0 else "Female",
                "SeniorCitizen": idx % 2,
                "Partner": "Yes" if idx % 3 else "No",
                "Dependents": "No" if idx % 2 else "Yes",
                "tenure": 10 + idx,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if idx % 2 else "DSL",
                "OnlineSecurity": "Yes" if idx % 2 else "No",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes" if idx % 2 else "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month" if idx % 2 else "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 60.0 + idx,
                "TotalCharges": 600.0 + 20 * idx,
                "Churn": "Yes" if idx % 2 else "No",
            }
            for idx in range(6)
        ]
    )
    test = train.drop(columns=["Churn"]).head(2).copy()
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)

    monkeypatch.setattr(
        lab,
        "_playground_model_result",
        lambda _model, _train_x, _test_x, _y, _cv: (0.915, np.array([0.1] * 6), np.array([0.4, 0.6])),
    )
    monkeypatch.setattr(lab, "_playground_original_path", lambda _data_dir: None)
    monkeypatch.setattr(
        lab,
        "_playground_best_blend",
        lambda _predictions, _y, step=0.05: (
            "rank",
            {"lightgbm": 0.35, "xgboost": 0.45, "catboost_te": 0.2},
            0.92001,
            np.array([0.3, 0.7]),
        ),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_catboost_result",
        lambda _train, _test, _orig, _folds: (0.917, np.array([0.18] * 6), np.array([0.45, 0.55])),
    )

    result = lab.benchmark_playground_telco(tmp_path, folds=3, write_submission=True)

    assert result.best_model == "blend"
    assert any(
        row["model"] == "blend"
        and row["score"] == 0.92001
        and row["blend_type"] == "rank"
        and row["weights"] == {"lightgbm": 0.35, "xgboost": 0.45, "catboost_te": 0.2}
        for row in result.benchmark_rows
    )
    submission = pd.read_csv(result.submission_path)
    assert submission["Churn"].tolist() == [0.3, 0.7]


def test_benchmark_playground_prefers_pseudo_model_when_it_wins(tmp_path, monkeypatch):
    train = pd.DataFrame(
        [
            {
                "id": idx,
                "gender": "Male" if idx % 2 == 0 else "Female",
                "SeniorCitizen": idx % 2,
                "Partner": "Yes" if idx % 3 else "No",
                "Dependents": "No" if idx % 2 else "Yes",
                "tenure": 10 + idx,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if idx % 2 else "DSL",
                "OnlineSecurity": "Yes" if idx % 2 else "No",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes" if idx % 2 else "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month" if idx % 2 else "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 60.0 + idx,
                "TotalCharges": 600.0 + 20 * idx,
                "Churn": "Yes" if idx % 2 else "No",
            }
            for idx in range(6)
        ]
    )
    test = train.drop(columns=["Churn"]).head(2).copy()
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)

    monkeypatch.setattr(lab, "_playground_original_path", lambda _data_dir: tmp_path / "orig.csv")
    monkeypatch.setattr(
        lab,
        "_playground_model_result",
        lambda _model, _train_x, _test_x, _y, _cv: (0.91001, np.array([0.1] * 6), np.array([0.4, 0.6])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_lightgbm_result",
        lambda _train, _test, _orig, _folds: (0.9188, np.array([0.18] * 6), np.array([0.3, 0.7])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_xgboost_result",
        lambda _train, _test, _orig, _folds: (0.91999, np.array([0.2] * 6), np.array([0.25, 0.75])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_xgboost_pseudo_result",
        lambda _train, _test, _orig, _folds: (0.92055, np.array([0.22] * 6), np.array([0.15, 0.85])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_advanced_catboost_result",
        lambda _train, _test, _orig, _folds: (0.9185, np.array([0.15] * 6), np.array([0.35, 0.65])),
    )
    monkeypatch.setattr(
        lab,
        "_playground_best_blend",
        lambda _predictions, _y, step=0.05: None,
    )
    pd.DataFrame({"dummy": [1]}).to_csv(tmp_path / "orig.csv", index=False)

    result = lab.benchmark_playground_telco(tmp_path, folds=3, write_submission=True)

    assert result.best_model == "xgboost_te_pseudo"
    assert any(row["model"] == "xgboost_te_pseudo" and row["score"] == 0.92055 for row in result.benchmark_rows)
    submission = pd.read_csv(result.submission_path)
    assert submission["Churn"].tolist() == [0.15, 0.85]


def test_playground_best_blend_can_prefer_rank_average():
    y = pd.Series([0, 1, 0, 1])
    predictions = {
        "model_a": (
            np.array([0.2, 0.8, 0.4, 0.6]),
            np.array([0.1, 0.9]),
        ),
        "model_b": (
            np.array([0.3, 0.7, 0.1, 0.9]),
            np.array([0.2, 0.8]),
        ),
    }

    blend = lab._playground_best_blend(predictions, y, step=0.5)

    assert blend is not None
    blend_type, weights, score, pred = blend
    assert blend_type in {"rank", "probability"}
    assert round(sum(weights.values()), 6) == 1.0
    assert score >= 0.5
    assert pred.shape == (2,)


def test_playground_best_blend_handles_four_models_by_using_subsets():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    predictions = {
        "model_a": (np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]), np.array([0.2, 0.8])),
        "model_b": (np.array([0.2, 0.8, 0.25, 0.75, 0.4, 0.6]), np.array([0.3, 0.7])),
        "model_c": (np.array([0.05, 0.95, 0.4, 0.6, 0.35, 0.65]), np.array([0.4, 0.6])),
        "model_d": (np.array([0.3, 0.7, 0.1, 0.9, 0.2, 0.8]), np.array([0.5, 0.5])),
    }

    blend = lab._playground_best_blend(predictions, y, step=0.5)

    assert blend is not None
    blend_type, weights, score, pred = blend
    assert blend_type in {"rank", "probability"}
    assert 2 <= len(weights) <= 3
    assert round(sum(weights.values()), 6) == 1.0
    assert score >= 0.5
    assert pred.shape == (2,)


def test_playground_catboost_selected_features_prunes_heavy_te_only_columns():
    selected = lab._playground_catboost_selected_features(
        [
            "tenure",
            "Contract",
            "FREQ_tenure",
            "BG_Contract_InternetService",
            "ORIG_proba_Contract",
            "ORIG_proba_BG_Contract_InternetService",
            "CAT_CNT_Contract",
            "CAT_RARE_Contract",
            "CAT_tenure",
        ]
    )

    assert "tenure" in selected
    assert "Contract" in selected
    assert "FREQ_tenure" in selected
    assert "BG_Contract_InternetService" in selected
    assert "ORIG_proba_Contract" in selected
    assert "ORIG_proba_BG_Contract_InternetService" not in selected
    assert "CAT_CNT_Contract" not in selected
    assert "CAT_RARE_Contract" not in selected
    assert "CAT_tenure" not in selected


def test_playground_lightgbm_selected_features_keeps_counts_and_interactions():
    selected = lab._playground_lightgbm_selected_features(
        [
            "tenure",
            "Contract",
            "FREQ_tenure",
            "CAT_CNT_Contract",
            "CAT_RARE_Contract",
            "BG_Contract_InternetService",
            "TG_Contract_InternetService_PaymentMethod",
            "ORIG_proba_Contract",
            "CAT_tenure",
        ]
    )

    assert "tenure" in selected
    assert "Contract" in selected
    assert "FREQ_tenure" in selected
    assert "CAT_CNT_Contract" in selected
    assert "CAT_RARE_Contract" in selected
    assert "BG_Contract_InternetService" in selected
    assert "ORIG_proba_Contract" in selected
    assert "TG_Contract_InternetService_PaymentMethod" not in selected
    assert "CAT_tenure" not in selected


def test_playground_lightgbm_te_columns_focuses_on_core_categories_and_bins():
    selected = lab._playground_lightgbm_te_columns(
        [
            "Contract",
            "tenure_bin",
            "BG_Contract_InternetService",
            "BG_Contract_InternetService_PaymentMethod",
            "CAT_CNT_Contract",
            "ORIG_proba_Contract",
        ]
    )

    assert selected == [
        "Contract",
        "tenure_bin",
        "BG_Contract_InternetService",
    ]


def test_playground_pseudo_label_helpers_focus_on_confident_predictions():
    predictions = np.array([0.01, 0.04, 0.18, 0.51, 0.83, 0.96, 0.99])

    mask = lab._playground_pseudo_label_mask(predictions)
    weights = lab._playground_pseudo_label_weights(predictions[mask])

    assert mask.tolist() == [True, False, False, False, False, False, True]
    assert weights.shape == (2,)
    assert weights.min() >= 0.2
    assert weights.max() <= 0.4


def test_playground_advanced_feature_frames_add_ngram_and_distribution_columns():
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
                "TotalCharges": 960.0,
                "Churn": "No",
            },
            {
                "id": 2,
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "Yes",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 60.0,
                "TotalCharges": 1440.0,
                "Churn": "Yes",
            },
        ]
    )
    test = train.drop(columns=["Churn"]).head(1).copy()
    orig = train.copy()

    train_frame, test_frame, feature_cols, te_cols, drop_raw_cols = lab._playground_advanced_feature_frames(
        train,
        test,
        orig,
    )

    for col in [
        "BG_Contract_InternetService",
        "TG_Contract_InternetService_PaymentMethod",
        "pctrank_orig_TC",
        "cond_pctrank_IS_TC",
        "resid_IS_MC",
    ]:
        assert col in feature_cols
        assert col in train_frame.columns
    assert "BG_Contract_InternetService" in te_cols
    assert "BG_Contract_InternetService" in drop_raw_cols
    assert str(train_frame["BG_Contract_InternetService"].dtype) == "category"
    assert test_frame["pctrank_orig_TC"].notna().all()


def test_playground_advanced_xgboost_result_averages_across_seed_ensemble(monkeypatch):
    train_frame = pd.DataFrame(
        {
            "num": list(map(float, range(12))),
            "Churn": [0, 1] * 6,
        }
    )
    test_frame = pd.DataFrame({"num": [10.0, 11.0]})

    monkeypatch.setattr(
        lab,
        "_playground_advanced_feature_frames",
        lambda _train, _test, _orig: (train_frame.copy(), test_frame.copy(), ["num"], [], []),
    )

    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            self.seed = kwargs["random_state"]

        def fit(self, _x_train, _y_train, eval_set=None, verbose=False):
            return self

        def predict_proba(self, frame):
            seed_score = {
                11: 0.11,
                42: 0.42,
                99: 0.99,
            }[self.seed]
            positive = np.full(len(frame), seed_score, dtype=float)
            return np.column_stack([1.0 - positive, positive])

    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=FakeXGBClassifier))

    score, oof, test_pred = lab._playground_advanced_xgboost_result(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        folds=3,
    )

    expected = np.mean([0.11, 0.42, 0.99])
    assert score == 0.5
    assert np.allclose(oof, expected)
    assert np.allclose(test_pred, expected)


def test_spaceship_build_features_adds_group_domain_columns_and_enforces_cryo():
    train = pd.DataFrame(
        [
            {
                "PassengerId": "0001_01",
                "HomePlanet": "Europa",
                "CryoSleep": True,
                "Cabin": "B/0/P",
                "Destination": "TRAPPIST-1e",
                "Age": 25.0,
                "VIP": False,
                "RoomService": 0.0,
                "FoodCourt": 0.0,
                "ShoppingMall": 0.0,
                "Spa": 0.0,
                "VRDeck": 0.0,
                "Name": "Ada Stone",
                "Transported": True,
            },
            {
                "PassengerId": "0002_01",
                "HomePlanet": "Earth",
                "CryoSleep": False,
                "Cabin": "F/10/S",
                "Destination": "55 Cancri e",
                "Age": 35.0,
                "VIP": False,
                "RoomService": 10.0,
                "FoodCourt": 20.0,
                "ShoppingMall": 0.0,
                "Spa": 5.0,
                "VRDeck": 0.0,
                "Name": "Bob River",
                "Transported": False,
            },
        ]
    )
    test = pd.DataFrame(
        [
            {
                "PassengerId": "0001_02",
                "HomePlanet": np.nan,
                "CryoSleep": np.nan,
                "Cabin": np.nan,
                "Destination": np.nan,
                "Age": np.nan,
                "VIP": np.nan,
                "RoomService": np.nan,
                "FoodCourt": np.nan,
                "ShoppingMall": np.nan,
                "Spa": np.nan,
                "VRDeck": np.nan,
                "Name": "Eve Stone",
            },
            {
                "PassengerId": "0003_01",
                "HomePlanet": "Mars",
                "CryoSleep": np.nan,
                "Cabin": "G/20/S",
                "Destination": "PSO J318.5-22",
                "Age": 14.0,
                "VIP": np.nan,
                "RoomService": 15.0,
                "FoodCourt": np.nan,
                "ShoppingMall": np.nan,
                "Spa": np.nan,
                "VRDeck": np.nan,
                "Name": "Tom Vale",
            },
        ]
    )

    train_x, test_x = lab._build_spaceship_features(train, test)

    for col in [
        "AgeGroup",
        "HomeDest",
        "DeckSide",
        "CabinNumBin",
        "GroupSpendMean",
        "SurnameCryoRate",
        "CryoSpendMismatch",
    ]:
        assert col in train_x.columns
        assert col in test_x.columns
    assert test_x.loc[0, "CryoSleep"] == 1
    assert test_x.loc[0, "HomePlanet"] == "Europa"
    assert test_x.loc[0, "GroupSize"] == 2
    assert test_x.loc[0, "NoSpend"] == 1
    assert test_x.loc[1, "CryoSleep"] == 0
    assert test_x.loc[1, "NoSpend"] == 0


def test_spaceship_best_threshold_can_outperform_default_threshold():
    probabilities = np.array([0.40, 0.45, 0.55, 0.60])
    y_true = np.array([0, 1, 1, 1])

    threshold, score = lab._spaceship_best_threshold(probabilities, y_true)

    assert threshold != 0.5
    assert score == 1.0


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


def test_store_sales_build_future_frame_uses_direct_lag_when_available():
    history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "store_nbr": 1, "family": "A", "onpromotion": 0, "sales": 10.0},
            {"date": pd.Timestamp("2024-01-02"), "store_nbr": 1, "family": "A", "onpromotion": 0, "sales": 11.0},
            {"date": pd.Timestamp("2024-01-08"), "store_nbr": 1, "family": "A", "onpromotion": 1, "sales": 12.0},
            {"date": pd.Timestamp("2024-01-09"), "store_nbr": 1, "family": "A", "onpromotion": 1, "sales": 13.0},
        ]
    )
    stores = pd.DataFrame([{"store_nbr": 1, "type": "D", "cluster": 3}])
    oil = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "dcoilwtico": 50.0},
            {"date": pd.Timestamp("2024-01-15"), "dcoilwtico": 51.0},
        ]
    )
    holidays = pd.DataFrame([{"date": pd.Timestamp("2024-01-15"), "locale": "National"}])

    history_features = lab._store_sales_make_features(history, oil, stores, holidays)
    category_maps = {
        col: {value: idx for idx, value in enumerate(sorted(pd.Index(history_features[col].astype(str)).drop_duplicates()))}
        for col in ("family", "type")
    }
    lag_lookup, history_summary, family_dow_history, store_dow_history = lab._store_sales_history_artifacts(history)
    target = pd.DataFrame(
        [
            {"id": 1, "date": pd.Timestamp("2024-01-15"), "store_nbr": 1, "family": "A", "onpromotion": 1},
        ]
    )

    future = lab._store_sales_build_future_frame(
        target,
        oil,
        stores,
        holidays,
        lag_lookup,
        history_summary,
        family_dow_history,
        store_dow_history,
        category_maps,
    )

    assert future.loc[0, "lag_7"] == 12.0
    assert future.loc[0, "is_holiday"] == 1


def test_store_sales_recursive_predictions_feed_prior_outputs_into_history(monkeypatch):
    class FakeModel:
        def predict(self, frame):
            return np.log1p(frame["signal"] + 1).to_numpy()

    history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "store_nbr": 1, "family": "A", "onpromotion": 0, "sales": 5.0},
        ]
    )
    target = pd.DataFrame(
        [
            {"id": 1, "date": pd.Timestamp("2024-01-02"), "store_nbr": 1, "family": "A", "onpromotion": 0},
            {"id": 2, "date": pd.Timestamp("2024-01-03"), "store_nbr": 1, "family": "A", "onpromotion": 0},
        ]
    )

    def fake_history_artifacts(current_history):
        return current_history[["sales"]].copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def fake_future_frame(day_rows, *_args, **_kwargs):
        lag_lookup = _args[3]
        return pd.DataFrame({"signal": np.repeat(float(lag_lookup["sales"].iloc[-1]), len(day_rows))})

    monkeypatch.setattr(lab, "_store_sales_history_artifacts", fake_history_artifacts)
    monkeypatch.setattr(lab, "_store_sales_build_future_frame", fake_future_frame)

    preds = lab._store_sales_recursive_predictions(
        FakeModel(),
        history,
        target,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        ["signal"],
    )

    assert np.allclose(preds, [6.0, 7.0])


def test_benchmark_store_sales_prefers_lightgbm_future_when_it_wins(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    train = pd.DataFrame(
        {
            "id": range(1, 21),
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "onpromotion": [idx % 2 for idx in range(20)],
            "sales": np.linspace(10.0, 29.0, 20),
        }
    )
    test = pd.DataFrame(
        [
            {"id": 101, "date": pd.Timestamp("2024-01-21"), "store_nbr": 1, "family": "A", "onpromotion": 0},
            {"id": 102, "date": pd.Timestamp("2024-01-22"), "store_nbr": 1, "family": "A", "onpromotion": 1},
        ]
    )
    stores = pd.DataFrame([{"store_nbr": 1, "type": "D", "cluster": 3}])
    oil = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "dcoilwtico": 50.0},
            {"date": pd.Timestamp("2024-01-22"), "dcoilwtico": 52.0},
        ]
    )
    holidays = pd.DataFrame([{"date": pd.Timestamp("2024-01-21"), "locale": "National"}])

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    stores.to_csv(tmp_path / "stores.csv", index=False)
    oil.to_csv(tmp_path / "oil.csv", index=False)
    holidays.to_csv(tmp_path / "holidays_events.csv", index=False)

    def fake_lightgbm_result(history, validation, future_test, stores_df, oil_df, holidays_df):
        assert not history.empty
        assert not validation.empty
        assert len(future_test) == 2
        return 0.12345, np.full(len(validation), 17.0), np.array([42.0, 43.0])

    monkeypatch.setattr(lab, "_store_sales_lightgbm_future_result", fake_lightgbm_result)

    result = lab.benchmark_store_sales(tmp_path, _folds=0, write_submission=True)

    assert result.best_model == "lightgbm_future"
    assert any(row["model"] == "lightgbm_future" and row["score"] == 0.12345 for row in result.benchmark_rows)

    submission = pd.read_csv(result.submission_path)
    assert submission["sales"].tolist() == [42.0, 43.0]
