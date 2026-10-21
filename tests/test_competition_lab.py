from __future__ import annotations


import numpy as np
import pandas as pd

from kaggle_portfolio.notebooks import competition_lab as lab
from kaggle_portfolio.notebooks.competition_lab import (
    house_prices,
    spaceship,
    store_sales,
)


def test_benchmarks_include_new_entered_competitions():
    assert "house-prices-advanced-regression-techniques" in lab.BENCHMARKS
    assert "store-sales-time-series-forecasting" in lab.BENCHMARKS


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

    train_x, test_x = spaceship._build_spaceship_features(train, test)

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

    threshold, score = spaceship._spaceship_best_threshold(probabilities, y_true)

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
                "GarageCars": 2,
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

    train_x, _test_x = house_prices._house_prepare_features(train, test)

    assert train_x.loc[0, "TotalSF"] == 2300
    assert train_x.loc[0, "TotalBath"] == 3.5
    assert train_x.loc[0, "QualSF"] == 9800
    assert train_x.loc[0, "TotalPorchSF"] == 60
    assert train_x.loc[0, "GarageScore"] == 1000
    assert train_x.loc[0, "OverallGrade"] == 35


def test_house_best_blend_prefers_stronger_weighted_mix():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = {
        "a": (
            np.array([1.1, 1.9, 3.2, 3.8]),
            np.array([1.5, 2.5]),
        ),
        "b": (
            np.array([0.9, 2.2, 2.8, 4.1]),
            np.array([1.4, 2.6]),
        ),
        "c": (
            np.array([1.4, 2.4, 3.4, 4.4]),
            np.array([1.6, 2.8]),
        ),
    }

    result = house_prices._house_best_blend(predictions, y)

    assert result is not None
    weights, score, test_pred = result
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert score < min(
        house_prices._house_rmse(y, pred[0]) for pred in predictions.values()
    )
    assert test_pred.shape == (2,)


def test_store_sales_prediction_frame_produces_complete_predictions():
    history = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": 10.0,
            },
            {
                "date": "2024-01-08",
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": 12.0,
            },
            {
                "date": "2024-01-15",
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
                "sales": 15.0,
            },
            {
                "date": "2024-01-22",
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
                "sales": 16.0,
            },
        ]
    )
    target = pd.DataFrame(
        [
            {"date": "2024-01-29", "store_nbr": 1, "family": "A", "onpromotion": 1},
            {"date": "2024-01-30", "store_nbr": 1, "family": "A", "onpromotion": 0},
        ]
    )

    frame = store_sales._store_sales_prediction_frame(history, target)

    assert frame["recent_dow_promo_mean"].notna().all()
    assert frame["recent_28_mean"].notna().all()
    assert frame["hybrid_mean"].notna().all()


def test_store_sales_build_future_frame_uses_direct_lag_when_available():
    history = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": 10.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": 11.0,
            },
            {
                "date": pd.Timestamp("2024-01-08"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
                "sales": 12.0,
            },
            {
                "date": pd.Timestamp("2024-01-09"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
                "sales": 13.0,
            },
        ]
    )
    stores = pd.DataFrame([{"store_nbr": 1, "type": "D", "cluster": 3}])
    oil = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "dcoilwtico": 50.0},
            {"date": pd.Timestamp("2024-01-15"), "dcoilwtico": 51.0},
        ]
    )
    holidays = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-15"), "locale": "National"}]
    )

    history_features = store_sales._store_sales_make_features(
        history, oil, stores, holidays
    )
    category_maps = {
        col: {
            value: idx
            for idx, value in enumerate(
                sorted(pd.Index(history_features[col].astype(str)).drop_duplicates())
            )
        }
        for col in ("family", "type")
    }
    lag_lookup, history_summary, family_dow_history, store_dow_history = (
        store_sales._store_sales_history_artifacts(history)
    )
    target = pd.DataFrame(
        [
            {
                "id": 1,
                "date": pd.Timestamp("2024-01-15"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
            },
        ]
    )

    future = store_sales._store_sales_build_future_frame(
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
            {
                "date": pd.Timestamp("2024-01-01"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": 5.0,
            },
        ]
    )
    target = pd.DataFrame(
        [
            {
                "id": 1,
                "date": pd.Timestamp("2024-01-02"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
            },
            {
                "id": 2,
                "date": pd.Timestamp("2024-01-03"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
            },
        ]
    )

    def fake_history_artifacts(current_history):
        return (
            current_history[["sales"]].copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    def fake_future_frame(day_rows, *_args, **_kwargs):
        lag_lookup = _args[3]
        return pd.DataFrame(
            {"signal": np.repeat(float(lag_lookup["sales"].iloc[-1]), len(day_rows))}
        )

    monkeypatch.setattr(
        store_sales, "_store_sales_history_artifacts", fake_history_artifacts
    )
    monkeypatch.setattr(
        store_sales, "_store_sales_build_future_frame", fake_future_frame
    )

    preds = store_sales._store_sales_recursive_predictions(
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


def test_benchmark_store_sales_prefers_lightgbm_future_when_it_wins(
    tmp_path, monkeypatch
):
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
            {
                "id": 101,
                "date": pd.Timestamp("2024-01-21"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
            },
            {
                "id": 102,
                "date": pd.Timestamp("2024-01-22"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
            },
        ]
    )
    stores = pd.DataFrame([{"store_nbr": 1, "type": "D", "cluster": 3}])
    oil = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "dcoilwtico": 50.0},
            {"date": pd.Timestamp("2024-01-22"), "dcoilwtico": 52.0},
        ]
    )
    holidays = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-21"), "locale": "National"}]
    )

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    stores.to_csv(tmp_path / "stores.csv", index=False)
    oil.to_csv(tmp_path / "oil.csv", index=False)
    holidays.to_csv(tmp_path / "holidays_events.csv", index=False)

    def fake_lightgbm_result(
        history, validation, future_test, stores_df, oil_df, holidays_df
    ):
        assert not history.empty
        assert not validation.empty
        assert len(future_test) == 2
        return 0.12345, np.full(len(validation), 17.0), np.array([42.0, 43.0])

    models = store_sales.StoreSalesModels(
        lightgbm_future_result=fake_lightgbm_result,
    )

    result = store_sales.benchmark_store_sales(
        tmp_path, _folds=0, write_submission=True, models=models
    )

    assert result.best_model == "lightgbm_future"
    assert any(
        row["model"] == "lightgbm_future" and row["score"] == 0.12345
        for row in result.benchmark_rows
    )

    submission = pd.read_csv(result.submission_path)
    assert submission["sales"].tolist() == [42.0, 43.0]


def test_store_sales_lightgbm_future_result_does_not_build_a_redundant_future_frame(
    monkeypatch,
):
    """Regression test for a refactor leftover.

    ``_store_sales_lightgbm_future_result`` used to build a future frame for the
    whole test set into ``_submission_future`` and then discard it without ever
    using it — the real submission predictions come from
    ``_store_sales_recursive_predictions`` instead, which builds its own future
    frame one day at a time. That dead build wasted a full future-frame
    construction over the entire test set on every run. This pins the exact
    number of ``_store_sales_build_future_frame`` calls the function performs
    (validation_future once, plus once per distinct date recursed over for the
    validation and test sets) so the leftover can't silently come back.
    """
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    history = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "onpromotion": [idx % 2 for idx in range(30)],
            "sales": np.linspace(10.0, 39.0, 30),
        }
    )
    validation = pd.DataFrame(
        [
            {
                "id": 101,
                "date": pd.Timestamp("2024-01-31"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": 40.0,
            },
            {
                "id": 102,
                "date": pd.Timestamp("2024-02-01"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
                "sales": 41.0,
            },
        ]
    )
    test = pd.DataFrame(
        [
            {
                "id": 201,
                "date": pd.Timestamp("2024-02-02"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
            },
            {
                "id": 202,
                "date": pd.Timestamp("2024-02-03"),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 1,
            },
        ]
    )
    stores = pd.DataFrame([{"store_nbr": 1, "type": "D", "cluster": 3}])
    oil = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "dcoilwtico": 50.0},
            {"date": pd.Timestamp("2024-02-03"), "dcoilwtico": 52.0},
        ]
    )
    holidays = pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-15"), "locale": "National"}]
    )

    class FakeLGBMRegressor:
        def __init__(self, **_kwargs):
            self._mean = 0.0

        def fit(self, _x_train, y_train, eval_set=None, callbacks=None):
            self._mean = float(np.mean(y_train))
            return self

        def predict(self, frame):
            return np.full(len(frame), self._mean)

    fake_lightgbm = types.SimpleNamespace(
        LGBMRegressor=FakeLGBMRegressor,
        early_stopping=lambda *_args, **_kwargs: None,
        log_evaluation=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lightgbm)

    real_build_future_frame = store_sales._store_sales_build_future_frame
    call_count = {"n": 0}

    def counting_build_future_frame(*args, **kwargs):
        call_count["n"] += 1
        return real_build_future_frame(*args, **kwargs)

    monkeypatch.setattr(
        store_sales, "_store_sales_build_future_frame", counting_build_future_frame
    )

    score, validation_pred, submission_pred = (
        store_sales._store_sales_lightgbm_future_result(
            history, validation, test, stores, oil, holidays
        )
    )

    assert np.isfinite(score)
    assert len(validation_pred) == 2
    assert len(submission_pred) == 2
    # validation_future (1) + one recursive build per distinct validation date (2)
    # + one recursive build per distinct test date (2). No extra, discarded
    # build for the whole test set.
    assert call_count["n"] == 5
