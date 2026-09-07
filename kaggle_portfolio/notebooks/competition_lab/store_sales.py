"""Store sales: time-series forecasting.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kaggle_portfolio.shared.layout import RepoLayout

from kaggle_portfolio.notebooks.competition_lab.runner import (  # noqa: F401
    LabResult,
    _benchmark_dir,
    _concat_feature_block,
    _ensure_data,
    _print_benchmarks,
    _safe_slug,
    _save_summary,
    _submission_dir,
    _submit,
)

ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = RepoLayout.resolve().lab_root
RANDOM_STATE = 42
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


def _store_sales_rmsle(y_true: Any, y_pred: Any) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_arr) - np.log1p(y_true_arr)) ** 2)))


def _store_sales_prediction_frame(
    history: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    history = history.copy()
    target = target.copy()
    history["date"] = pd.to_datetime(history["date"])
    target["date"] = pd.to_datetime(target["date"])
    history["dow"] = history["date"].dt.dayofweek
    target["dow"] = target["date"].dt.dayofweek

    recent_140 = history.loc[
        history["date"] >= history["date"].max() - pd.Timedelta(days=140)
    ]
    recent_56 = history.loc[
        history["date"] >= history["date"].max() - pd.Timedelta(days=56)
    ]
    recent_28 = history.loc[
        history["date"] >= history["date"].max() - pd.Timedelta(days=28)
    ]

    group_sf_dow_promo = (
        recent_140.groupby(["store_nbr", "family", "dow", "onpromotion"])["sales"]
        .mean()
        .rename("pred_sf_dow_promo")
        .reset_index()
    )
    group_sf_dow = (
        recent_140.groupby(["store_nbr", "family", "dow"])["sales"]
        .mean()
        .rename("pred_sf_dow")
        .reset_index()
    )
    group_sf_28 = (
        recent_28.groupby(["store_nbr", "family"])["sales"]
        .mean()
        .rename("pred_sf_28")
        .reset_index()
    )
    group_sf_56 = (
        recent_56.groupby(["store_nbr", "family"])["sales"]
        .mean()
        .rename("pred_sf_56")
        .reset_index()
    )
    group_family_dow = (
        recent_140.groupby(["family", "dow"])["sales"]
        .mean()
        .rename("pred_family_dow")
        .reset_index()
    )
    group_store_dow = (
        recent_140.groupby(["store_nbr", "dow"])["sales"]
        .mean()
        .rename("pred_store_dow")
        .reset_index()
    )
    global_mean = float(history["sales"].mean())

    frame = (
        target.merge(
            group_sf_dow_promo,
            on=["store_nbr", "family", "dow", "onpromotion"],
            how="left",
        )
        .merge(group_sf_dow, on=["store_nbr", "family", "dow"], how="left")
        .merge(group_sf_28, on=["store_nbr", "family"], how="left")
        .merge(group_sf_56, on=["store_nbr", "family"], how="left")
        .merge(group_family_dow, on=["family", "dow"], how="left")
        .merge(group_store_dow, on=["store_nbr", "dow"], how="left")
    )
    frame["recent_dow_promo_mean"] = (
        frame["pred_sf_dow_promo"]
        .fillna(frame["pred_sf_dow"])
        .fillna(frame["pred_sf_28"])
        .fillna(frame["pred_sf_56"])
        .fillna(frame["pred_family_dow"])
        .fillna(frame["pred_store_dow"])
        .fillna(global_mean)
    )
    frame["recent_28_mean"] = (
        frame["pred_sf_28"]
        .fillna(frame["pred_sf_56"])
        .fillna(frame["pred_sf_dow"])
        .fillna(frame["pred_family_dow"])
        .fillna(frame["pred_store_dow"])
        .fillna(global_mean)
    )
    frame["hybrid_mean"] = (
        0.65 * frame["recent_dow_promo_mean"] + 0.35 * frame["recent_28_mean"]
    )
    return frame


def _store_sales_make_features(
    df: pd.DataFrame,
    oil_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy().sort_values(["store_nbr", "family", "date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    oil_filled = (
        oil_df.set_index("date")["dcoilwtico"].resample("D").interpolate("linear")
    )
    df["oil_price"] = df["date"].map(oil_filled).ffill().fillna(50.0)

    national_holidays = holidays_df.loc[
        holidays_df["locale"] == "National", "date"
    ].drop_duplicates()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    df = df.merge(
        stores_df[["store_nbr", "type", "cluster"]], on="store_nbr", how="left"
    )

    if "sales" in df.columns:
        grouped_sales = df.groupby(["store_nbr", "family"])["sales"]
        grouped_promo = df.groupby(["store_nbr", "family"])["onpromotion"]
        for lag in (7, 14, 28):
            df[f"lag_{lag}"] = grouped_sales.shift(lag)
        for window in (7, 14, 28):
            df[f"roll_mean_{window}"] = grouped_sales.transform(
                lambda s: s.shift(1).rolling(window).mean()
            )
            df[f"roll_std_{window}"] = grouped_sales.transform(
                lambda s: s.shift(1).rolling(window).std()
            )
        df["ewma_7"] = grouped_sales.transform(lambda s: s.shift(1).ewm(span=7).mean())
        df["promo_roll_mean_14"] = grouped_promo.transform(
            lambda s: s.shift(1).rolling(14).mean()
        )
        df["promo_roll_mean_28"] = grouped_promo.transform(
            lambda s: s.shift(1).rolling(28).mean()
        )
        df["history_mean"] = grouped_sales.transform(
            lambda s: s.shift(1).expanding().mean()
        )
        df["trend_7_28"] = df["roll_mean_7"] / (df["roll_mean_28"] + 1)
        df["sales_momentum"] = df["roll_mean_7"] - df["roll_mean_28"]

    df["oil_to_trend"] = df["oil_price"] / (
        df.get("roll_mean_28", pd.Series(0, index=df.index)).fillna(0) + 1
    )
    df["promo_x_trend"] = df["onpromotion"] * df.get(
        "trend_7_28", pd.Series(1.0, index=df.index)
    ).fillna(1.0)
    return df


def _store_sales_history_artifacts(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = history.sort_values(["store_nbr", "family", "date"]).copy()
    lag_lookup = history[["store_nbr", "family", "date", "sales"]].copy()
    history_summary = (
        history.groupby(["store_nbr", "family"])[["sales", "onpromotion"]]
        .apply(
            lambda g: pd.Series(
                {
                    "lag_7_fill": g["sales"].shift(7).dropna().iloc[-1]
                    if g["sales"].shift(7).notna().any()
                    else g["sales"].tail(7).mean(),
                    "lag_14_fill": g["sales"].shift(14).dropna().iloc[-1]
                    if g["sales"].shift(14).notna().any()
                    else g["sales"].tail(14).mean(),
                    "lag_28_fill": g["sales"].shift(28).dropna().iloc[-1]
                    if g["sales"].shift(28).notna().any()
                    else g["sales"].tail(28).mean(),
                    "roll_mean_7_fill": g["sales"].tail(7).mean(),
                    "roll_mean_14_fill": g["sales"].tail(14).mean(),
                    "roll_mean_28_fill": g["sales"].tail(28).mean(),
                    "roll_std_7_fill": g["sales"].tail(7).std(),
                    "roll_std_14_fill": g["sales"].tail(14).std(),
                    "roll_std_28_fill": g["sales"].tail(28).std(),
                    "ewma_7_fill": g["sales"].ewm(span=7).mean().iloc[-1],
                    "promo_roll_mean_14_fill": g["onpromotion"].tail(14).mean(),
                    "promo_roll_mean_28_fill": g["onpromotion"].tail(28).mean(),
                    "history_mean_fill": g["sales"].mean(),
                    "trend_7_28_fill": g["sales"].tail(7).mean()
                    / (g["sales"].tail(28).mean() + 1),
                    "sales_momentum_fill": g["sales"].tail(7).mean()
                    - g["sales"].tail(28).mean(),
                }
            )
        )
        .reset_index()
    )
    family_dow_history = (
        history.assign(dayofweek=history["date"].dt.dayofweek)
        .groupby(["family", "dayofweek"])["sales"]
        .mean()
        .rename("family_dow_mean")
        .reset_index()
    )
    store_dow_history = (
        history.assign(dayofweek=history["date"].dt.dayofweek)
        .groupby(["store_nbr", "dayofweek"])["sales"]
        .mean()
        .rename("store_dow_mean")
        .reset_index()
    )
    return lag_lookup, history_summary, family_dow_history, store_dow_history


def _store_sales_build_future_frame(
    target: pd.DataFrame,
    oil_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    lag_lookup: pd.DataFrame,
    history_summary: pd.DataFrame,
    family_dow_history: pd.DataFrame,
    store_dow_history: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    ordered_target = target.copy()
    ordered_target["_row_order"] = np.arange(len(ordered_target))
    future = _store_sales_make_features(ordered_target, oil_df, stores_df, holidays_df)
    future = future.merge(history_summary, on=["store_nbr", "family"], how="left")
    future = future.merge(family_dow_history, on=["family", "dayofweek"], how="left")
    future = future.merge(store_dow_history, on=["store_nbr", "dayofweek"], how="left")

    for lag in (7, 14, 28):
        lagged = lag_lookup.rename(columns={"sales": f"lag_{lag}_direct"}).copy()
        lagged["forecast_date"] = lagged["date"] + pd.Timedelta(days=lag)
        future = future.merge(
            lagged[["store_nbr", "family", "forecast_date", f"lag_{lag}_direct"]],
            left_on=["store_nbr", "family", "date"],
            right_on=["store_nbr", "family", "forecast_date"],
            how="left",
        ).drop(columns=["forecast_date"])
        future[f"lag_{lag}"] = future[f"lag_{lag}_direct"].fillna(
            future[f"lag_{lag}_fill"]
        )

    fill_map = {
        "roll_mean_7": "roll_mean_7_fill",
        "roll_mean_14": "roll_mean_14_fill",
        "roll_mean_28": "roll_mean_28_fill",
        "roll_std_7": "roll_std_7_fill",
        "roll_std_14": "roll_std_14_fill",
        "roll_std_28": "roll_std_28_fill",
        "ewma_7": "ewma_7_fill",
        "promo_roll_mean_14": "promo_roll_mean_14_fill",
        "promo_roll_mean_28": "promo_roll_mean_28_fill",
        "history_mean": "history_mean_fill",
        "trend_7_28": "trend_7_28_fill",
        "sales_momentum": "sales_momentum_fill",
    }
    for feature, fallback in fill_map.items():
        future[feature] = future.get(
            feature, pd.Series(np.nan, index=future.index)
        ).fillna(future[fallback])

    future["oil_to_trend"] = future["oil_price"] / (future["roll_mean_28"] + 1)
    future["promo_x_trend"] = future["onpromotion"] * future["trend_7_28"]
    for col, mapping in category_maps.items():
        future[col] = future[col].astype(str).map(mapping).fillna(-1).astype(int)
    future = future.sort_values("_row_order").drop(columns=["_row_order"])
    return future.fillna(0)


def _store_sales_recursive_predictions(
    model: Any,
    history: pd.DataFrame,
    target: pd.DataFrame,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    feature_cols: list[str],
) -> np.ndarray:
    working_history = history[
        ["date", "store_nbr", "family", "onpromotion", "sales"]
    ].copy()
    ordered_target = target.copy()
    ordered_target["_row_order"] = np.arange(len(ordered_target))
    predictions: list[pd.DataFrame] = []

    for pred_date in sorted(pd.to_datetime(ordered_target["date"]).drop_duplicates()):
        day_rows = ordered_target.loc[ordered_target["date"] == pred_date].copy()
        lag_lookup, history_summary, family_dow_history, store_dow_history = (
            _store_sales_history_artifacts(working_history)
        )
        future_day = _store_sales_build_future_frame(
            day_rows.drop(columns=["_row_order"]),
            oil_df,
            stores_df,
            holidays_df,
            lag_lookup,
            history_summary,
            family_dow_history,
            store_dow_history,
            category_maps,
        )
        day_pred = np.clip(np.expm1(model.predict(future_day[feature_cols])), 0, None)
        predictions.append(
            pd.DataFrame(
                {"_row_order": day_rows["_row_order"].to_numpy(), "pred": day_pred}
            )
        )
        history_extension = day_rows[
            ["date", "store_nbr", "family", "onpromotion"]
        ].copy()
        history_extension["sales"] = day_pred
        working_history = pd.concat(
            [working_history, history_extension], ignore_index=True
        )

    ordered_predictions = pd.concat(predictions, ignore_index=True).sort_values(
        "_row_order"
    )
    return ordered_predictions["pred"].to_numpy(dtype=float)


def _store_sales_lightgbm_future_result(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed") from exc

    history_features = _store_sales_make_features(
        history, oil_df, stores_df, holidays_df
    )
    category_maps: dict[str, dict[str, int]] = {}
    for col in ("family", "type"):
        mapping = {
            value: idx
            for idx, value in enumerate(
                sorted(pd.Index(history_features[col].astype(str)).drop_duplicates())
            )
        }
        category_maps[col] = mapping
        history_features[col] = (
            history_features[col].astype(str).map(mapping).astype(int)
        )
    history_features = history_features.fillna(0)

    lag_lookup, history_summary, family_dow_history, store_dow_history = (
        _store_sales_history_artifacts(history)
    )
    validation_future = _store_sales_build_future_frame(
        validation.drop(columns=["sales"]),
        oil_df,
        stores_df,
        holidays_df,
        lag_lookup,
        history_summary,
        family_dow_history,
        store_dow_history,
        category_maps,
    )
    # FIXME: computed and then discarded before the return — a refactor
    # leftover. Left in place rather than deleted because it is outside the
    # scope of this change and may have been meant to feed the submission.
    _submission_future = _store_sales_build_future_frame(
        test,
        oil_df,
        stores_df,
        holidays_df,
        lag_lookup,
        history_summary,
        family_dow_history,
        store_dow_history,
        category_maps,
    )

    feature_cols = [
        col
        for col in history_features.columns
        if col not in {"id", "date", "sales"}
        and history_features[col].dtype != "object"
    ]
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        history_features[feature_cols],
        np.log1p(history_features["sales"].clip(lower=0)),
        eval_set=[
            (
                validation_future[feature_cols],
                np.log1p(validation["sales"].clip(lower=0)),
            )
        ],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    validation_pred = _store_sales_recursive_predictions(
        model,
        history,
        validation[["id", "date", "store_nbr", "family", "onpromotion"]],
        stores_df,
        oil_df,
        holidays_df,
        category_maps,
        feature_cols,
    )
    submission_pred = _store_sales_recursive_predictions(
        model,
        history,
        test,
        stores_df,
        oil_df,
        holidays_df,
        category_maps,
        feature_cols,
    )
    return (
        _store_sales_rmsle(validation["sales"], validation_pred),
        validation_pred,
        np.clip(submission_pred, 0, None),
    )


def benchmark_store_sales(
    data_dir: Path, _folds: int, write_submission: bool
) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    valid_dates = sorted(train["date"].drop_duplicates())[-16:]
    validation = train.loc[train["date"].isin(valid_dates)].copy()
    history = train.loc[~train["date"].isin(valid_dates)].copy()
    validation_frame = _store_sales_prediction_frame(history, validation)

    benchmarks = [
        {
            "model": "recent_dow_promo_mean",
            "score": round(
                _store_sales_rmsle(
                    validation["sales"], validation_frame["recent_dow_promo_mean"]
                ),
                5,
            ),
        },
        {
            "model": "recent_28_mean",
            "score": round(
                _store_sales_rmsle(
                    validation["sales"], validation_frame["recent_28_mean"]
                ),
                5,
            ),
        },
        {
            "model": "hybrid_mean",
            "score": round(
                _store_sales_rmsle(
                    validation["sales"], validation_frame["hybrid_mean"]
                ),
                5,
            ),
        },
    ]
    learned_predictions: dict[str, np.ndarray] = {}
    stores_path = data_dir / "stores.csv"
    oil_path = data_dir / "oil.csv"
    holidays_path = data_dir / "holidays_events.csv"
    if stores_path.exists() and oil_path.exists() and holidays_path.exists():
        try:
            stores_df = pd.read_csv(stores_path)
            oil_df = pd.read_csv(oil_path, parse_dates=["date"])
            holidays_df = pd.read_csv(holidays_path, parse_dates=["date"])
            future_score, _validation_pred, submission_pred = (
                _store_sales_lightgbm_future_result(
                    history,
                    validation,
                    test,
                    stores_df,
                    oil_df,
                    holidays_df,
                )
            )
            benchmarks.append(
                {"model": "lightgbm_future", "score": round(future_score, 5)}
            )
            learned_predictions["lightgbm_future"] = submission_pred
        except RuntimeError:
            pass
    best = min(benchmarks, key=lambda row: row["score"])

    submission_path = None
    if write_submission:
        submission_frame = _store_sales_prediction_frame(train, test)
        submission_path = _submission_dir("store-sales-time-series-forecasting") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        if best["model"] in learned_predictions:
            sales = learned_predictions[best["model"]]
        else:
            sales = submission_frame[best["model"]].clip(lower=0)
        pd.DataFrame({"id": test["id"], "sales": sales}).to_csv(
            submission_path, index=False
        )

    return LabResult(
        competition="store-sales-time-series-forecasting",
        metric_name="rmsle",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
