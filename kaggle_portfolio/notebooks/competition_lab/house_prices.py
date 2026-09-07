"""House prices: tabular regression.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
)


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

RANDOM_STATE = 42
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


def _house_prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.drop(columns=["SalePrice"]), test], axis=0, ignore_index=True
    )

    def _col(name: str, default: float = 0.0) -> pd.Series:
        if name in combined:
            return combined[name].fillna(default)
        return pd.Series(default, index=combined.index, dtype=float)

    none_fill_cols = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MasVnrType",
    ]
    zero_fill_cols = [
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "GarageCars",
        "GarageArea",
        "GarageYrBlt",
    ]
    mode_fill_cols = [
        "MSZoning",
        "KitchenQual",
        "Electrical",
        "Exterior1st",
        "Exterior2nd",
        "SaleType",
        "Utilities",
    ]

    for col in none_fill_cols:
        if col in combined:
            combined[col] = combined[col].fillna("None")
    for col in zero_fill_cols:
        if col in combined:
            combined[col] = combined[col].fillna(0)
    for col in mode_fill_cols:
        if col in combined and combined[col].notna().any():
            combined[col] = combined[col].fillna(combined[col].mode().iloc[0])
    if "LotFrontage" in combined:
        combined["LotFrontage"] = combined.groupby("Neighborhood")[
            "LotFrontage"
        ].transform(lambda s: s.fillna(s.median()))
        combined["LotFrontage"] = combined["LotFrontage"].fillna(
            combined["LotFrontage"].median()
        )

    combined["MSSubClass"] = combined["MSSubClass"].astype(str)
    yr_sold_num = pd.to_numeric(combined["YrSold"], errors="coerce").fillna(0)
    combined["TotalSF"] = _col("TotalBsmtSF") + _col("1stFlrSF") + _col("2ndFlrSF")
    combined["TotalBath"] = (
        _col("FullBath")
        + 0.5 * _col("HalfBath")
        + _col("BsmtFullBath")
        + 0.5 * _col("BsmtHalfBath")
    )
    combined["HouseAge"] = yr_sold_num - _col("YearBuilt")
    combined["RemodAge"] = yr_sold_num - _col("YearRemodAdd")
    combined["HasGarage"] = _col("GarageArea").gt(0).astype(int)
    combined["HasBsmt"] = _col("TotalBsmtSF").gt(0).astype(int)
    combined["HasPool"] = _col("PoolArea").gt(0).astype(int)
    combined["HasFireplace"] = _col("Fireplaces").gt(0).astype(int)
    combined["HasSecondFloor"] = _col("2ndFlrSF").gt(0).astype(int)
    combined["TotalPorchSF"] = (
        _col("WoodDeckSF")
        + _col("OpenPorchSF")
        + _col("EnclosedPorch")
        + _col("3SsnPorch")
        + _col("ScreenPorch")
    )
    combined["TotalOutsideSF"] = (
        _col("LotArea") + combined["TotalPorchSF"] + _col("PoolArea")
    )
    combined["QualSF"] = _col("OverallQual") * _col("GrLivArea")
    combined["TotalHomeQuality"] = _col("OverallQual") + _col("OverallCond")
    combined["OverallGrade"] = _col("OverallQual") * _col("OverallCond")
    combined["TotalRooms"] = _col("TotRmsAbvGrd") + _col("KitchenAbvGr")
    combined["AgeWhenSold"] = yr_sold_num - _col("YearBuilt")
    combined["AgeSinceRemodel"] = yr_sold_num - _col("YearRemodAdd")
    combined["LivLotRatio"] = _col("GrLivArea") / _col("LotArea", 1.0).clip(lower=1)
    combined["BathPerRoom"] = combined["TotalBath"] / combined["TotalRooms"].replace(
        0, 1
    )
    combined["GarageScore"] = _col("GarageCars") * _col("GarageArea")
    combined["BsmtScore"] = _col("BsmtFinSF1") + _col("BsmtFinSF2") + _col("BsmtUnfSF")
    combined["QualBath"] = _col("OverallQual") * combined["TotalBath"]
    combined["QualKitchen"] = _col("OverallQual") * _col("KitchenAbvGr")
    combined["HasRemodel"] = _col("YearRemodAdd").gt(_col("YearBuilt")).astype(int)
    combined["IsNewHouse"] = _col("YearBuilt").ge(yr_sold_num - 1).astype(int)
    for cat_col in ("YrSold", "MoSold"):
        if cat_col in combined:
            combined[cat_col] = combined[cat_col].astype(str)

    for col in (
        "LotArea",
        "GrLivArea",
        "TotalSF",
        "1stFlrSF",
        "2ndFlrSF",
        "MasVnrArea",
        "TotalBsmtSF",
        "TotalOutsideSF",
        "GarageScore",
        "BsmtScore",
    ):
        if col in combined:
            combined[f"log_{col.lower()}"] = np.log1p(combined[col].clip(lower=0))

    train_x = combined.iloc[: len(train)].reset_index(drop=True)
    test_x = combined.iloc[len(train) :].reset_index(drop=True)
    return train_x, test_x


def _house_rmse(y_true: Any, y_pred: Any) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def _house_blend_candidates(names: list[str]) -> list[tuple[str, ...]]:
    return [
        combo
        for size in range(2, min(4, len(names)) + 1)
        for combo in combinations(names, size)
    ]


def _house_weight_options(size: int, step: float = 0.05) -> list[tuple[float, ...]]:
    unit = int(round(1.0 / step))
    weights: list[tuple[float, ...]] = []

    def _build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(
                tuple(
                    (prefix + [remaining])[idx] * step for idx in range(len(prefix) + 1)
                )
            )
            return
        for value in range(1, remaining - slots + 2):
            _build(prefix + [value], remaining - value, slots - 1)

    _build([], unit, size)
    return weights


def _house_best_blend(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    y_true: pd.Series | np.ndarray,
) -> tuple[dict[str, float], float, np.ndarray] | None:
    if len(predictions) < 2:
        return None

    names = list(predictions)
    best_weights: dict[str, float] | None = None
    best_score = float("inf")
    best_pred: np.ndarray | None = None
    target = np.asarray(y_true, dtype=float)

    for combo in _house_blend_candidates(names):
        oof_stack = np.column_stack([predictions[name][0] for name in combo])
        test_stack = np.column_stack([predictions[name][1] for name in combo])
        for weights in _house_weight_options(len(combo)):
            weight_arr = np.asarray(weights, dtype=float)
            blended_oof = oof_stack @ weight_arr
            score = _house_rmse(target, blended_oof)
            if score < best_score:
                best_score = score
                best_weights = {
                    name: float(weight) for name, weight in zip(combo, weights)
                }
                best_pred = test_stack @ weight_arr

    if best_weights is None or best_pred is None:
        return None
    return best_weights, best_score, best_pred


def benchmark_house_prices(
    data_dir: Path, folds: int, write_submission: bool
) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    if outlier_mask.any():
        train = train.loc[~outlier_mask].reset_index(drop=True)

    y = np.log1p(train["SalePrice"])
    train_x, test_x = _house_prepare_features(train, test)
    dense_train = pd.get_dummies(train_x, dummy_na=True).apply(
        pd.to_numeric, errors="coerce"
    )
    dense_test = pd.get_dummies(test_x, dummy_na=True).apply(
        pd.to_numeric, errors="coerce"
    )
    dense_test = dense_test.reindex(columns=dense_train.columns, fill_value=0)
    medians = dense_train.median()
    dense_train = dense_train.fillna(medians)
    dense_test = dense_test.fillna(medians)
    skew_candidates = [
        col
        for col in dense_train.columns
        if dense_train[col].dtype.kind in "fiu" and dense_train[col].nunique() > 10
    ]
    skewness = dense_train[skew_candidates].apply(pd.Series.skew).abs()
    for col in skewness[skewness > 0.75].index:
        offset = 0.0
        col_min = min(float(dense_train[col].min()), float(dense_test[col].min()))
        if col_min <= -1.0:
            offset = abs(col_min) + 1.0
        dense_train[col] = np.log1p(dense_train[col] + offset)
        dense_test[col] = np.log1p(dense_test[col] + offset)

    cv = KFold(n_splits=min(max(3, folds), 5), shuffle=True, random_state=RANDOM_STATE)
    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}
    blend_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _fit_house_model(name: str, model: Any) -> None:
        oof = cross_val_predict(model, dense_train, y, cv=cv, n_jobs=1)
        score = _house_rmse(y, oof)
        benchmarks.append({"model": name, "score": round(score, 5)})
        model.fit(dense_train, y)
        test_pred_log = model.predict(dense_test)
        trained_predictions[name] = np.expm1(test_pred_log).clip(min=0)
        blend_inputs[name] = (oof, test_pred_log)

    lasso = Pipeline(
        [
            ("scale", RobustScaler()),
            ("model", Lasso(alpha=0.0005, max_iter=50000)),
        ]
    )
    _fit_house_model("lasso", lasso)

    elastic = Pipeline(
        [
            ("scale", RobustScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.0005,
                    l1_ratio=0.9,
                    max_iter=50000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    _fit_house_model("elasticnet", elastic)

    kernel_ridge = Pipeline(
        [
            ("scale", RobustScaler()),
            ("model", KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)),
        ]
    )
    _fit_house_model("kernel_ridge", kernel_ridge)

    gbr = GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=15,
        min_samples_split=10,
        loss="huber",
        random_state=RANDOM_STATE,
    )
    _fit_house_model("gradient_boosting", gbr)

    try:
        import lightgbm as lgb

        lgb_model = lgb.LGBMRegressor(
            n_estimators=2500,
            learning_rate=0.01,
            num_leaves=20,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.002,
            reg_lambda=0.4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        _fit_house_model("lightgbm", lgb_model)
    except ImportError:
        pass

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=1.0,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.0005,
            reg_lambda=1.0,
            gamma=0.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        _fit_house_model("xgboost", xgb_model)
    except ImportError:
        pass

    blend_result = _house_best_blend(blend_inputs, y)
    if blend_result is not None:
        blend_weights, blend_rmse, blend_pred = blend_result
        benchmarks.append(
            {
                "model": "blend",
                "score": round(float(blend_rmse), 5),
                "weights": {
                    name: round(weight, 2) for name, weight in blend_weights.items()
                },
            }
        )
        trained_predictions["blend"] = np.expm1(blend_pred).clip(min=0)

    best = min(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir(
            "house-prices-advanced-regression-techniques"
        ) / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame(
            {"Id": test["Id"], "SalePrice": trained_predictions[best["model"]]}
        ).to_csv(
            submission_path,
            index=False,
        )

    return LabResult(
        competition="house-prices-advanced-regression-techniques",
        metric_name="rmse",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
