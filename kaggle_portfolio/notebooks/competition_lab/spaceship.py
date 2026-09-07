"""Spaceship Titanic: tabular binary classification.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
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


def _build_spaceship_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.drop(columns=["Transported"]), test], axis=0, ignore_index=True
    )

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    group_id = combined["PassengerId"].astype(str).str.split("_").str[0]
    combined["GroupId"] = group_id
    combined["GroupSize"] = group_id.map(group_id.value_counts()).astype(int)
    combined["GroupMemberIdx"] = (
        pd.to_numeric(
            combined["PassengerId"].astype(str).str.split("_").str[-1], errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )
    combined["Surname"] = (
        combined["Name"].fillna("Unknown Unknown").astype(str).str.split().str[-1]
    )
    combined["SurnameSize"] = (
        combined["Surname"].map(combined["Surname"].value_counts()).astype(int)
    )

    cabin = (
        combined["Cabin"].fillna("Unknown/0/U").astype(str).str.split("/", expand=True)
    )
    combined["Deck"] = cabin[0].fillna("Unknown").astype(str)
    combined["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    combined["Side"] = cabin[2].fillna("Unknown").astype(str)
    combined["CabinKnown"] = combined["Cabin"].notna().astype(int)

    def _mode_map(df: pd.DataFrame, key: str, value: str) -> dict[str, object]:
        grouped = (
            df[[key, value]]
            .dropna(subset=[value])
            .groupby(key, observed=False)[value]
            .agg(
                lambda s: s.mode(dropna=True).iloc[0]
                if not s.mode(dropna=True).empty
                else s.iloc[0]
            )
        )
        return grouped.to_dict()

    for col in spend_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["SpendMissingCount"] = combined[spend_cols].isna().sum(axis=1).astype(int)
    raw_total_spend = combined[spend_cols].sum(axis=1, min_count=1)
    combined["NoSpendObserved"] = combined[spend_cols].fillna(0.0).sum(axis=1).eq(0)

    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        combined[col] = combined[col].fillna(
            combined["GroupId"].map(_mode_map(combined, "GroupId", col))
        )
    combined["HomePlanet"] = combined["HomePlanet"].fillna(
        combined["Surname"].map(_mode_map(combined, "Surname", "HomePlanet"))
    )
    combined["Destination"] = combined["Destination"].fillna(
        combined["Surname"].map(_mode_map(combined, "Surname", "Destination"))
    )
    combined["HomePlanet"] = combined["HomePlanet"].fillna(
        combined["Deck"].map(_mode_map(combined, "Deck", "HomePlanet"))
    )
    combined["Destination"] = combined["Destination"].fillna("TRAPPIST-1e")

    deck_cabin_median = combined.groupby("Deck", observed=False)["CabinNum"].median()
    combined["CabinNum"] = combined["CabinNum"].fillna(
        combined["Deck"].map(deck_cabin_median)
    )
    combined["CabinNum"] = (
        combined["CabinNum"].fillna(combined["CabinNum"].median()).astype(float)
    )

    combined["CryoSleep"] = combined["CryoSleep"].where(
        combined["CryoSleep"].notna(), np.nan
    )
    cryo_missing = combined["CryoSleep"].isna()
    combined.loc[cryo_missing & combined["NoSpendObserved"], "CryoSleep"] = True
    cryo_missing = combined["CryoSleep"].isna()
    combined.loc[cryo_missing & (raw_total_spend.fillna(0) > 0), "CryoSleep"] = False
    combined["CryoSleep"] = combined["CryoSleep"].where(
        combined["CryoSleep"].notna(),
        combined["GroupId"].map(_mode_map(combined, "GroupId", "CryoSleep")),
    )
    combined["CryoSleep"] = combined["CryoSleep"].where(
        combined["CryoSleep"].notna(), False
    )

    for col in spend_cols:
        group_median = combined.groupby(["HomePlanet", "Deck"], observed=False)[
            col
        ].transform("median")
        combined[col] = combined[col].fillna(group_median)
        combined[col] = combined[col].fillna(combined[col].median())
    combined.loc[combined["CryoSleep"].astype(bool), spend_cols] = 0.0

    combined["Age"] = combined["Age"].fillna(
        combined.groupby("GroupId", observed=False)["Age"].transform("median")
    )
    combined["Age"] = combined["Age"].fillna(
        combined.groupby(["HomePlanet", "Deck"], observed=False)["Age"].transform(
            "median"
        )
    )
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())
    combined["VIP"] = combined["VIP"].where(
        combined["VIP"].notna(),
        combined["GroupId"].map(_mode_map(combined, "GroupId", "VIP")),
    )
    combined["VIP"] = combined["VIP"].where(combined["VIP"].notna(), False)

    combined["HomePlanet"] = combined["HomePlanet"].fillna("Unknown").astype(str)
    combined["Destination"] = combined["Destination"].fillna("Unknown").astype(str)
    combined["Deck"] = combined["Deck"].fillna("Unknown").astype(str)
    combined["Side"] = combined["Side"].fillna("Unknown").astype(str)
    combined["CryoSleep"] = combined["CryoSleep"].astype(bool)
    combined["VIP"] = combined["VIP"].astype(bool)

    combined["TotalSpend"] = combined[spend_cols].sum(axis=1)
    combined["LogSpend"] = np.log1p(combined["TotalSpend"])
    combined["LuxurySpend"] = combined["Spa"] + combined["VRDeck"]
    combined["EssentialSpend"] = (
        combined["RoomService"] + combined["FoodCourt"] + combined["ShoppingMall"]
    )
    combined["SpendPerPerson"] = combined["TotalSpend"] / combined["GroupSize"].replace(
        0, 1
    )
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)
    combined["CryoSpendMismatch"] = (
        (combined["CryoSleep"] & (combined["TotalSpend"] > 0))
        | (~combined["CryoSleep"] & (combined["TotalSpend"] == 0))
    ).astype(int)
    combined["AgeGroup"] = (
        pd.cut(
            combined["Age"],
            bins=[0, 12, 17, 30, 45, 60, 100],
            labels=["Child", "Teen", "Young", "Adult", "Middle", "Senior"],
            include_lowest=True,
        )
        .astype(object)
        .fillna("Unknown")
    )
    combined["IsChild"] = (combined["Age"] < 13).astype(int)
    combined["IsSenior"] = (combined["Age"] >= 60).astype(int)
    combined["HomeDest"] = combined["HomePlanet"] + "__" + combined["Destination"]
    combined["DeckSide"] = combined["Deck"] + "__" + combined["Side"]
    combined["CabinNumBin"] = (
        pd.qcut(
            combined["CabinNum"].rank(method="first"),
            q=10,
            labels=False,
            duplicates="drop",
        )
        .astype(int)
        .astype(str)
    )
    combined["GroupSpendMean"] = combined.groupby("GroupId", observed=False)[
        "TotalSpend"
    ].transform("mean")
    combined["GroupSpendStd"] = (
        combined.groupby("GroupId", observed=False)["TotalSpend"]
        .transform("std")
        .fillna(0.0)
    )
    combined["GroupAgeMean"] = combined.groupby("GroupId", observed=False)[
        "Age"
    ].transform("mean")
    combined["GroupNoSpendRate"] = combined.groupby("GroupId", observed=False)[
        "NoSpend"
    ].transform("mean")
    combined["SurnameSpendMean"] = combined.groupby("Surname", observed=False)[
        "TotalSpend"
    ].transform("mean")
    combined["SurnameCryoRate"] = combined.groupby("Surname", observed=False)[
        "CryoSleep"
    ].transform("mean")

    features = [
        "HomePlanet",
        "Destination",
        "CryoSleep",
        "VIP",
        "Deck",
        "Side",
        "AgeGroup",
        "HomeDest",
        "DeckSide",
        "CabinNumBin",
        "Age",
        "CabinNum",
        "GroupSize",
        "GroupMemberIdx",
        "SurnameSize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpend",
        "LogSpend",
        "LuxurySpend",
        "EssentialSpend",
        "SpendPerPerson",
        "NoSpend",
        "SpendMissingCount",
        "IsAlone",
        "IsChild",
        "IsSenior",
        "CabinKnown",
        "CryoSpendMismatch",
        "GroupSpendMean",
        "GroupSpendStd",
        "GroupAgeMean",
        "GroupNoSpendRate",
        "SurnameSpendMean",
        "SurnameCryoRate",
    ]
    engineered = combined[features].copy()
    engineered["CryoSleep"] = engineered["CryoSleep"].astype(int)
    engineered["VIP"] = engineered["VIP"].astype(int)
    train_x = engineered.iloc[: len(train)].reset_index(drop=True)
    test_x = engineered.iloc[len(train) :].reset_index(drop=True)
    return train_x, test_x


def _spaceship_best_threshold(
    probabilities: np.ndarray, y_true: pd.Series | np.ndarray
) -> tuple[float, float]:
    y_array = np.asarray(y_true).astype(int)
    best_threshold = 0.5
    best_score = float(accuracy_score(y_array, probabilities >= best_threshold))
    for threshold in np.arange(0.35, 0.66, 0.01):
        score = float(accuracy_score(y_array, probabilities >= threshold))
        if score > best_score:
            best_threshold = float(round(threshold, 2))
            best_score = score
    return best_threshold, best_score


def _spaceship_catboost(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    folds: int,
) -> tuple[float, np.ndarray, float]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("catboost is not installed") from exc

    cat_cols = train_x.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        train_x[col] = train_x[col].fillna("Unknown").astype(str)
        test_x[col] = test_x[col].fillna("Unknown").astype(str)
    cat_idx = [train_x.columns.get_loc(col) for col in cat_cols]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(train_x), dtype=float)
    test_prob = np.zeros(len(test_x), dtype=float)
    for train_idx, valid_idx in skf.split(train_x, y):
        model = CatBoostClassifier(
            depth=8,
            iterations=900,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="Accuracy",
            l2_leaf_reg=6.0,
            random_strength=0.8,
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        model.fit(
            train_x.iloc[train_idx],
            y.iloc[train_idx],
            cat_features=cat_idx,
            eval_set=(train_x.iloc[valid_idx], y.iloc[valid_idx]),
            use_best_model=True,
            verbose=False,
        )
        oof_prob[valid_idx] = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
        test_prob += model.predict_proba(test_x)[:, 1] / folds

    best_threshold, best_score = _spaceship_best_threshold(oof_prob, y)
    submission_preds = (test_prob >= best_threshold).astype(bool)
    return best_score, submission_preds, best_threshold


def benchmark_spaceship(
    data_dir: Path, folds: int, write_submission: bool
) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Transported"].astype(int)
    train_x, test_x = _build_spaceship_features(train, test)

    cat_cols = train_x.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [col for col in train_x.columns if col not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    candidates: list[tuple[str, Any]] = [
        ("logreg", LogisticRegression(max_iter=2000, C=2.0)),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=2
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_depth=8, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=700, random_state=RANDOM_STATE, min_samples_leaf=2
            ),
        ),
    ]

    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}
    dense_cache_train = pd.get_dummies(train_x, drop_first=False)
    dense_cache_test = pd.get_dummies(test_x, drop_first=False)
    dense_cache_test = dense_cache_test.reindex(
        columns=dense_cache_train.columns, fill_value=0
    )

    for name, model in candidates:
        if name == "hgb":
            scores = cross_val_score(
                model, dense_cache_train, y, cv=skf, scoring="accuracy", n_jobs=1
            )
            benchmarks.append({"model": name, "score": round(float(scores.mean()), 5)})
            model.fit(dense_cache_train, y)
            trained_predictions[name] = model.predict(dense_cache_test).astype(bool)
            continue
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, train_x, y, cv=skf, scoring="accuracy", n_jobs=1)
        benchmarks.append({"model": name, "score": round(float(scores.mean()), 5)})
        pipe.fit(train_x, y)
        trained_predictions[name] = pipe.predict(test_x).astype(bool)

    try:
        cat_score, cat_preds, cat_threshold = _spaceship_catboost(
            train_x.copy(), test_x.copy(), y, folds
        )
        benchmarks.append(
            {
                "model": "catboost",
                "score": round(cat_score, 5),
                "threshold": round(cat_threshold, 2),
            }
        )
        trained_predictions["catboost"] = cat_preds
    except RuntimeError:
        pass

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = (
            _submission_dir("spaceship-titanic")
            / f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame(
            {
                "PassengerId": test["PassengerId"],
                "Transported": trained_predictions[best["model"]].astype(bool),
            }
        ).to_csv(submission_path, index=False)

    return LabResult(
        competition="spaceship-titanic",
        metric_name="accuracy",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
