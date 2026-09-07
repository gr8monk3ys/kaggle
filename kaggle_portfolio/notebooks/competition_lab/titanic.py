"""Titanic: binary classification on the classic tabular starter.

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


def _build_titanic_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.drop(columns=["Survived"]), test], axis=0, ignore_index=True
    )

    combined["Title"] = (
        combined["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .fillna("Unknown")
        .str.strip()
    )
    combined["FamilySize"] = (
        combined["SibSp"].fillna(0) + combined["Parch"].fillna(0) + 1
    )
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
    combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"].replace(0, 1)
    combined["Embarked"] = combined["Embarked"].fillna(
        combined["Embarked"].mode().iloc[0]
    )
    combined["CabinDeck"] = combined["Cabin"].fillna("U").astype(str).str[0]
    combined["TicketPrefix"] = (
        combined["Ticket"]
        .fillna("NONE")
        .astype(str)
        .str.replace(r"[./]", " ", regex=True)
        .str.split()
        .str[0]
        .where(lambda s: ~s.str.isdigit(), "NONE")
    )
    age_group = combined.groupby(["Pclass", "Title"])["Age"].transform("median")
    combined["Age"] = combined["Age"].fillna(age_group).fillna(combined["Age"].median())
    combined["Pclass"] = combined["Pclass"].astype(str)

    features = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "CabinDeck",
        "TicketPrefix",
        "Title",
    ]
    engineered = combined[features].copy()
    train_x = engineered.iloc[: len(train)].reset_index(drop=True)
    test_x = engineered.iloc[len(train) :].reset_index(drop=True)
    return train_x, test_x


def _titanic_catboost(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    folds: int,
) -> tuple[float, np.ndarray]:
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
    scores: list[float] = []
    for train_idx, valid_idx in skf.split(train_x, y):
        model = CatBoostClassifier(
            depth=6,
            iterations=500,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        model.fit(
            train_x.iloc[train_idx],
            y.iloc[train_idx],
            cat_features=cat_idx,
            verbose=False,
        )
        preds = model.predict(train_x.iloc[valid_idx]).reshape(-1)
        scores.append(accuracy_score(y.iloc[valid_idx], preds))

    final_model = CatBoostClassifier(
        depth=6,
        iterations=500,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    final_model.fit(train_x, y, cat_features=cat_idx, verbose=False)
    submission_preds = final_model.predict(test_x).reshape(-1).astype(int)
    return float(np.mean(scores)), submission_preds


def benchmark_titanic(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Survived"].astype(int)
    train_x, test_x = _build_titanic_features(train, test)

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
        ("logreg", LogisticRegression(max_iter=2000, C=2.5)),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=2
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
    for name, model in candidates:
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, train_x, y, cv=skf, scoring="accuracy", n_jobs=1)
        benchmarks.append({"model": name, "score": round(float(scores.mean()), 5)})
        pipe.fit(train_x, y)
        trained_predictions[name] = pipe.predict(test_x)

    try:
        cat_score, cat_preds = _titanic_catboost(
            train_x.copy(), test_x.copy(), y, folds
        )
        benchmarks.append({"model": "catboost", "score": round(cat_score, 5)})
        trained_predictions["catboost"] = cat_preds
    except RuntimeError:
        pass

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = (
            _submission_dir("titanic")
            / f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame(
            {
                "PassengerId": test["PassengerId"],
                "Survived": trained_predictions[best["model"]].astype(int),
            }
        ).to_csv(submission_path, index=False)

    return LabResult(
        competition="titanic",
        metric_name="accuracy",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
