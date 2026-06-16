#!/usr/bin/env python3
"""Benchmark local Kaggle competition baselines and optionally submit them.

Supports a focused set of entered competitions already used in this repo:
    - titanic
    - spaceship-titanic
    - nlp-getting-started
    - house-prices-advanced-regression-techniques
    - store-sales-time-series-forecasting
    - playground-series-s6e3
    - deep-past-initiative-machine-translation
    - march-machine-learning-mania-2026

Examples
--------
    python -m kaggle_portfolio.notebooks.local_competition_lab titanic
    python -m kaggle_portfolio.notebooks.local_competition_lab titanic --write-submission
    python -m kaggle_portfolio.notebooks.local_competition_lab spaceship-titanic --submit
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder

from kaggle_portfolio.shared.kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = ROOT / ".competition_lab"
RANDOM_STATE = 42

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


@dataclass(frozen=True)
class LabResult:
    competition: str
    metric_name: str
    best_model: str
    best_score: float
    benchmark_rows: list[dict[str, Any]]
    submission_path: Path | None = None


def _run_kaggle(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*kaggle_command(), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _ensure_data(slug: str, force_download: bool = False) -> Path:
    data_dir = LAB_ROOT / slug / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files and not force_download:
        return data_dir

    result = _run_kaggle("competitions", "download", "-c", slug, "-p", str(data_dir), "--force")
    if result.returncode != 0:
        raise SystemExit(
            f"Failed to download {slug}: {summarize_subprocess_error(result.stdout, result.stderr)}"
        )

    zip_files = list(data_dir.glob("*.zip"))
    for archive in zip_files:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    return data_dir


def _submission_dir(slug: str) -> Path:
    out = LAB_ROOT / slug / "submissions"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _benchmark_dir(slug: str) -> Path:
    out = LAB_ROOT / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")


def _save_summary(result: LabResult) -> None:
    out_dir = _benchmark_dir(result.competition)
    payload = {
        "competition": result.competition,
        "metric_name": result.metric_name,
        "best_model": result.best_model,
        "best_score": round(result.best_score, 6),
        "benchmarks": result.benchmark_rows,
        "submission_path": str(result.submission_path) if result.submission_path else None,
    }
    (out_dir / "latest_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_benchmarks(result: LabResult) -> None:
    print(f"{BLUE}=== {result.competition} local benchmark ==={RESET}")
    print(f"{'Model':<20} {result.metric_name:>10}")
    print("-" * 33)
    for row in result.benchmark_rows:
        color = GREEN if row["model"] == result.best_model else RESET
        print(f"{color}{row['model']:<20}{RESET} {row['score']:>10.5f}")
    print("")
    print(
        f"Best: {GREEN}{result.best_model}{RESET} "
        f"({result.metric_name}={result.best_score:.5f})"
    )
    if result.submission_path:
        print(f"Submission file: {result.submission_path}")


def _submit(slug: str, submission_path: Path, message: str) -> None:
    result = _run_kaggle(
        "competitions",
        "submit",
        "-c",
        slug,
        "-f",
        str(submission_path),
        "-m",
        message,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Submission failed: {summarize_subprocess_error(result.stdout, result.stderr)}"
        )
    print(result.stdout.strip() or "Submission accepted.")


def _build_titanic_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train.drop(columns=["Survived"]), test], axis=0, ignore_index=True)

    combined["Title"] = (
        combined["Name"].str.extract(r",\s*([^.]*)\.", expand=False).fillna("Unknown").str.strip()
    )
    combined["FamilySize"] = combined["SibSp"].fillna(0) + combined["Parch"].fillna(0) + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
    combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"].replace(0, 1)
    combined["Embarked"] = combined["Embarked"].fillna(combined["Embarked"].mode().iloc[0])
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
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    candidates: list[tuple[str, Any]] = [
        ("logreg", LogisticRegression(max_iter=2000, C=2.5)),
        ("rf", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=2)),
        ("et", ExtraTreesClassifier(n_estimators=700, random_state=RANDOM_STATE, min_samples_leaf=2)),
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
        cat_score, cat_preds = _titanic_catboost(train_x.copy(), test_x.copy(), y, folds)
        benchmarks.append({"model": "catboost", "score": round(cat_score, 5)})
        trained_predictions["catboost"] = cat_preds
    except RuntimeError:
        pass

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir("titanic") / f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        pd.DataFrame(
            {"PassengerId": test["PassengerId"], "Survived": trained_predictions[best["model"]].astype(int)}
        ).to_csv(submission_path, index=False)

    return LabResult(
        competition="titanic",
        metric_name="accuracy",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


def _build_spaceship_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train.drop(columns=["Transported"]), test], axis=0, ignore_index=True)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    group_id = combined["PassengerId"].astype(str).str.split("_").str[0]
    combined["GroupId"] = group_id
    combined["GroupSize"] = group_id.map(group_id.value_counts()).astype(int)
    combined["GroupMemberIdx"] = (
        pd.to_numeric(combined["PassengerId"].astype(str).str.split("_").str[-1], errors="coerce").fillna(0).astype(int)
    )
    combined["Surname"] = combined["Name"].fillna("Unknown Unknown").astype(str).str.split().str[-1]
    combined["SurnameSize"] = combined["Surname"].map(combined["Surname"].value_counts()).astype(int)

    cabin = combined["Cabin"].fillna("Unknown/0/U").astype(str).str.split("/", expand=True)
    combined["Deck"] = cabin[0].fillna("Unknown").astype(str)
    combined["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    combined["Side"] = cabin[2].fillna("Unknown").astype(str)
    combined["CabinKnown"] = combined["Cabin"].notna().astype(int)

    def _mode_map(df: pd.DataFrame, key: str, value: str) -> dict[str, object]:
        grouped = (
            df[[key, value]]
            .dropna(subset=[value])
            .groupby(key, observed=False)[value]
            .agg(lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else s.iloc[0])
        )
        return grouped.to_dict()

    for col in spend_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["SpendMissingCount"] = combined[spend_cols].isna().sum(axis=1).astype(int)
    raw_total_spend = combined[spend_cols].sum(axis=1, min_count=1)
    combined["NoSpendObserved"] = combined[spend_cols].fillna(0.0).sum(axis=1).eq(0)

    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        combined[col] = combined[col].fillna(combined["GroupId"].map(_mode_map(combined, "GroupId", col)))
    combined["HomePlanet"] = combined["HomePlanet"].fillna(combined["Surname"].map(_mode_map(combined, "Surname", "HomePlanet")))
    combined["Destination"] = combined["Destination"].fillna(combined["Surname"].map(_mode_map(combined, "Surname", "Destination")))
    combined["HomePlanet"] = combined["HomePlanet"].fillna(combined["Deck"].map(_mode_map(combined, "Deck", "HomePlanet")))
    combined["Destination"] = combined["Destination"].fillna("TRAPPIST-1e")

    deck_cabin_median = combined.groupby("Deck", observed=False)["CabinNum"].median()
    combined["CabinNum"] = combined["CabinNum"].fillna(combined["Deck"].map(deck_cabin_median))
    combined["CabinNum"] = combined["CabinNum"].fillna(combined["CabinNum"].median()).astype(float)

    combined["CryoSleep"] = combined["CryoSleep"].where(combined["CryoSleep"].notna(), np.nan)
    cryo_missing = combined["CryoSleep"].isna()
    combined.loc[cryo_missing & combined["NoSpendObserved"], "CryoSleep"] = True
    cryo_missing = combined["CryoSleep"].isna()
    combined.loc[cryo_missing & (raw_total_spend.fillna(0) > 0), "CryoSleep"] = False
    combined["CryoSleep"] = combined["CryoSleep"].where(
        combined["CryoSleep"].notna(),
        combined["GroupId"].map(_mode_map(combined, "GroupId", "CryoSleep")),
    )
    combined["CryoSleep"] = combined["CryoSleep"].where(combined["CryoSleep"].notna(), False)

    for col in spend_cols:
        group_median = combined.groupby(["HomePlanet", "Deck"], observed=False)[col].transform("median")
        combined[col] = combined[col].fillna(group_median)
        combined[col] = combined[col].fillna(combined[col].median())
    combined.loc[combined["CryoSleep"].astype(bool), spend_cols] = 0.0

    combined["Age"] = combined["Age"].fillna(combined.groupby("GroupId", observed=False)["Age"].transform("median"))
    combined["Age"] = combined["Age"].fillna(combined.groupby(["HomePlanet", "Deck"], observed=False)["Age"].transform("median"))
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
    combined["EssentialSpend"] = combined["RoomService"] + combined["FoodCourt"] + combined["ShoppingMall"]
    combined["SpendPerPerson"] = combined["TotalSpend"] / combined["GroupSize"].replace(0, 1)
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)
    combined["CryoSpendMismatch"] = (
        (combined["CryoSleep"] & (combined["TotalSpend"] > 0))
        | (~combined["CryoSleep"] & (combined["TotalSpend"] == 0))
    ).astype(int)
    combined["AgeGroup"] = pd.cut(
        combined["Age"],
        bins=[0, 12, 17, 30, 45, 60, 100],
        labels=["Child", "Teen", "Young", "Adult", "Middle", "Senior"],
        include_lowest=True,
    ).astype(object).fillna("Unknown")
    combined["IsChild"] = (combined["Age"] < 13).astype(int)
    combined["IsSenior"] = (combined["Age"] >= 60).astype(int)
    combined["HomeDest"] = combined["HomePlanet"] + "__" + combined["Destination"]
    combined["DeckSide"] = combined["Deck"] + "__" + combined["Side"]
    combined["CabinNumBin"] = pd.qcut(
        combined["CabinNum"].rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    ).astype(int).astype(str)
    combined["GroupSpendMean"] = combined.groupby("GroupId", observed=False)["TotalSpend"].transform("mean")
    combined["GroupSpendStd"] = combined.groupby("GroupId", observed=False)["TotalSpend"].transform("std").fillna(0.0)
    combined["GroupAgeMean"] = combined.groupby("GroupId", observed=False)["Age"].transform("mean")
    combined["GroupNoSpendRate"] = combined.groupby("GroupId", observed=False)["NoSpend"].transform("mean")
    combined["SurnameSpendMean"] = combined.groupby("Surname", observed=False)["TotalSpend"].transform("mean")
    combined["SurnameCryoRate"] = combined.groupby("Surname", observed=False)["CryoSleep"].transform("mean")

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


def _spaceship_best_threshold(probabilities: np.ndarray, y_true: pd.Series | np.ndarray) -> tuple[float, float]:
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


def benchmark_spaceship(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Transported"].astype(int)
    train_x, test_x = _build_spaceship_features(train, test)

    cat_cols = train_x.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [col for col in train_x.columns if col not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    candidates: list[tuple[str, Any]] = [
        ("logreg", LogisticRegression(max_iter=2000, C=2.0)),
        ("rf", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=2)),
        ("hgb", HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE)),
        ("et", ExtraTreesClassifier(n_estimators=700, random_state=RANDOM_STATE, min_samples_leaf=2)),
    ]

    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}
    dense_cache_train = pd.get_dummies(train_x, drop_first=False)
    dense_cache_test = pd.get_dummies(test_x, drop_first=False)
    dense_cache_test = dense_cache_test.reindex(columns=dense_cache_train.columns, fill_value=0)

    for name, model in candidates:
        if name == "hgb":
            scores = cross_val_score(model, dense_cache_train, y, cv=skf, scoring="accuracy", n_jobs=1)
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
        cat_score, cat_preds, cat_threshold = _spaceship_catboost(train_x.copy(), test_x.copy(), y, folds)
        benchmarks.append({"model": "catboost", "score": round(cat_score, 5), "threshold": round(cat_threshold, 2)})
        trained_predictions["catboost"] = cat_preds
    except RuntimeError:
        pass

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir("spaceship-titanic") / f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        pd.DataFrame(
            {"PassengerId": test["PassengerId"], "Transported": trained_predictions[best["model"]].astype(bool)}
        ).to_csv(submission_path, index=False)

    return LabResult(
        competition="spaceship-titanic",
        metric_name="accuracy",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


def benchmark_nlp(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["target"].astype(int)
    text_train = train["text"].fillna("")
    text_test = test["text"].fillna("")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    candidates: list[tuple[str, Pipeline]] = [
        (
            "word_lr",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
                    ("model", LogisticRegression(max_iter=2000, C=4.0)),
                ]
            ),
        ),
        (
            "char_lr",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=90000, sublinear_tf=True)),
                    ("model", LogisticRegression(max_iter=2000, C=3.0)),
                ]
            ),
        ),
        (
            "cnb",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=70000, sublinear_tf=True)),
                    ("model", ComplementNB(alpha=0.4)),
                ]
            ),
        ),
    ]

    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}
    for name, pipe in candidates:
        scores = cross_val_score(pipe, text_train, y, cv=skf, scoring="f1", n_jobs=1)
        benchmarks.append({"model": name, "score": round(float(scores.mean()), 5)})
        pipe.fit(text_train, y)
        trained_predictions[name] = pipe.predict(text_test)

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir("nlp-getting-started") / f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        pd.DataFrame({"id": test["id"], "target": trained_predictions[best["model"]].astype(int)}).to_csv(
            submission_path,
            index=False,
        )

    return LabResult(
        competition="nlp-getting-started",
        metric_name="f1",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


def _playground_prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train.drop(columns=["Churn"]), test], axis=0, ignore_index=True)
    combined["TotalCharges"] = pd.to_numeric(combined["TotalCharges"], errors="coerce")
    tenure = combined["tenure"].replace(0, np.nan)
    combined["ChargesPerTenure"] = combined["TotalCharges"] / tenure
    combined["MonthlyToTenureRatio"] = combined["MonthlyCharges"] / tenure
    combined["IsNewCustomer"] = combined["tenure"].fillna(0).le(6).astype(int)
    combined["HasFiber"] = combined["InternetService"].fillna("").eq("Fiber optic").astype(int)
    combined["HasAutoPay"] = (
        combined["PaymentMethod"].fillna("").str.contains("automatic", case=False, regex=False).astype(int)
    )
    combined["HasStreaming"] = (
        combined["StreamingTV"].fillna("").eq("Yes") | combined["StreamingMovies"].fillna("").eq("Yes")
    ).astype(int)
    combined["HasSecurityBundle"] = (
        combined["OnlineSecurity"].fillna("").eq("Yes")
        | combined["TechSupport"].fillna("").eq("Yes")
        | combined["OnlineBackup"].fillna("").eq("Yes")
        | combined["DeviceProtection"].fillna("").eq("Yes")
    ).astype(int)

    object_cols = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        combined[col] = combined[col].fillna("Missing").astype(str)
        combined[col] = pd.Categorical(combined[col]).codes.astype(int)

    combined = combined.fillna(-1)
    train_x = combined.iloc[: len(train)].reset_index(drop=True)
    test_x = combined.iloc[len(train) :].reset_index(drop=True)
    return train_x, test_x


def _playground_original_path(data_dir: Path) -> Path | None:
    candidate = data_dir.parent / "orig" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    return candidate if candidate.exists() else None


def _playground_model_result(
    model: Any,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> tuple[float, np.ndarray, np.ndarray]:
    oof = cross_val_predict(model, train_x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    score = float(roc_auc_score(y, oof))
    model.fit(train_x, y)
    test_pred = model.predict_proba(test_x)[:, 1]
    return score, oof, test_pred


def _concat_feature_block(df: pd.DataFrame, updates: dict[str, Any]) -> pd.DataFrame:
    if not updates:
        return df
    block = pd.DataFrame(updates, index=df.index)
    return pd.concat([df, block], axis=1).copy()


def _playground_advanced_feature_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    def pctrank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ref = np.sort(np.asarray(reference, dtype=np.float32))
        if ref.size == 0:
            return np.zeros(len(values), dtype=np.float32)
        return (np.searchsorted(ref, values, side="left") / ref.size).astype(np.float32)

    def zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ref = np.asarray(reference, dtype=np.float32)
        if ref.size == 0:
            return np.zeros(len(values), dtype=np.float32)
        sigma = float(ref.std())
        if sigma == 0.0 or np.isnan(sigma):
            return np.zeros(len(values), dtype=np.float32)
        return ((values - float(ref.mean())) / sigma).astype(np.float32)

    train = train.copy()
    test = test.copy()
    orig = orig.copy()
    if "customerID" in orig.columns:
        orig = orig.drop(columns=["customerID"])

    target = "Churn"
    train[target] = (
        train[target].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(train[target]).astype(int)
    )
    orig[target] = (
        orig[target].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(orig[target]).astype(int)
    )
    cat_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for df in (train, test, orig):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("missing").str.strip()
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df[col] = df[col].fillna(df[col].median())

    new_num_cols: list[str] = []
    freq_maps = {
        col: pd.concat([train[col], test[col], orig[col]], axis=0).value_counts(normalize=True)
        for col in num_cols
    }
    train = _concat_feature_block(
        train,
        {f"FREQ_{col}": train[col].map(freq_maps[col]).fillna(0).astype("float32") for col in num_cols},
    )
    test = _concat_feature_block(
        test,
        {f"FREQ_{col}": test[col].map(freq_maps[col]).fillna(0).astype("float32") for col in num_cols},
    )
    orig = _concat_feature_block(
        orig,
        {f"FREQ_{col}": orig[col].map(freq_maps[col]).fillna(0).astype("float32") for col in num_cols},
    )
    new_num_cols.extend([f"FREQ_{col}" for col in num_cols])

    all_num = pd.concat([train[num_cols], test[num_cols], orig[num_cols]], axis=0, ignore_index=True)
    rank_updates_train: dict[str, Any] = {}
    rank_updates_test: dict[str, Any] = {}
    rank_updates_orig: dict[str, Any] = {}
    for col in num_cols:
        ranks = all_num[col].rank(method="average", pct=True).astype("float32").to_numpy()
        rank_updates_train[f"RANK_{col}"] = ranks[: len(train)]
        rank_updates_test[f"RANK_{col}"] = ranks[len(train) : len(train) + len(test)]
        rank_updates_orig[f"RANK_{col}"] = ranks[len(train) + len(test) :]
    train = _concat_feature_block(train, rank_updates_train)
    test = _concat_feature_block(test, rank_updates_test)
    orig = _concat_feature_block(orig, rank_updates_orig)
    new_num_cols.extend([f"RANK_{col}" for col in num_cols])

    def _power_updates(df: pd.DataFrame) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        for col in num_cols:
            values = df[col].astype("float32")
            updates[f"LOG1P_{col}"] = np.log1p(values.clip(lower=0)).astype("float32")
            updates[f"SQRT_{col}"] = np.sqrt(values.clip(lower=0)).astype("float32")
            updates[f"INV1P_{col}"] = (1.0 / (1.0 + values.clip(lower=0))).astype("float32")
        return updates

    train = _concat_feature_block(train, _power_updates(train))
    test = _concat_feature_block(test, _power_updates(test))
    orig = _concat_feature_block(orig, _power_updates(orig))
    new_num_cols.extend([f"LOG1P_{col}" for col in num_cols])
    new_num_cols.extend([f"SQRT_{col}" for col in num_cols])
    new_num_cols.extend([f"INV1P_{col}" for col in num_cols])

    def _core_numeric_updates(df: pd.DataFrame) -> dict[str, Any]:
        charges_deviation = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        service_yes_count = (df[service_cols] == "Yes").sum(axis=1).astype("float32")
        return {
            "charges_deviation": charges_deviation,
            "abs_charges_dev": np.abs(charges_deviation).astype("float32"),
            "monthly_to_total_ratio": (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32"),
            "total_to_monthly_ratio": (df["TotalCharges"] / (df["MonthlyCharges"] + 1)).astype("float32"),
            "avg_monthly_charges": (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32"),
            "tenure_x_monthly": (df["tenure"] * df["MonthlyCharges"]).astype("float32"),
            "tenure_x_total": (df["tenure"] * df["TotalCharges"]).astype("float32"),
            "service_yes_count": service_yes_count,
            "service_no_count": (df[service_cols] == "No").sum(axis=1).astype("float32"),
            "service_other_count": (
                df[service_cols].isin(["No phone service", "No internet service"]).sum(axis=1).astype("float32")
            ),
            "service_count": service_yes_count,
            "has_internet": (df["InternetService"] != "No").astype("float32"),
            "has_phone": (df["PhoneService"] == "Yes").astype("float32"),
        }

    train = _concat_feature_block(train, _core_numeric_updates(train))
    test = _concat_feature_block(test, _core_numeric_updates(test))
    orig = _concat_feature_block(orig, _core_numeric_updates(orig))
    new_num_cols.extend(
        [
            "charges_deviation",
            "abs_charges_dev",
            "monthly_to_total_ratio",
            "total_to_monthly_ratio",
            "avg_monthly_charges",
            "tenure_x_monthly",
            "tenure_x_total",
            "service_yes_count",
            "service_no_count",
            "service_other_count",
            "service_count",
            "has_internet",
            "has_phone",
        ]
    )

    new_cat_cols: list[str] = []
    tenure_bins = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
    monthly_bins = pd.qcut(
        pd.concat([train["MonthlyCharges"], test["MonthlyCharges"], orig["MonthlyCharges"]]),
        q=40,
        retbins=True,
        duplicates="drop",
    )[1]
    total_bins = pd.qcut(
        pd.concat([train["TotalCharges"], test["TotalCharges"], orig["TotalCharges"]]),
        q=60,
        retbins=True,
        duplicates="drop",
    )[1]
    train = _concat_feature_block(
        train,
        {
            "tenure_bin": pd.cut(train["tenure"], bins=tenure_bins, include_lowest=True).astype(str),
            "MonthlyCharges_bin": pd.cut(train["MonthlyCharges"], bins=monthly_bins, include_lowest=True).astype(str),
            "TotalCharges_bin": pd.cut(train["TotalCharges"], bins=total_bins, include_lowest=True).astype(str),
        },
    )
    test = _concat_feature_block(
        test,
        {
            "tenure_bin": pd.cut(test["tenure"], bins=tenure_bins, include_lowest=True).astype(str),
            "MonthlyCharges_bin": pd.cut(test["MonthlyCharges"], bins=monthly_bins, include_lowest=True).astype(str),
            "TotalCharges_bin": pd.cut(test["TotalCharges"], bins=total_bins, include_lowest=True).astype(str),
        },
    )
    orig = _concat_feature_block(
        orig,
        {
            "tenure_bin": pd.cut(orig["tenure"], bins=tenure_bins, include_lowest=True).astype(str),
            "MonthlyCharges_bin": pd.cut(orig["MonthlyCharges"], bins=monthly_bins, include_lowest=True).astype(str),
            "TotalCharges_bin": pd.cut(orig["TotalCharges"], bins=total_bins, include_lowest=True).astype(str),
        },
    )
    new_cat_cols.extend(["tenure_bin", "MonthlyCharges_bin", "TotalCharges_bin"])

    yn_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
    ]
    def _yn_updates(df: pd.DataFrame) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        for col in yn_cols:
            values = df[col].astype(str)
            updates[f"ISYES_{col}"] = (values == "Yes").astype("float32")
            updates[f"ISNO_{col}"] = (values == "No").astype("float32")
            updates[f"ISOTHER_{col}"] = (~values.isin(["Yes", "No"])).astype("float32")
        return updates

    train = _concat_feature_block(train, _yn_updates(train))
    test = _concat_feature_block(test, _yn_updates(test))
    orig = _concat_feature_block(orig, _yn_updates(orig))
    new_num_cols.extend([f"ISYES_{col}" for col in yn_cols])
    new_num_cols.extend([f"ISNO_{col}" for col in yn_cols])
    new_num_cols.extend([f"ISOTHER_{col}" for col in yn_cols])

    cat_feature_updates = {"train": {}, "test": {}, "orig": {}}
    for left, right in (
        ("Contract", "InternetService"),
        ("PaymentMethod", "Contract"),
        ("InternetService", "OnlineSecurity"),
        ("PaymentMethod", "PaperlessBilling"),
        ("Contract", "PaperlessBilling"),
        ("InternetService", "TechSupport"),
    ):
        name = f"{left}__{right}"
        cat_feature_updates["train"][name] = train[left].astype(str) + "|" + train[right].astype(str)
        cat_feature_updates["test"][name] = test[left].astype(str) + "|" + test[right].astype(str)
        cat_feature_updates["orig"][name] = orig[left].astype(str) + "|" + orig[right].astype(str)
        new_cat_cols.append(name)

    for left, middle, right in (("Contract", "InternetService", "PaymentMethod"),):
        name = f"{left}__{middle}__{right}"
        cat_feature_updates["train"][name] = (
            train[left].astype(str) + "|" + train[middle].astype(str) + "|" + train[right].astype(str)
        )
        cat_feature_updates["test"][name] = (
            test[left].astype(str) + "|" + test[middle].astype(str) + "|" + test[right].astype(str)
        )
        cat_feature_updates["orig"][name] = (
            orig[left].astype(str) + "|" + orig[middle].astype(str) + "|" + orig[right].astype(str)
        )
        new_cat_cols.append(name)

    ngram_top_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
    ]
    for left, right in combinations(ngram_top_cols, 2):
        name = f"BG_{left}_{right}"
        cat_feature_updates["train"][name] = train[left].astype(str) + "_" + train[right].astype(str)
        cat_feature_updates["test"][name] = test[left].astype(str) + "_" + test[right].astype(str)
        cat_feature_updates["orig"][name] = orig[left].astype(str) + "_" + orig[right].astype(str)
        new_cat_cols.append(name)

    for left, middle, right in combinations(ngram_top_cols[:4], 3):
        name = f"TG_{left}_{middle}_{right}"
        cat_feature_updates["train"][name] = (
            train[left].astype(str) + "_" + train[middle].astype(str) + "_" + train[right].astype(str)
        )
        cat_feature_updates["test"][name] = (
            test[left].astype(str) + "_" + test[middle].astype(str) + "_" + test[right].astype(str)
        )
        cat_feature_updates["orig"][name] = (
            orig[left].astype(str) + "_" + orig[middle].astype(str) + "_" + orig[right].astype(str)
        )
        new_cat_cols.append(name)
    train = _concat_feature_block(train, cat_feature_updates["train"])
    test = _concat_feature_block(test, cat_feature_updates["test"])
    orig = _concat_feature_block(orig, cat_feature_updates["orig"])

    counted_cat_cols = cat_cols + new_cat_cols
    all_cat_frame = pd.concat([train[counted_cat_cols], test[counted_cat_cols], orig[counted_cat_cols]], ignore_index=True)
    count_updates_train: dict[str, Any] = {}
    count_updates_test: dict[str, Any] = {}
    count_updates_orig: dict[str, Any] = {}
    for col in counted_cat_cols:
        counts = all_cat_frame[col].value_counts(dropna=False)
        train_counts = train[col].map(counts).fillna(0).astype("float32")
        test_counts = test[col].map(counts).fillna(0).astype("float32")
        orig_counts = orig[col].map(counts).fillna(0).astype("float32")
        count_updates_train[f"CAT_CNT_{col}"] = train_counts
        count_updates_test[f"CAT_CNT_{col}"] = test_counts
        count_updates_orig[f"CAT_CNT_{col}"] = orig_counts
        count_updates_train[f"CAT_RARE_{col}"] = (train_counts <= 50).astype("float32")
        count_updates_test[f"CAT_RARE_{col}"] = (test_counts <= 50).astype("float32")
        count_updates_orig[f"CAT_RARE_{col}"] = (orig_counts <= 50).astype("float32")
        new_num_cols.extend([f"CAT_CNT_{col}", f"CAT_RARE_{col}"])
    train = _concat_feature_block(train, count_updates_train)
    test = _concat_feature_block(test, count_updates_test)
    orig = _concat_feature_block(orig, count_updates_orig)

    orig_global = float(orig[target].mean())
    orig_proba_updates_train: dict[str, Any] = {}
    orig_proba_updates_test: dict[str, Any] = {}
    orig_proba_updates_orig: dict[str, Any] = {}
    for col in cat_cols + num_cols + new_cat_cols:
        lookup = orig.groupby(col, observed=False)[target].mean()
        name = f"ORIG_proba_{col}"
        orig_proba_updates_train[name] = train[col].map(lookup).fillna(orig_global).astype("float32")
        orig_proba_updates_test[name] = test[col].map(lookup).fillna(orig_global).astype("float32")
        orig_proba_updates_orig[name] = orig[col].map(lookup).fillna(orig_global).astype("float32")
        new_num_cols.append(name)
    train = _concat_feature_block(train, orig_proba_updates_train)
    test = _concat_feature_block(test, orig_proba_updates_test)
    orig = _concat_feature_block(orig, orig_proba_updates_orig)

    orig_churner_tc = orig.loc[orig[target] == 1, "TotalCharges"].to_numpy(dtype=np.float32)
    orig_nonchurner_tc = orig.loc[orig[target] == 0, "TotalCharges"].to_numpy(dtype=np.float32)
    orig_tc = orig["TotalCharges"].to_numpy(dtype=np.float32)
    orig_is_mc_mean = orig.groupby("InternetService", observed=False)["MonthlyCharges"].mean()
    distribution_cols = [
        "pctrank_nonchurner_TC",
        "pctrank_churner_TC",
        "pctrank_orig_TC",
        "zscore_churn_gap_TC",
        "zscore_nonchurner_TC",
        "pctrank_churn_gap_TC",
        "resid_IS_MC",
        "cond_pctrank_IS_TC",
        "cond_pctrank_C_TC",
    ]
    def _distribution_updates(df: pd.DataFrame) -> dict[str, Any]:
        tc = df["TotalCharges"].to_numpy(dtype=np.float32)
        updates: dict[str, Any] = {
            "pctrank_nonchurner_TC": pctrank_against(tc, orig_nonchurner_tc),
            "pctrank_churner_TC": pctrank_against(tc, orig_churner_tc),
            "pctrank_orig_TC": pctrank_against(tc, orig_tc),
            "zscore_churn_gap_TC": (
                np.abs(zscore_against(tc, orig_churner_tc)) - np.abs(zscore_against(tc, orig_nonchurner_tc))
            ).astype(np.float32),
            "zscore_nonchurner_TC": zscore_against(tc, orig_nonchurner_tc),
            "pctrank_churn_gap_TC": (
                pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)
            ).astype(np.float32),
            "resid_IS_MC": (
                df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0).to_numpy(dtype=np.float32)
            ).astype(np.float32),
        }
        cond_is_vals = np.zeros(len(df), dtype=np.float32)
        for cat_val in orig["InternetService"].dropna().astype(str).unique():
            mask = df["InternetService"].astype(str) == cat_val
            if not mask.any():
                continue
            ref = orig.loc[orig["InternetService"].astype(str) == cat_val, "TotalCharges"].to_numpy(dtype=np.float32)
            cond_is_vals[mask.to_numpy()] = pctrank_against(
                df.loc[mask, "TotalCharges"].to_numpy(dtype=np.float32),
                ref,
            )
        updates["cond_pctrank_IS_TC"] = cond_is_vals

        cond_contract_vals = np.zeros(len(df), dtype=np.float32)
        for cat_val in orig["Contract"].dropna().astype(str).unique():
            mask = df["Contract"].astype(str) == cat_val
            if not mask.any():
                continue
            ref = orig.loc[orig["Contract"].astype(str) == cat_val, "TotalCharges"].to_numpy(dtype=np.float32)
            cond_contract_vals[mask.to_numpy()] = pctrank_against(
                df.loc[mask, "TotalCharges"].to_numpy(dtype=np.float32),
                ref,
            )
        updates["cond_pctrank_C_TC"] = cond_contract_vals
        return updates

    train = _concat_feature_block(train, _distribution_updates(train))
    test = _concat_feature_block(test, _distribution_updates(test))
    new_num_cols.extend(distribution_cols)

    num_as_cat: list[str] = []
    num_as_cat_updates_train: dict[str, Any] = {}
    num_as_cat_updates_test: dict[str, Any] = {}
    num_as_cat_updates_orig: dict[str, Any] = {}
    for col in num_cols:
        cat_name = f"CAT_{col}"
        num_as_cat.append(cat_name)
        num_as_cat_updates_train[cat_name] = train[col].astype(str)
        num_as_cat_updates_test[cat_name] = test[col].astype(str)
        num_as_cat_updates_orig[cat_name] = orig[col].astype(str)
    train = _concat_feature_block(train, num_as_cat_updates_train)
    test = _concat_feature_block(test, num_as_cat_updates_test)
    orig = _concat_feature_block(orig, num_as_cat_updates_orig)

    for df in (train, test, orig):
        for col in cat_cols + new_cat_cols + num_as_cat:
            df[col] = df[col].astype("category")

    feature_cols = num_cols + cat_cols + new_num_cols + new_cat_cols + num_as_cat
    te_cols = num_as_cat + cat_cols + new_cat_cols
    drop_raw_cols = num_as_cat + cat_cols + new_cat_cols
    return train, test, feature_cols, te_cols, drop_raw_cols


def _playground_advanced_xgboost_result(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    folds: int,
    seeds: tuple[int, ...] = (11, 42, 99),
) -> tuple[float, np.ndarray, np.ndarray]:
    target = "Churn"
    train_frame, test_frame, feature_cols, te_cols, drop_raw_cols = _playground_advanced_feature_frames(
        train,
        test,
        orig,
    )
    n_splits = min(max(3, folds), 5)
    inner_splits = min(3, n_splits)
    stats = ["std", "min", "max"]
    oof_sum = np.zeros(len(train_frame), dtype=float)
    oof_count = np.zeros(len(train_frame), dtype=float)
    test_pred = np.zeros(len(test_frame), dtype=float)
    total_models = 0

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed") from exc

    params = {
        "n_estimators": 6000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.81,
        "colsample_bytree": 0.55,
        "min_child_weight": 6,
        "reg_alpha": 1.25,
        "reg_lambda": 1.3,
        "gamma": 0.35,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "enable_categorical": True,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 200,
    }

    for seed in seeds:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, valid_idx in outer_cv.split(train_frame, train_frame[target]):
            x_train = train_frame.iloc[train_idx][feature_cols + [target]].reset_index(drop=True).copy()
            y_train = train_frame.iloc[train_idx][target].to_numpy()
            y_valid = train_frame.iloc[valid_idx][target].to_numpy()
            x_valid = train_frame.iloc[valid_idx][feature_cols].reset_index(drop=True).copy()
            x_test = test_frame[feature_cols].reset_index(drop=True).copy()
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

            te_stat_cols = [f"TE1_{col}_{stat}" for col in te_cols for stat in stats]
            x_train = _concat_feature_block(x_train, {name: np.nan for name in te_stat_cols})

            for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
                x_inner_train = x_train.loc[inner_train_idx, feature_cols + [target]].copy()
                x_inner_valid = x_train.loc[inner_valid_idx, feature_cols].copy()
                for col in te_cols:
                    grouped = x_inner_train.groupby(col, observed=False)[target].agg(stats)
                    grouped.columns = [f"TE1_{col}_{stat}" for stat in stats]
                    x_inner_valid = x_inner_valid.merge(grouped, on=col, how="left")
                    for name in grouped.columns:
                        x_train.loc[inner_valid_idx, name] = x_inner_valid[name].to_numpy(dtype="float32")

            for col in te_cols:
                grouped = x_train.groupby(col, observed=False)[target].agg(stats)
                grouped.columns = [f"TE1_{col}_{stat}" for stat in stats]
                x_valid = x_valid.merge(grouped.astype("float32"), on=col, how="left")
                x_test = x_test.merge(grouped.astype("float32"), on=col, how="left")
                for name in grouped.columns:
                    x_train[name] = x_train[name].fillna(0).astype("float32")
                    x_valid[name] = x_valid[name].fillna(0).astype("float32")
                    x_test[name] = x_test[name].fillna(0).astype("float32")

            if te_cols:
                mean_encoder = TargetEncoder(
                    cv=inner_splits,
                    shuffle=True,
                    smooth="auto",
                    target_type="binary",
                    random_state=seed,
                )
                mean_cols = [f"TE_{col}" for col in te_cols]
                x_train = pd.concat(
                    [
                        x_train,
                        pd.DataFrame(
                            mean_encoder.fit_transform(x_train[te_cols], y_train),
                            columns=mean_cols,
                            index=x_train.index,
                        ),
                    ],
                    axis=1,
                ).copy()
                x_valid = pd.concat(
                    [
                        x_valid,
                        pd.DataFrame(
                            mean_encoder.transform(x_valid[te_cols]),
                            columns=mean_cols,
                            index=x_valid.index,
                        ),
                    ],
                    axis=1,
                ).copy()
                x_test = pd.concat(
                    [
                        x_test,
                        pd.DataFrame(
                            mean_encoder.transform(x_test[te_cols]),
                            columns=mean_cols,
                            index=x_test.index,
                        ),
                    ],
                    axis=1,
                ).copy()

            for df in (x_train, x_valid, x_test):
                for col in te_cols:
                    df[col] = df[col].astype(str).astype("category")
                df.drop(columns=drop_raw_cols, inplace=True)
            x_train.drop(columns=[target], inplace=True)

            model = xgb.XGBClassifier(**params, random_state=seed)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=False,
            )
            valid_pred = model.predict_proba(x_valid)[:, 1]
            oof_sum[valid_idx] += valid_pred
            oof_count[valid_idx] += 1.0
            test_pred += model.predict_proba(x_test)[:, 1]
            total_models += 1

    if total_models == 0 or np.any(oof_count == 0):
        raise RuntimeError("XGBoost did not produce a complete OOF prediction.")

    oof = oof_sum / oof_count
    test_pred = test_pred / total_models
    return float(roc_auc_score(train_frame[target].to_numpy(), oof)), oof, test_pred


def _playground_pseudo_label_mask(
    predictions: np.ndarray,
    lower_quantile: float = 0.08,
    upper_quantile: float = 0.92,
    absolute_confidence: float = 0.92,
) -> np.ndarray:
    lower_threshold = min(float(np.quantile(predictions, lower_quantile)), 1.0 - absolute_confidence)
    upper_threshold = max(float(np.quantile(predictions, upper_quantile)), absolute_confidence)
    return (predictions <= lower_threshold) | (predictions >= upper_threshold)


def _playground_pseudo_label_weights(predictions: np.ndarray) -> np.ndarray:
    confidence = np.abs(predictions - 0.5) * 2.0
    return np.clip(0.15 + (0.25 * confidence), 0.2, 0.4)


def _playground_advanced_xgboost_pseudo_result(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    folds: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    target = "Churn"
    train_frame, test_frame, feature_cols, te_cols, drop_raw_cols = _playground_advanced_feature_frames(
        train,
        test,
        orig,
    )
    n_splits = min(max(3, folds), 5)
    inner_splits = min(3, n_splits)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    stats = ["std", "min", "max"]
    oof = np.zeros(len(train_frame), dtype=float)
    test_pred = np.zeros(len(test_frame), dtype=float)

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed") from exc

    params = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 80,
    }

    for train_idx, valid_idx in outer_cv.split(train_frame, train_frame[target]):
        x_train = train_frame.iloc[train_idx][feature_cols + [target]].reset_index(drop=True).copy()
        y_train = train_frame.iloc[train_idx][target].to_numpy()
        y_valid = train_frame.iloc[valid_idx][target].to_numpy()
        x_valid = train_frame.iloc[valid_idx][feature_cols].reset_index(drop=True).copy()
        x_test = test_frame[feature_cols].reset_index(drop=True).copy()
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

        te_stat_cols = [f"TE1_{col}_{stat}" for col in te_cols for stat in stats]
        x_train = _concat_feature_block(x_train, {name: np.nan for name in te_stat_cols})

        for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
            x_inner_train = x_train.loc[inner_train_idx, feature_cols + [target]].copy()
            x_inner_valid = x_train.loc[inner_valid_idx, feature_cols].copy()
            for col in te_cols:
                grouped = x_inner_train.groupby(col, observed=False)[target].agg(stats)
                grouped.columns = [f"TE1_{col}_{stat}" for stat in stats]
                x_inner_valid = x_inner_valid.merge(grouped, on=col, how="left")
                for name in grouped.columns:
                    x_train.loc[inner_valid_idx, name] = x_inner_valid[name].to_numpy(dtype="float32")

        for col in te_cols:
            grouped = x_train.groupby(col, observed=False)[target].agg(stats)
            grouped.columns = [f"TE1_{col}_{stat}" for stat in stats]
            x_valid = x_valid.merge(grouped.astype("float32"), on=col, how="left")
            x_test = x_test.merge(grouped.astype("float32"), on=col, how="left")
            for name in grouped.columns:
                x_train[name] = x_train[name].fillna(0).astype("float32")
                x_valid[name] = x_valid[name].fillna(0).astype("float32")
                x_test[name] = x_test[name].fillna(0).astype("float32")

        mean_encoder = TargetEncoder(
            cv=inner_splits,
            shuffle=True,
            smooth="auto",
            target_type="binary",
            random_state=RANDOM_STATE,
        )
        mean_cols = [f"TE_{col}" for col in te_cols]
        x_train = pd.concat(
            [
                x_train,
                pd.DataFrame(
                    mean_encoder.fit_transform(x_train[te_cols], y_train),
                    columns=mean_cols,
                    index=x_train.index,
                ),
            ],
            axis=1,
        ).copy()
        x_valid = pd.concat(
            [
                x_valid,
                pd.DataFrame(
                    mean_encoder.transform(x_valid[te_cols]),
                    columns=mean_cols,
                    index=x_valid.index,
                ),
            ],
            axis=1,
        ).copy()
        x_test = pd.concat(
            [
                x_test,
                pd.DataFrame(
                    mean_encoder.transform(x_test[te_cols]),
                    columns=mean_cols,
                    index=x_test.index,
                ),
            ],
            axis=1,
        ).copy()

        for df in (x_train, x_valid, x_test):
            for col in te_cols:
                df[col] = df[col].astype(str).astype("category")
            df.drop(columns=drop_raw_cols, inplace=True)
        x_train.drop(columns=[target], inplace=True)

        base_model = xgb.XGBClassifier(**params)
        base_model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        base_test_pred = base_model.predict_proba(x_test)[:, 1]
        pseudo_mask = _playground_pseudo_label_mask(base_test_pred)
        min_pseudo_rows = max(2000, len(base_test_pred) // 50)

        if pseudo_mask.sum() < min_pseudo_rows:
            oof[valid_idx] = base_model.predict_proba(x_valid)[:, 1]
            test_pred += base_test_pred / n_splits
            continue

        pseudo_x = x_test.loc[pseudo_mask].copy()
        pseudo_y = (base_test_pred[pseudo_mask] >= 0.5).astype(int)
        pseudo_weights = _playground_pseudo_label_weights(base_test_pred[pseudo_mask])

        augmented_x = pd.concat([x_train, pseudo_x], axis=0, ignore_index=True).copy()
        augmented_y = np.concatenate([y_train, pseudo_y])
        sample_weight = np.concatenate([np.ones(len(y_train), dtype=float), pseudo_weights])

        model = xgb.XGBClassifier(**params)
        model.fit(
            augmented_x,
            augmented_y,
            sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1] / n_splits

    return float(roc_auc_score(train_frame[target].to_numpy(), oof)), oof, test_pred


def _playground_catboost_selected_features(feature_cols: list[str]) -> list[str]:
    base_num_cols = {"tenure", "MonthlyCharges", "TotalCharges"}
    base_cat_cols = {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    }
    selected_feature_cols: list[str] = []
    for col in feature_cols:
        if col in base_num_cols or col in base_cat_cols:
            selected_feature_cols.append(col)
            continue
        if col in {
            "charges_deviation",
            "abs_charges_dev",
            "monthly_to_total_ratio",
            "total_to_monthly_ratio",
            "avg_monthly_charges",
            "tenure_x_monthly",
            "tenure_x_total",
            "service_yes_count",
            "service_no_count",
            "service_other_count",
            "service_count",
            "has_internet",
            "has_phone",
            "pctrank_nonchurner_TC",
            "pctrank_churner_TC",
            "pctrank_orig_TC",
            "zscore_churn_gap_TC",
            "zscore_nonchurner_TC",
            "pctrank_churn_gap_TC",
            "resid_IS_MC",
            "cond_pctrank_IS_TC",
            "cond_pctrank_C_TC",
            "tenure_bin",
            "MonthlyCharges_bin",
            "TotalCharges_bin",
        }:
            selected_feature_cols.append(col)
            continue
        if (
            col.startswith(("FREQ_", "RANK_", "LOG1P_", "SQRT_", "INV1P_", "ISYES_", "ISNO_", "ISOTHER_"))
            or "__" in col
            or col.startswith(("BG_", "TG_"))
        ):
            selected_feature_cols.append(col)
            continue
        if col.startswith("ORIG_proba_"):
            source_col = col.removeprefix("ORIG_proba_")
            if source_col in base_num_cols or source_col in base_cat_cols or source_col in {
                "tenure_bin",
                "MonthlyCharges_bin",
                "TotalCharges_bin",
            }:
                selected_feature_cols.append(col)
    return selected_feature_cols


def _playground_lightgbm_selected_features(feature_cols: list[str]) -> list[str]:
    base_num_cols = {"tenure", "MonthlyCharges", "TotalCharges"}
    base_cat_cols = {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    }
    core_num_cols = {
        "charges_deviation",
        "abs_charges_dev",
        "monthly_to_total_ratio",
        "total_to_monthly_ratio",
        "avg_monthly_charges",
        "tenure_x_monthly",
        "tenure_x_total",
        "service_yes_count",
        "service_no_count",
        "service_other_count",
        "service_count",
        "has_internet",
        "has_phone",
        "pctrank_nonchurner_TC",
        "pctrank_churner_TC",
        "pctrank_orig_TC",
        "zscore_churn_gap_TC",
        "zscore_nonchurner_TC",
        "pctrank_churn_gap_TC",
        "resid_IS_MC",
        "cond_pctrank_IS_TC",
        "cond_pctrank_C_TC",
        "tenure_bin",
        "MonthlyCharges_bin",
        "TotalCharges_bin",
    }
    selected_feature_cols: list[str] = []
    for col in feature_cols:
        if col in base_num_cols or col in base_cat_cols or col in core_num_cols:
            selected_feature_cols.append(col)
            continue
        if (
            col.startswith(
                (
                    "FREQ_",
                    "RANK_",
                    "LOG1P_",
                    "SQRT_",
                    "INV1P_",
                    "CAT_CNT_",
                    "CAT_RARE_",
                    "ORIG_proba_",
                    "ISYES_",
                    "ISNO_",
                    "ISOTHER_",
                )
            )
            or col.startswith("BG_")
            or "__" in col
        ):
            selected_feature_cols.append(col)
    return selected_feature_cols


def _playground_lightgbm_te_columns(feature_cols: list[str]) -> list[str]:
    preferred = {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_bin",
        "MonthlyCharges_bin",
        "TotalCharges_bin",
        "BG_Contract_InternetService",
        "BG_Contract_PaymentMethod",
        "BG_InternetService_PaymentMethod",
        "BG_PaperlessBilling_PaymentMethod",
    }
    return [col for col in feature_cols if col in preferred]


def _playground_advanced_lightgbm_result(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    folds: int,
    seeds: tuple[int, ...] = (11, 42),
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed") from exc

    target = "Churn"
    train_frame, test_frame, feature_cols, te_cols, _drop_raw_cols = _playground_advanced_feature_frames(
        train,
        test,
        orig,
    )
    selected_feature_cols = _playground_lightgbm_selected_features(feature_cols)
    selected_te_cols = [col for col in _playground_lightgbm_te_columns(te_cols) if col in selected_feature_cols]
    train_model = train_frame[selected_feature_cols].copy()
    test_model = test_frame[selected_feature_cols].copy()
    y = train_frame[target].to_numpy()
    n_splits = min(max(3, folds), 5)
    inner_splits = min(3, n_splits)
    stats = ["mean", "std"]

    raw_cat_cols = [col for col in selected_feature_cols if str(train_model[col].dtype) == "category"]
    oof_sum = np.zeros(len(train_model), dtype=float)
    oof_count = np.zeros(len(train_model), dtype=float)
    test_pred = np.zeros(len(test_model), dtype=float)
    total_models = 0

    params = {
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 48,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_samples": 120,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "objective": "binary",
        "metric": "auc",
        "n_jobs": -1,
        "verbose": -1,
    }

    for seed in seeds:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, valid_idx in outer_cv.split(train_model, y):
            x_train = train_model.iloc[train_idx].reset_index(drop=True).copy()
            y_train = y[train_idx]
            x_valid = train_model.iloc[valid_idx].reset_index(drop=True).copy()
            y_valid = y[valid_idx]
            x_test = test_model.copy()
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

            te_stat_cols = [f"LGB_TE1_{col}_{stat}" for col in selected_te_cols for stat in stats]
            x_train = _concat_feature_block(x_train, {name: np.nan for name in te_stat_cols})

            for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
                x_inner_train = x_train.loc[inner_train_idx, selected_te_cols].copy()
                x_inner_train[target] = y_train[inner_train_idx]
                x_inner_valid = x_train.loc[inner_valid_idx, selected_te_cols].copy()
                for col in selected_te_cols:
                    grouped = x_inner_train.groupby(col, observed=False)[target].agg(stats)
                    grouped.columns = [f"LGB_TE1_{col}_{stat}" for stat in stats]
                    x_inner_valid = x_inner_valid.merge(grouped, on=col, how="left")
                    for name in grouped.columns:
                        x_train.loc[inner_valid_idx, name] = x_inner_valid[name].to_numpy(dtype="float32")

            for col in selected_te_cols:
                grouped = pd.DataFrame({col: x_train[col], target: y_train}).groupby(col, observed=False)[target].agg(stats)
                grouped.columns = [f"LGB_TE1_{col}_{stat}" for stat in stats]
                x_valid = x_valid.merge(grouped.astype("float32"), on=col, how="left")
                x_test = x_test.merge(grouped.astype("float32"), on=col, how="left")
                for name in grouped.columns:
                    x_train[name] = x_train[name].fillna(0).astype("float32")
                    x_valid[name] = x_valid[name].fillna(0).astype("float32")
                    x_test[name] = x_test[name].fillna(0).astype("float32")

            if selected_te_cols:
                mean_encoder = TargetEncoder(
                    cv=inner_splits,
                    shuffle=True,
                    smooth="auto",
                    target_type="binary",
                    random_state=seed,
                )
                mean_cols = [f"LGB_TE_{col}" for col in selected_te_cols]
                x_train = pd.concat(
                    [
                        x_train,
                        pd.DataFrame(
                            mean_encoder.fit_transform(x_train[selected_te_cols], y_train),
                            columns=mean_cols,
                            index=x_train.index,
                        ),
                    ],
                    axis=1,
                ).copy()
                x_valid = pd.concat(
                    [
                        x_valid,
                        pd.DataFrame(
                            mean_encoder.transform(x_valid[selected_te_cols]),
                            columns=mean_cols,
                            index=x_valid.index,
                        ),
                    ],
                    axis=1,
                ).copy()
                x_test = pd.concat(
                    [
                        x_test,
                        pd.DataFrame(
                            mean_encoder.transform(x_test[selected_te_cols]),
                            columns=mean_cols,
                            index=x_test.index,
                        ),
                    ],
                    axis=1,
                ).copy()

            model = lgb.LGBMClassifier(**params, random_state=seed)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="auc",
                categorical_feature=raw_cat_cols,
                callbacks=[
                    lgb.early_stopping(120, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            valid_pred = model.predict_proba(x_valid)[:, 1]
            oof_sum[valid_idx] += valid_pred
            oof_count[valid_idx] += 1.0
            test_pred += model.predict_proba(x_test)[:, 1]
            total_models += 1

    if total_models == 0 or np.any(oof_count == 0):
        raise RuntimeError("LightGBM did not produce a complete OOF prediction.")

    oof = oof_sum / oof_count
    test_pred = test_pred / total_models
    return float(roc_auc_score(y, oof)), oof, test_pred


def _playground_advanced_catboost_result(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    folds: int,
    seeds: tuple[int, ...] = (11, 42),
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("catboost is not installed") from exc

    target = "Churn"
    train_frame, test_frame, feature_cols, _te_cols, _drop_raw_cols = _playground_advanced_feature_frames(
        train,
        test,
        orig,
    )
    selected_feature_cols = _playground_catboost_selected_features(feature_cols)
    train_model = train_frame[selected_feature_cols].copy()
    test_model = test_frame[selected_feature_cols].copy()
    y = train_frame[target].to_numpy()
    n_splits = min(max(3, folds), 5)

    cat_cols = [col for col in selected_feature_cols if str(train_model[col].dtype) == "category"]
    for df in (train_model, test_model):
        for col in cat_cols:
            df[col] = df[col].astype(str)

    cat_idx = [train_model.columns.get_loc(col) for col in cat_cols]
    oof_sum = np.zeros(len(train_model), dtype=float)
    oof_count = np.zeros(len(train_model), dtype=float)
    test_pred = np.zeros(len(test_model), dtype=float)
    total_models = 0

    params = {
        "iterations": 900,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 6.0,
        "subsample": 0.8,
        "bootstrap_type": "Bernoulli",
        "random_strength": 0.8,
        "allow_writing_files": False,
        "verbose": False,
    }

    for seed in seeds:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, valid_idx in outer_cv.split(train_model, y):
            model = CatBoostClassifier(**params, random_seed=seed)
            model.fit(
                train_model.iloc[train_idx],
                y[train_idx],
                cat_features=cat_idx,
                eval_set=(train_model.iloc[valid_idx], y[valid_idx]),
                use_best_model=True,
                verbose=False,
            )
            valid_pred = model.predict_proba(train_model.iloc[valid_idx])[:, 1]
            oof_sum[valid_idx] += valid_pred
            oof_count[valid_idx] += 1.0
            test_pred += model.predict_proba(test_model)[:, 1]
            total_models += 1

    if total_models == 0 or np.any(oof_count == 0):
        raise RuntimeError("CatBoost did not produce a complete OOF prediction.")

    oof = oof_sum / oof_count
    test_pred = test_pred / total_models
    return float(roc_auc_score(y, oof)), oof, test_pred


def _playground_best_blend(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
    step: float = 0.05,
) -> tuple[str, dict[str, float], float, np.ndarray] | None:
    names = list(predictions)
    if len(names) < 2:
        return None

    def _rank_scale(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
        denom = max(len(values) - 1, 1)
        return order.astype(float) / denom

    def _consider(weights: dict[str, float]) -> tuple[str, float, np.ndarray]:
        blend_names = tuple(weights)
        prob_oof = sum(weights[name] * oof_frames[name] for name in blend_names)
        prob_score = float(roc_auc_score(y, prob_oof))
        prob_pred = sum(weights[name] * test_frames[name] for name in blend_names)

        rank_oof = sum(weights[name] * rank_oof_frames[name] for name in blend_names)
        rank_score = float(roc_auc_score(y, rank_oof))
        rank_pred = sum(weights[name] * rank_test_frames[name] for name in blend_names)

        if rank_score > prob_score:
            return "rank", rank_score, rank_pred
        return "probability", prob_score, prob_pred

    units = max(2, int(round(1.0 / step)))
    best_weights: dict[str, float] | None = None
    best_kind = "probability"
    best_score = float("-inf")
    best_pred: np.ndarray | None = None

    for subset_size in range(2, min(len(names), 3) + 1):
        for subset in combinations(names, subset_size):
            oof_frames = {name: predictions[name][0] for name in subset}
            test_frames = {name: predictions[name][1] for name in subset}
            rank_oof_frames = {name: _rank_scale(predictions[name][0]) for name in subset}
            rank_test_frames = {name: _rank_scale(predictions[name][1]) for name in subset}

            if len(subset) == 2:
                for left_units in range(1, units):
                    weights = {
                        subset[0]: left_units / units,
                        subset[1]: 1.0 - (left_units / units),
                    }
                    kind, score, pred = _consider(weights)
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        best_kind = kind
                        best_pred = pred
                continue

            for first_units in range(units + 1):
                for second_units in range(units - first_units + 1):
                    third_units = units - first_units - second_units
                    raw_units = [first_units, second_units, third_units]
                    if sum(unit > 0 for unit in raw_units) < 2:
                        continue
                    weights = {name: raw_units[idx] / units for idx, name in enumerate(subset)}
                    kind, score, pred = _consider(weights)
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        best_kind = kind
                        best_pred = pred

    if best_weights is None or best_pred is None:
        return None
    return best_kind, best_weights, best_score, best_pred


def benchmark_playground_telco(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Churn"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(train["Churn"]).astype(int)
    train_x, test_x = _playground_prepare_features(train, test)
    skf = StratifiedKFold(n_splits=min(max(3, folds), 5), shuffle=True, random_state=RANDOM_STATE)

    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}
    blend_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    try:
        import lightgbm as lgb

        lgb_model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_score, lgb_oof, lgb_pred = _playground_model_result(lgb_model, train_x, test_x, y, skf)
        benchmarks.append({"model": "lightgbm", "score": round(float(lgb_score), 5)})
        trained_predictions["lightgbm"] = lgb_pred
        blend_inputs["lightgbm"] = (lgb_oof, lgb_pred)
    except ImportError:
        pass

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_score, xgb_oof, xgb_pred = _playground_model_result(xgb_model, train_x, test_x, y, skf)
        benchmarks.append({"model": "xgboost", "score": round(float(xgb_score), 5)})
        trained_predictions["xgboost"] = xgb_pred
        blend_inputs["xgboost"] = (xgb_oof, xgb_pred)
    except ImportError:
        pass

    original_path = _playground_original_path(data_dir)
    if original_path is not None:
        try:
            lgb_score, lgb_oof, lgb_pred = _playground_advanced_lightgbm_result(
                train,
                test,
                pd.read_csv(original_path),
                folds,
            )
            benchmarks.append({"model": "lightgbm_te", "score": round(float(lgb_score), 5)})
            trained_predictions["lightgbm_te"] = lgb_pred
            blend_inputs["lightgbm_te"] = (lgb_oof, lgb_pred)
        except (RuntimeError, ValueError):
            pass
        try:
            advanced_score, advanced_oof, advanced_pred = _playground_advanced_xgboost_result(
                train,
                test,
                pd.read_csv(original_path),
                folds,
            )
            benchmarks.append({"model": "xgboost_te", "score": round(float(advanced_score), 5)})
            trained_predictions["xgboost_te"] = advanced_pred
            blend_inputs["xgboost_te"] = (advanced_oof, advanced_pred)
        except (RuntimeError, ValueError):
            pass
        try:
            pseudo_score, pseudo_oof, pseudo_pred = _playground_advanced_xgboost_pseudo_result(
                train,
                test,
                pd.read_csv(original_path),
                folds,
            )
            benchmarks.append({"model": "xgboost_te_pseudo", "score": round(float(pseudo_score), 5)})
            trained_predictions["xgboost_te_pseudo"] = pseudo_pred
            blend_inputs["xgboost_te_pseudo"] = (pseudo_oof, pseudo_pred)
        except (RuntimeError, ValueError):
            pass
        try:
            cat_score, cat_oof, cat_pred = _playground_advanced_catboost_result(
                train,
                test,
                pd.read_csv(original_path),
                folds,
            )
            benchmarks.append({"model": "catboost_te", "score": round(float(cat_score), 5)})
            trained_predictions["catboost_te"] = cat_pred
            blend_inputs["catboost_te"] = (cat_oof, cat_pred)
        except (RuntimeError, ValueError):
            pass

    blend_result = _playground_best_blend(blend_inputs, y)
    if blend_result is not None:
        blend_kind, blend_weights, blend_score, blend_pred = blend_result
        benchmarks.append(
            {
                "model": "blend",
                "score": round(float(blend_score), 5),
                "blend_type": blend_kind,
                "weights": {
                    name: round(weight, 2)
                    for name, weight in blend_weights.items()
                    if weight > 0
                },
            }
        )
        trained_predictions["blend"] = blend_pred

    if not benchmarks:
        raise SystemExit("No Playground Series S6E3 models are available locally.")

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir("playground-series-s6e3") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame({"id": test["id"], "Churn": trained_predictions[best["model"]]}).to_csv(
            submission_path,
            index=False,
        )

    return LabResult(
        competition="playground-series-s6e3",
        metric_name="roc_auc",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


def _house_prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train.drop(columns=["SalePrice"]), test], axis=0, ignore_index=True)

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
    mode_fill_cols = ["MSZoning", "KitchenQual", "Electrical", "Exterior1st", "Exterior2nd", "SaleType", "Utilities"]

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
        combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        combined["LotFrontage"] = combined["LotFrontage"].fillna(combined["LotFrontage"].median())

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
    combined["TotalOutsideSF"] = _col("LotArea") + combined["TotalPorchSF"] + _col("PoolArea")
    combined["QualSF"] = _col("OverallQual") * _col("GrLivArea")
    combined["TotalHomeQuality"] = _col("OverallQual") + _col("OverallCond")
    combined["OverallGrade"] = _col("OverallQual") * _col("OverallCond")
    combined["TotalRooms"] = _col("TotRmsAbvGrd") + _col("KitchenAbvGr")
    combined["AgeWhenSold"] = yr_sold_num - _col("YearBuilt")
    combined["AgeSinceRemodel"] = yr_sold_num - _col("YearRemodAdd")
    combined["LivLotRatio"] = _col("GrLivArea") / _col("LotArea", 1.0).clip(lower=1)
    combined["BathPerRoom"] = combined["TotalBath"] / combined["TotalRooms"].replace(0, 1)
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
    return [combo for size in range(2, min(4, len(names)) + 1) for combo in combinations(names, size)]


def _house_weight_options(size: int, step: float = 0.05) -> list[tuple[float, ...]]:
    unit = int(round(1.0 / step))
    weights: list[tuple[float, ...]] = []

    def _build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(tuple((prefix + [remaining])[idx] * step for idx in range(len(prefix) + 1)))
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
                best_weights = {name: float(weight) for name, weight in zip(combo, weights)}
                best_pred = test_stack @ weight_arr

    if best_weights is None or best_pred is None:
        return None
    return best_weights, best_score, best_pred


def benchmark_house_prices(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    if outlier_mask.any():
        train = train.loc[~outlier_mask].reset_index(drop=True)

    y = np.log1p(train["SalePrice"])
    train_x, test_x = _house_prepare_features(train, test)
    dense_train = pd.get_dummies(train_x, dummy_na=True).apply(pd.to_numeric, errors="coerce")
    dense_test = pd.get_dummies(test_x, dummy_na=True).apply(pd.to_numeric, errors="coerce")
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
            ("model", ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=50000, random_state=RANDOM_STATE)),
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
                "weights": {name: round(weight, 2) for name, weight in blend_weights.items()},
            }
        )
        trained_predictions["blend"] = np.expm1(blend_pred).clip(min=0)

    best = min(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir("house-prices-advanced-regression-techniques") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame({"Id": test["Id"], "SalePrice": trained_predictions[best["model"]]}).to_csv(
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


def _store_sales_rmsle(y_true: Any, y_pred: Any) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_arr) - np.log1p(y_true_arr)) ** 2)))


def _store_sales_prediction_frame(history: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    target = target.copy()
    history["date"] = pd.to_datetime(history["date"])
    target["date"] = pd.to_datetime(target["date"])
    history["dow"] = history["date"].dt.dayofweek
    target["dow"] = target["date"].dt.dayofweek

    recent_140 = history.loc[history["date"] >= history["date"].max() - pd.Timedelta(days=140)]
    recent_56 = history.loc[history["date"] >= history["date"].max() - pd.Timedelta(days=56)]
    recent_28 = history.loc[history["date"] >= history["date"].max() - pd.Timedelta(days=28)]

    group_sf_dow_promo = (
        recent_140.groupby(["store_nbr", "family", "dow", "onpromotion"])["sales"]
        .mean()
        .rename("pred_sf_dow_promo")
        .reset_index()
    )
    group_sf_dow = (
        recent_140.groupby(["store_nbr", "family", "dow"])["sales"].mean().rename("pred_sf_dow").reset_index()
    )
    group_sf_28 = recent_28.groupby(["store_nbr", "family"])["sales"].mean().rename("pred_sf_28").reset_index()
    group_sf_56 = recent_56.groupby(["store_nbr", "family"])["sales"].mean().rename("pred_sf_56").reset_index()
    group_family_dow = recent_140.groupby(["family", "dow"])["sales"].mean().rename("pred_family_dow").reset_index()
    group_store_dow = (
        recent_140.groupby(["store_nbr", "dow"])["sales"].mean().rename("pred_store_dow").reset_index()
    )
    global_mean = float(history["sales"].mean())

    frame = (
        target.merge(group_sf_dow_promo, on=["store_nbr", "family", "dow", "onpromotion"], how="left")
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
    frame["hybrid_mean"] = 0.65 * frame["recent_dow_promo_mean"] + 0.35 * frame["recent_28_mean"]
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

    oil_filled = oil_df.set_index("date")["dcoilwtico"].resample("D").interpolate("linear")
    df["oil_price"] = df["date"].map(oil_filled).ffill().fillna(50.0)

    national_holidays = holidays_df.loc[holidays_df["locale"] == "National", "date"].drop_duplicates()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    df = df.merge(stores_df[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

    if "sales" in df.columns:
        grouped_sales = df.groupby(["store_nbr", "family"])["sales"]
        grouped_promo = df.groupby(["store_nbr", "family"])["onpromotion"]
        for lag in (7, 14, 28):
            df[f"lag_{lag}"] = grouped_sales.shift(lag)
        for window in (7, 14, 28):
            df[f"roll_mean_{window}"] = grouped_sales.transform(lambda s: s.shift(1).rolling(window).mean())
            df[f"roll_std_{window}"] = grouped_sales.transform(lambda s: s.shift(1).rolling(window).std())
        df["ewma_7"] = grouped_sales.transform(lambda s: s.shift(1).ewm(span=7).mean())
        df["promo_roll_mean_14"] = grouped_promo.transform(lambda s: s.shift(1).rolling(14).mean())
        df["promo_roll_mean_28"] = grouped_promo.transform(lambda s: s.shift(1).rolling(28).mean())
        df["history_mean"] = grouped_sales.transform(lambda s: s.shift(1).expanding().mean())
        df["trend_7_28"] = df["roll_mean_7"] / (df["roll_mean_28"] + 1)
        df["sales_momentum"] = df["roll_mean_7"] - df["roll_mean_28"]

    df["oil_to_trend"] = df["oil_price"] / (df.get("roll_mean_28", pd.Series(0, index=df.index)).fillna(0) + 1)
    df["promo_x_trend"] = df["onpromotion"] * df.get("trend_7_28", pd.Series(1.0, index=df.index)).fillna(1.0)
    return df


def _store_sales_history_artifacts(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                    "trend_7_28_fill": g["sales"].tail(7).mean() / (g["sales"].tail(28).mean() + 1),
                    "sales_momentum_fill": g["sales"].tail(7).mean() - g["sales"].tail(28).mean(),
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
        future[f"lag_{lag}"] = future[f"lag_{lag}_direct"].fillna(future[f"lag_{lag}_fill"])

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
        future[feature] = future.get(feature, pd.Series(np.nan, index=future.index)).fillna(future[fallback])

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
    working_history = history[["date", "store_nbr", "family", "onpromotion", "sales"]].copy()
    ordered_target = target.copy()
    ordered_target["_row_order"] = np.arange(len(ordered_target))
    predictions: list[pd.DataFrame] = []

    for pred_date in sorted(pd.to_datetime(ordered_target["date"]).drop_duplicates()):
        day_rows = ordered_target.loc[ordered_target["date"] == pred_date].copy()
        lag_lookup, history_summary, family_dow_history, store_dow_history = _store_sales_history_artifacts(
            working_history
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
        predictions.append(pd.DataFrame({"_row_order": day_rows["_row_order"].to_numpy(), "pred": day_pred}))
        history_extension = day_rows[["date", "store_nbr", "family", "onpromotion"]].copy()
        history_extension["sales"] = day_pred
        working_history = pd.concat([working_history, history_extension], ignore_index=True)

    ordered_predictions = pd.concat(predictions, ignore_index=True).sort_values("_row_order")
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

    history_features = _store_sales_make_features(history, oil_df, stores_df, holidays_df)
    category_maps: dict[str, dict[str, int]] = {}
    for col in ("family", "type"):
        mapping = {
            value: idx
            for idx, value in enumerate(sorted(pd.Index(history_features[col].astype(str)).drop_duplicates()))
        }
        category_maps[col] = mapping
        history_features[col] = history_features[col].astype(str).map(mapping).astype(int)
    history_features = history_features.fillna(0)

    lag_lookup, history_summary, family_dow_history, store_dow_history = _store_sales_history_artifacts(history)
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
    submission_future = _store_sales_build_future_frame(
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
        col for col in history_features.columns if col not in {"id", "date", "sales"} and history_features[col].dtype != "object"
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
        eval_set=[(validation_future[feature_cols], np.log1p(validation["sales"].clip(lower=0)))],
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


def benchmark_store_sales(data_dir: Path, _folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    valid_dates = sorted(train["date"].drop_duplicates())[-16:]
    validation = train.loc[train["date"].isin(valid_dates)].copy()
    history = train.loc[~train["date"].isin(valid_dates)].copy()
    validation_frame = _store_sales_prediction_frame(history, validation)

    benchmarks = [
        {
            "model": "recent_dow_promo_mean",
            "score": round(_store_sales_rmsle(validation["sales"], validation_frame["recent_dow_promo_mean"]), 5),
        },
        {
            "model": "recent_28_mean",
            "score": round(_store_sales_rmsle(validation["sales"], validation_frame["recent_28_mean"]), 5),
        },
        {
            "model": "hybrid_mean",
            "score": round(_store_sales_rmsle(validation["sales"], validation_frame["hybrid_mean"]), 5),
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
            future_score, _validation_pred, submission_pred = _store_sales_lightgbm_future_result(
                history,
                validation,
                test,
                stores_df,
                oil_df,
                holidays_df,
            )
            benchmarks.append({"model": "lightgbm_future", "score": round(future_score, 5)})
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
        pd.DataFrame({"id": test["id"], "sales": sales}).to_csv(submission_path, index=False)

    return LabResult(
        competition="store-sales-time-series-forecasting",
        metric_name="rmsle",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


def _deep_past_normalize(text: str) -> str:
    normalized = str(text or "").lower()
    for old, new in {
        "…": " ",
        "...": " ",
        "„": " ",
        "“": " ",
        "”": " ",
        '"': " ",
        "'": " ",
        "`": " ",
        "´": " ",
        "{": " ",
        "}": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "/": " ",
        "\\": " ",
        ",": " ",
        ".": " ",
        ";": " ",
        ":": " ",
        "!": " ",
        "?": " ",
    }.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _deep_past_best_match(corpus: pd.Series, query: str) -> tuple[int, float]:
    normalized_corpus = corpus.fillna("").map(_deep_past_normalize)
    normalized_query = _deep_past_normalize(query)

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=1)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)

    char_matrix = char_vec.fit_transform(normalized_corpus)
    word_matrix = word_vec.fit_transform(normalized_corpus)
    char_score = linear_kernel(char_vec.transform([normalized_query]), char_matrix)[0]
    word_score = linear_kernel(word_vec.transform([normalized_query]), word_matrix)[0]
    scores = 0.7 * char_score + 0.3 * word_score

    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])


def _deep_past_display_name_candidates(row: pd.Series) -> list[str]:
    candidates: list[str] = []
    for value in [row.get("label", ""), row.get("aliases", ""), row.get("note", "")]:
        raw = str(value or "").strip()
        if not raw or raw.lower() == "nan":
            continue
        parts = [part.strip() for part in raw.split("|")]
        for part in parts:
            if not part:
                continue
            candidates.append(part)
            stripped = re.sub(r"^cuneiform\s+(tablet|envelope)\s+", "", part, flags=re.IGNORECASE).strip()
            if stripped and stripped != part:
                candidates.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _deep_past_optional_csv(
    data_dir: Path,
    file_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    path = data_dir / file_name
    if not path.exists():
        return pd.DataFrame(columns=required_columns)
    frame = pd.read_csv(path)
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def _deep_past_sentence_rows(sentences: pd.DataFrame, published_row: pd.Series) -> pd.DataFrame:
    if sentences.empty or "display_name" not in sentences:
        return sentences.iloc[0:0]
    display_names = sentences["display_name"].astype(str).str.strip()
    best = sentences.iloc[0:0]
    for candidate in _deep_past_display_name_candidates(published_row):
        matches = sentences.loc[display_names == candidate]
        if len(matches) > len(best):
            best = matches
    if best.empty:
        return best
    return (
        best.loc[:, ["line_number", "translation"]]
        .dropna(subset=["line_number", "translation"])
        .sort_values("line_number")
        .reset_index(drop=True)
    )


def _deep_past_assign_sentences_to_rows(test: pd.DataFrame, sentence_rows: pd.DataFrame) -> list[str]:
    ordered_test = test.sort_values(["line_start", "line_end"]).reset_index(drop=True)
    ordered_sentences = sentence_rows.sort_values("line_number").reset_index(drop=True)
    predictions: list[str] = []

    for idx, row in ordered_test.iterrows():
        start = int(row["line_start"])
        next_start = int(ordered_test.loc[idx + 1, "line_start"]) if idx + 1 < len(ordered_test) else None
        if next_start is None:
            mask = ordered_sentences["line_number"] >= start
        else:
            mask = (ordered_sentences["line_number"] >= start) & (ordered_sentences["line_number"] < next_start)
        translation = " ".join(ordered_sentences.loc[mask, "translation"].astype(str)).strip()
        predictions.append(translation)

    return predictions


def _deep_past_split_translation_by_rows(text: str, test: pd.DataFrame) -> list[str]:
    weights = (
        test.sort_values(["line_start", "line_end"])["line_end"].fillna(test["line_start"]).astype(int)
        - test.sort_values(["line_start", "line_end"])["line_start"].astype(int)
        + 1
    ).clip(lower=1).tolist()
    words = str(text or "").split()
    if not words:
        return ["" for _ in weights]

    total_weight = sum(weights) or len(weights)
    chunks: list[str] = []
    position = 0
    for idx, weight in enumerate(weights):
        remaining_words = len(words) - position
        remaining_groups = len(weights) - idx
        if idx == len(weights) - 1:
            take = remaining_words
        else:
            take = max(1, round(len(words) * weight / total_weight))
            take = min(take, remaining_words - (remaining_groups - 1))
        chunks.append(" ".join(words[position : position + take]).strip())
        position += take
    return chunks


def _deep_past_train_retrieval(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[list[str], float]:
    query = " ".join(test.sort_values(["line_start", "line_end"])["transliteration"].astype(str))
    best_idx, best_score = _deep_past_best_match(train["transliteration"], query)
    best_translation = str(train.iloc[best_idx]["translation"])
    predictions = _deep_past_split_translation_by_rows(best_translation, test)
    fallback = sample.sort_values("id")["translation"].astype(str).tolist()
    completed = [pred.strip() or fallback[idx] for idx, pred in enumerate(predictions)]
    return completed, best_score


def benchmark_deep_past(data_dir: Path, _folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv").sort_values(["line_start", "line_end"]).reset_index(drop=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv").sort_values("id").reset_index(drop=True)
    published = _deep_past_optional_csv(data_dir, "published_texts.csv", ["transliteration", "label", "aliases", "note"])
    sentences = _deep_past_optional_csv(
        data_dir,
        "Sentences_Oare_FirstWord_LinNum.csv",
        ["display_name", "line_number", "translation"],
    )

    query = " ".join(test["transliteration"].astype(str))
    published_available = not published.empty and "transliteration" in published and published["transliteration"].notna().any()
    sentences_available = (
        not sentences.empty
        and "display_name" in sentences
        and "translation" in sentences
        and "line_number" in sentences
    )
    published_score = 0.0
    published_row = pd.Series(dtype="object")
    sentence_rows = sentences.iloc[0:0]
    sentence_predictions: list[str] = []
    if published_available:
        published_idx, published_score = _deep_past_best_match(published["transliteration"], query)
        published_row = published.iloc[published_idx]
        if sentences_available:
            sentence_rows = _deep_past_sentence_rows(sentences, published_row)
            if not sentence_rows.empty:
                sentence_predictions = _deep_past_assign_sentences_to_rows(test, sentence_rows)
    train_predictions, train_score = _deep_past_train_retrieval(train, test, sample)
    sentence_coverage = (
        sum(1 for pred in sentence_predictions if pred.strip()) / len(test) if len(sentence_predictions) == len(test) else 0.0
    )
    published_decision_score = min(1.0, published_score + 0.15 * sentence_coverage)

    benchmarks: list[dict[str, Any]] = [
        {
            "model": "published_sentence_match",
            "score": round(published_decision_score, 5),
            "source_label": str(published_row.get("label", "")),
            "available": bool(published_available and sentences_available),
        },
        {
            "model": "train_retrieval",
            "score": round(train_score, 5),
        },
    ]

    chosen_model = "train_retrieval"
    chosen_score = train_score
    predictions = train_predictions
    if (
        published_score >= 0.6
        and len(sentence_predictions) == len(test)
        and all(pred.strip() for pred in sentence_predictions)
        and published_decision_score >= train_score
    ):
        chosen_model = "published_sentence_match"
        chosen_score = published_decision_score
        predictions = sentence_predictions

    submission_path = None
    if write_submission:
        submission_path = (
            _submission_dir("deep-past-initiative-machine-translation")
            / f"submission_{_safe_slug(chosen_model)}_{int(chosen_score * 100000)}.csv"
        )
        pd.DataFrame({"id": test["id"], "translation": predictions}).to_csv(submission_path, index=False)

    return LabResult(
        competition="deep-past-initiative-machine-translation",
        metric_name="decision_score",
        best_model=chosen_model,
        best_score=float(chosen_score),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


def _march_seed_number(value: Any) -> float:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return np.nan
    return float(digits[:2])


def _march_team_game_rows(results: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["WTeamID"],
            "OppTeamID": results["LTeamID"],
            "Score": results["WScore"],
            "OppScore": results["LScore"],
            "Win": 1,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["LTeamID"],
            "OppTeamID": results["WTeamID"],
            "Score": results["LScore"],
            "OppScore": results["WScore"],
            "Win": 0,
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Margin"] = team_games["Score"] - team_games["OppScore"]
    return team_games.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)


def _march_team_game_rows_detailed(results: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["WTeamID"],
            "OppTeamID": results["LTeamID"],
            "Score": results["WScore"],
            "OppScore": results["LScore"],
            "Win": 1,
            "Loc": results["WLoc"].fillna("N"),
            "FGM": results["WFGM"],
            "FGA": results["WFGA"],
            "FGM3": results["WFGM3"],
            "FGA3": results["WFGA3"],
            "FTM": results["WFTM"],
            "FTA": results["WFTA"],
            "OR": results["WOR"],
            "DR": results["WDR"],
            "Ast": results["WAst"],
            "TO": results["WTO"],
            "Stl": results["WStl"],
            "Blk": results["WBlk"],
            "PF": results["WPF"],
            "OppFGM": results["LFGM"],
            "OppFGA": results["LFGA"],
            "OppFGM3": results["LFGM3"],
            "OppFGA3": results["LFGA3"],
            "OppFTM": results["LFTM"],
            "OppFTA": results["LFTA"],
            "OppOR": results["LOR"],
            "OppDR": results["LDR"],
            "OppAst": results["LAst"],
            "OppTO": results["LTO"],
            "OppStl": results["LStl"],
            "OppBlk": results["LBlk"],
            "OppPF": results["LPF"],
        }
    )
    loser_loc = results["WLoc"].map({"H": "A", "A": "H"}).fillna("N")
    losers = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["LTeamID"],
            "OppTeamID": results["WTeamID"],
            "Score": results["LScore"],
            "OppScore": results["WScore"],
            "Win": 0,
            "Loc": loser_loc,
            "FGM": results["LFGM"],
            "FGA": results["LFGA"],
            "FGM3": results["LFGM3"],
            "FGA3": results["LFGA3"],
            "FTM": results["LFTM"],
            "FTA": results["LFTA"],
            "OR": results["LOR"],
            "DR": results["LDR"],
            "Ast": results["LAst"],
            "TO": results["LTO"],
            "Stl": results["LStl"],
            "Blk": results["LBlk"],
            "PF": results["LPF"],
            "OppFGM": results["WFGM"],
            "OppFGA": results["WFGA"],
            "OppFGM3": results["WFGM3"],
            "OppFGA3": results["WFGA3"],
            "OppFTM": results["WFTM"],
            "OppFTA": results["WFTA"],
            "OppOR": results["WOR"],
            "OppDR": results["WDR"],
            "OppAst": results["WAst"],
            "OppTO": results["WTO"],
            "OppStl": results["WStl"],
            "OppBlk": results["WBlk"],
            "OppPF": results["WPF"],
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Margin"] = team_games["Score"] - team_games["OppScore"]
    team_games["Possessions"] = (
        team_games["FGA"] - team_games["OR"] + team_games["TO"] + (0.475 * team_games["FTA"])
    ).clip(lower=1.0)
    team_games["OppPossessions"] = (
        team_games["OppFGA"] - team_games["OppOR"] + team_games["OppTO"] + (0.475 * team_games["OppFTA"])
    ).clip(lower=1.0)
    team_games["Pace"] = ((team_games["Possessions"] + team_games["OppPossessions"]) / 2.0).clip(lower=1.0)
    team_games["OffEff"] = 100.0 * team_games["Score"] / team_games["Pace"]
    team_games["DefEff"] = 100.0 * team_games["OppScore"] / team_games["Pace"]
    team_games["NetEff"] = team_games["OffEff"] - team_games["DefEff"]
    team_games["eFG"] = (team_games["FGM"] + (0.5 * team_games["FGM3"])) / team_games["FGA"].clip(lower=1.0)
    team_games["OppEfg"] = (
        team_games["OppFGM"] + (0.5 * team_games["OppFGM3"])
    ) / team_games["OppFGA"].clip(lower=1.0)
    team_games["TOVRate"] = team_games["TO"] / team_games["Possessions"]
    team_games["OppTOVRate"] = team_games["OppTO"] / team_games["OppPossessions"]
    team_games["ORBRate"] = team_games["OR"] / (team_games["OR"] + team_games["OppDR"]).clip(lower=1.0)
    team_games["OppORBRate"] = team_games["OppOR"] / (team_games["OppOR"] + team_games["DR"]).clip(lower=1.0)
    team_games["FTRate"] = team_games["FTA"] / team_games["FGA"].clip(lower=1.0)
    team_games["OppFTRate"] = team_games["OppFTA"] / team_games["OppFGA"].clip(lower=1.0)
    team_games["AstRate"] = team_games["Ast"] / team_games["FGM"].clip(lower=1.0)
    team_games["OppAstRate"] = team_games["OppAst"] / team_games["OppFGM"].clip(lower=1.0)
    team_games["IsHome"] = team_games["Loc"].eq("H").astype(int)
    team_games["IsAway"] = team_games["Loc"].eq("A").astype(int)
    team_games["IsNeutral"] = team_games["Loc"].eq("N").astype(int)
    return team_games.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)


def _march_elo_features(results: pd.DataFrame) -> pd.DataFrame:
    ratings_rows: list[dict[str, float]] = []
    for season, season_games in results.sort_values(["Season", "DayNum"]).groupby("Season", sort=True):
        season_ratings: dict[int, float] = {}
        for game in season_games.itertuples(index=False):
            winner_rating = season_ratings.get(int(game.WTeamID), 1500.0)
            loser_rating = season_ratings.get(int(game.LTeamID), 1500.0)
            expected_winner = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
            margin = max(int(game.WScore) - int(game.LScore), 1)
            k_factor = 20.0 * min(2.5, 1.0 + (margin - 1) / 25.0)
            season_ratings[int(game.WTeamID)] = winner_rating + k_factor * (1.0 - expected_winner)
            season_ratings[int(game.LTeamID)] = loser_rating + k_factor * (0.0 - (1.0 - expected_winner))

        for team_id, rating in season_ratings.items():
            ratings_rows.append({"Season": season, "TeamID": team_id, "elo": rating})
    return pd.DataFrame(ratings_rows)


def _march_massey_features(massey: pd.DataFrame) -> pd.DataFrame:
    if massey.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])

    working = massey[["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]].copy()
    working["season_latest_day"] = working.groupby("Season")["RankingDayNum"].transform("max")

    latest = working.loc[working["RankingDayNum"] == working["season_latest_day"]]
    latest_features = latest.groupby(["Season", "TeamID"]).agg(
        massey_latest_mean=("OrdinalRank", "mean"),
        massey_latest_median=("OrdinalRank", "median"),
        massey_latest_best=("OrdinalRank", "min"),
        massey_latest_worst=("OrdinalRank", "max"),
        massey_latest_std=("OrdinalRank", "std"),
        massey_latest_count=("OrdinalRank", "size"),
    ).reset_index()

    recent = working.loc[working["RankingDayNum"] >= (working["season_latest_day"] - 7)]
    recent_features = recent.groupby(["Season", "TeamID"]).agg(
        massey_recent_mean=("OrdinalRank", "mean"),
        massey_recent_best=("OrdinalRank", "min"),
        massey_recent_std=("OrdinalRank", "std"),
    ).reset_index()

    previous = working.loc[
        (working["RankingDayNum"] >= (working["season_latest_day"] - 14))
        & (working["RankingDayNum"] < (working["season_latest_day"] - 7))
    ]
    previous_features = previous.groupby(["Season", "TeamID"]).agg(
        massey_prev_mean=("OrdinalRank", "mean"),
    ).reset_index()

    features = latest_features.merge(recent_features, on=["Season", "TeamID"], how="left")
    features = features.merge(previous_features, on=["Season", "TeamID"], how="left")
    features["massey_trend"] = features["massey_prev_mean"] - features["massey_recent_mean"]
    return features


def _march_training_weights(seasons: pd.Series) -> np.ndarray:
    season_values = seasons.astype(float).to_numpy()
    if len(season_values) == 0:
        return np.array([], dtype=float)
    min_season = float(np.min(season_values))
    max_season = float(np.max(season_values))
    if max_season <= min_season:
        return np.ones(len(season_values), dtype=float)
    scaled = (season_values - min_season) / (max_season - min_season)
    return 1.0 + (1.5 * scaled)


def _march_fit_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray) -> Any:
    if isinstance(model, Pipeline):
        estimator_name = model.steps[-1][0]
        try:
            model.fit(x_train, y_train, **{f"{estimator_name}__sample_weight": sample_weight})
            return model
        except TypeError:
            pass
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return model
    except TypeError:
        model.fit(x_train, y_train)
        return model


def _march_team_features(results: pd.DataFrame, seeds: pd.DataFrame, massey: pd.DataFrame | None = None) -> pd.DataFrame:
    team_games = _march_team_game_rows_detailed(results)
    recent = (
        team_games.groupby(["Season", "TeamID"], group_keys=False)
        .tail(10)
        .groupby(["Season", "TeamID"])
        .agg(
            recent_win_pct=("Win", "mean"),
            recent_margin=("Margin", "mean"),
            recent_net_eff=("NetEff", "mean"),
            recent_off_eff=("OffEff", "mean"),
            recent_def_eff=("DefEff", "mean"),
            recent_efg=("eFG", "mean"),
            recent_tov_rate=("TOVRate", "mean"),
            recent_orb_rate=("ORBRate", "mean"),
        )
        .reset_index()
    )
    season_features = (
        team_games.groupby(["Season", "TeamID"])
        .agg(
            games=("Win", "size"),
            win_pct=("Win", "mean"),
            avg_score=("Score", "mean"),
            avg_allowed=("OppScore", "mean"),
            avg_margin=("Margin", "mean"),
            avg_pace=("Pace", "mean"),
            off_eff=("OffEff", "mean"),
            def_eff=("DefEff", "mean"),
            net_eff=("NetEff", "mean"),
            efg=("eFG", "mean"),
            opp_efg=("OppEfg", "mean"),
            tov_rate=("TOVRate", "mean"),
            opp_tov_rate=("OppTOVRate", "mean"),
            orb_rate=("ORBRate", "mean"),
            opp_orb_rate=("OppORBRate", "mean"),
            ft_rate=("FTRate", "mean"),
            opp_ft_rate=("OppFTRate", "mean"),
            ast_rate=("AstRate", "mean"),
            opp_ast_rate=("OppAstRate", "mean"),
            home_share=("IsHome", "mean"),
            away_share=("IsAway", "mean"),
            neutral_share=("IsNeutral", "mean"),
        )
        .reset_index()
    )
    neutral = (
        team_games.loc[team_games["IsNeutral"] == 1]
        .groupby(["Season", "TeamID"])
        .agg(
            neutral_win_pct=("Win", "mean"),
            neutral_margin=("Margin", "mean"),
        )
        .reset_index()
    )
    close = (
        team_games.loc[team_games["Margin"].abs() <= 5]
        .groupby(["Season", "TeamID"])
        .agg(
            close_games=("Win", "size"),
            close_win_pct=("Win", "mean"),
        )
        .reset_index()
    )
    elo = _march_elo_features(results)
    features = season_features.merge(recent, on=["Season", "TeamID"], how="left")
    features = features.merge(neutral, on=["Season", "TeamID"], how="left")
    features = features.merge(close, on=["Season", "TeamID"], how="left")
    features = features.merge(elo, on=["Season", "TeamID"], how="left")

    seed_features = seeds.copy()
    seed_features["seed"] = seed_features["Seed"].map(_march_seed_number)
    seed_features = seed_features[["Season", "TeamID", "seed"]]
    features = features.merge(seed_features, on=["Season", "TeamID"], how="left")
    features["seed_missing"] = features["seed"].isna().astype(int)
    features["seed"] = features["seed"].fillna(20.0)
    features["elo"] = features["elo"].fillna(1500.0)
    features["recent_win_pct"] = features["recent_win_pct"].fillna(features["win_pct"])
    features["recent_margin"] = features["recent_margin"].fillna(features["avg_margin"])
    features["recent_net_eff"] = features["recent_net_eff"].fillna(features["net_eff"])
    features["recent_off_eff"] = features["recent_off_eff"].fillna(features["off_eff"])
    features["recent_def_eff"] = features["recent_def_eff"].fillna(features["def_eff"])
    features["recent_efg"] = features["recent_efg"].fillna(features["efg"])
    features["recent_tov_rate"] = features["recent_tov_rate"].fillna(features["tov_rate"])
    features["recent_orb_rate"] = features["recent_orb_rate"].fillna(features["orb_rate"])
    features["neutral_win_pct"] = features["neutral_win_pct"].fillna(features["win_pct"])
    features["neutral_margin"] = features["neutral_margin"].fillna(features["avg_margin"])
    features["close_games"] = features["close_games"].fillna(0.0)
    features["close_win_pct"] = features["close_win_pct"].fillna(features["win_pct"])

    opp_base = features[["Season", "TeamID", "win_pct", "net_eff", "off_eff", "def_eff", "elo"]].rename(
        columns={
            "TeamID": "OppTeamID",
            "win_pct": "sos_win_pct",
            "net_eff": "sos_net_eff",
            "off_eff": "sos_off_eff",
            "def_eff": "sos_def_eff",
            "elo": "sos_elo",
        }
    )
    schedule_strength = (
        team_games[["Season", "TeamID", "OppTeamID"]]
        .merge(opp_base, on=["Season", "OppTeamID"], how="left")
        .groupby(["Season", "TeamID"])
        .agg(
            sos_win_pct=("sos_win_pct", "mean"),
            sos_net_eff=("sos_net_eff", "mean"),
            sos_off_eff=("sos_off_eff", "mean"),
            sos_def_eff=("sos_def_eff", "mean"),
            sos_elo=("sos_elo", "mean"),
        )
        .reset_index()
    )
    features = features.merge(schedule_strength, on=["Season", "TeamID"], how="left")
    features["net_eff_vs_schedule"] = features["net_eff"] - features["sos_net_eff"]
    features["elo_vs_schedule"] = features["elo"] - features["sos_elo"]

    if massey is not None and not massey.empty:
        features = features.merge(_march_massey_features(massey), on=["Season", "TeamID"], how="left")

    numeric_cols = [col for col in features.columns if col not in {"Season", "TeamID"}]
    for col in numeric_cols:
        season_medians = features.groupby("Season")[col].transform("median")
        features[col] = features[col].fillna(season_medians)
        if features[col].isna().any():
            fallback = features[col].median()
            features[col] = features[col].fillna(0.0 if pd.isna(fallback) else float(fallback))

    features["has_massey"] = features.get("massey_latest_count", pd.Series(0.0, index=features.index)).gt(0).astype(int)
    return features


def _march_submission_pairs(sample: pd.DataFrame) -> pd.DataFrame:
    parsed = sample["ID"].astype(str).str.split("_", expand=True)
    if parsed.shape[1] != 3:
        raise ValueError("Unexpected March Mania submission ID format.")
    return pd.DataFrame(
        {
            "ID": sample["ID"].astype(str),
            "Season": parsed[0].astype(int),
            "Team1": parsed[1].astype(int),
            "Team2": parsed[2].astype(int),
        }
    )


def _march_matchups(
    games: pd.DataFrame,
    features: pd.DataFrame,
    *,
    include_target: bool,
) -> pd.DataFrame:
    feature_map = features.set_index(["Season", "TeamID"]).to_dict("index")
    base_cols = [col for col in features.columns if col not in {"Season", "TeamID"}]
    rows: list[dict[str, Any]] = []

    for game in games.itertuples(index=False):
        season = int(game.Season)
        if hasattr(game, "WTeamID") and hasattr(game, "LTeamID"):
            team_a = int(game.WTeamID)
            team_b = int(game.LTeamID)
        else:
            team_a = int(game.Team1)
            team_b = int(game.Team2)
        team1, team2 = sorted((team_a, team_b))
        feat_1 = feature_map.get((season, team1))
        feat_2 = feature_map.get((season, team2))
        if feat_1 is None or feat_2 is None:
            continue

        row: dict[str, Any] = {"Season": season, "Team1": team1, "Team2": team2}
        row["is_women"] = 1 if team1 >= 3000 else 0
        if include_target:
            row["target"] = 1 if team1 == team_a else 0
        for col in base_cols:
            value_1 = float(feat_1[col])
            value_2 = float(feat_2[col])
            row[f"{col}_1"] = value_1
            row[f"{col}_2"] = value_2
            row[f"{col}_diff"] = value_1 - value_2
        for col in ("seed", "elo", "net_eff", "recent_net_eff", "massey_latest_mean"):
            diff_key = f"{col}_diff"
            if diff_key in row:
                row[f"{col}_abs_diff"] = abs(row[diff_key])
        if "elo_diff" in row:
            row["elo_win_prob_1"] = 1.0 / (1.0 + (10.0 ** (-row["elo_diff"] / 400.0)))
        if "seed_diff" in row:
            row["seed_win_prob_1"] = 1.0 / (1.0 + np.exp(row["seed_diff"] / 1.5))
        if "net_eff_diff" in row:
            row["net_eff_win_prob_1"] = 1.0 / (1.0 + np.exp(-row["net_eff_diff"] / 5.0))
        if "recent_net_eff_diff" in row:
            row["recent_net_eff_win_prob_1"] = 1.0 / (1.0 + np.exp(-row["recent_net_eff_diff"] / 5.0))
        if "massey_latest_mean_diff" in row:
            row["massey_win_prob_1"] = 1.0 / (1.0 + np.exp(row["massey_latest_mean_diff"] / 7.5))
        rows.append(row)

    return pd.DataFrame(rows)


def _march_build_models() -> dict[str, Any]:
    return {
        "lr": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=4000, C=0.8)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.035,
                        max_depth=4,
                        max_iter=450,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def benchmark_march_mania(data_dir: Path, _folds: int, write_submission: bool) -> LabResult:
    regular_season = pd.concat(
        [
            pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv"),
            pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv"),
        ],
        ignore_index=True,
    )
    tournament = pd.concat(
        [
            pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv"),
            pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv"),
        ],
        ignore_index=True,
    )
    seeds = pd.concat(
        [
            pd.read_csv(data_dir / "MNCAATourneySeeds.csv"),
            pd.read_csv(data_dir / "WNCAATourneySeeds.csv"),
        ],
        ignore_index=True,
    )
    massey = pd.read_csv(data_dir / "MMasseyOrdinals.csv")

    features = _march_team_features(regular_season, seeds, massey)
    train_df = _march_matchups(tournament, features, include_target=True)
    if train_df.empty:
        raise SystemExit("Failed to build March Mania training rows from downloaded competition files.")

    feature_cols = [col for col in train_df.columns if col not in {"target", "Team1", "Team2"}]
    holdout_seasons = sorted(season for season in train_df["Season"].unique() if season >= 2021)
    if not holdout_seasons:
        holdout_seasons = sorted(train_df["Season"].unique())[-5:]

    model_defs = _march_build_models()
    season_predictions: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {name: {} for name in model_defs}
    for season in holdout_seasons:
        train_mask = train_df["Season"] < season
        valid_mask = train_df["Season"] == season
        if int(train_mask.sum()) == 0 or int(valid_mask.sum()) == 0:
            continue
        x_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "target"].astype(int)
        x_valid = train_df.loc[valid_mask, feature_cols]
        y_valid = train_df.loc[valid_mask, "target"].astype(int)
        sample_weight = _march_training_weights(train_df.loc[train_mask, "Season"])
        for name, model in _march_build_models().items():
            model = _march_fit_model(model, x_train, y_train, sample_weight)
            probs = model.predict_proba(x_valid)[:, 1]
            season_predictions[name][season] = (y_valid.to_numpy(), probs)

    benchmarks: list[dict[str, Any]] = []
    for name, rows in season_predictions.items():
        scores = [(season, brier_score_loss(y_true, probs)) for season, (y_true, probs) in rows.items()]
        if scores:
            benchmarks.append({"model": name, "score": round(float(np.mean([score for _, score in scores])), 5)})

    model_names = list(model_defs)
    for combo_size in range(2, len(model_names) + 1):
        for combo in combinations(model_names, combo_size):
            common_seasons = sorted(set.intersection(*(set(season_predictions[name]) for name in combo)))
            if not common_seasons:
                continue
            ensemble_scores = []
            for season in common_seasons:
                y_true = season_predictions[combo[0]][season][0]
                blended = np.mean([season_predictions[name][season][1] for name in combo], axis=0)
                ensemble_scores.append((season, brier_score_loss(y_true, blended)))
            benchmarks.append(
                {
                    "model": f"{'_'.join(combo)}_ensemble",
                    "score": round(float(np.mean([score for _, score in ensemble_scores])), 5),
                    "members": list(combo),
                }
            )

    if not benchmarks:
        raise SystemExit("March Mania benchmark did not produce any holdout scores.")

    best = min(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        sample_path = (
            data_dir / "SampleSubmissionStage2.csv"
            if (data_dir / "SampleSubmissionStage2.csv").exists()
            else data_dir / "SampleSubmissionStage1.csv"
        )
        sample = pd.read_csv(sample_path)
        submission_pairs = _march_submission_pairs(sample)
        submission_features = _march_matchups(submission_pairs, features, include_target=False)
        if submission_features.empty:
            raise SystemExit("Failed to build March Mania submission rows.")

        train_x = train_df[feature_cols]
        train_y = train_df["target"].astype(int)
        submit_x = submission_features[feature_cols]
        sample_weight = _march_training_weights(train_df["Season"])
        models = _march_build_models()
        fitted: dict[str, Any] = {}
        for name, model in models.items():
            fitted[name] = _march_fit_model(model, train_x, train_y, sample_weight)

        if best.get("members"):
            preds = np.mean([fitted[name].predict_proba(submit_x)[:, 1] for name in best["members"]], axis=0)
        else:
            preds = fitted[best["model"]].predict_proba(submit_x)[:, 1]

        submission_path = _submission_dir("march-machine-learning-mania-2026") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame({"ID": submission_pairs["ID"], "Pred": preds}).to_csv(submission_path, index=False)

    return LabResult(
        competition="march-machine-learning-mania-2026",
        metric_name="brier",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )


BENCHMARKS = {
    "deep-past-initiative-machine-translation": benchmark_deep_past,
    "house-prices-advanced-regression-techniques": benchmark_house_prices,
    "march-machine-learning-mania-2026": benchmark_march_mania,
    "playground-series-s6e3": benchmark_playground_telco,
    "store-sales-time-series-forecasting": benchmark_store_sales,
    "titanic": benchmark_titanic,
    "spaceship-titanic": benchmark_spaceship,
    "nlp-getting-started": benchmark_nlp,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local competition benchmarks and generate Kaggle submissions.")
    parser.add_argument("slug", choices=sorted(BENCHMARKS))
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--write-submission", action="store_true", help="Write the best local submission CSV")
    parser.add_argument("--submit", action="store_true", help="Submit the generated CSV to Kaggle")
    parser.add_argument("--force-download", action="store_true", help="Re-download competition data even if cached locally")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = _ensure_data(args.slug, force_download=args.force_download)
    bench_fn = BENCHMARKS[args.slug]
    result = bench_fn(data_dir, args.cv_folds, write_submission=(args.write_submission or args.submit))
    _save_summary(result)
    _print_benchmarks(result)

    if args.submit:
        if result.submission_path is None:
            raise SystemExit("No submission file was generated.")
        message = f"Local {result.best_model} baseline via competition-lab ({result.metric_name}={result.best_score:.5f})"
        _submit(args.slug, result.submission_path, message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
