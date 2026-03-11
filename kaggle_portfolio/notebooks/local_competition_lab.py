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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

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

    group_id = combined["PassengerId"].astype(str).str.split("_").str[0]
    combined["GroupId"] = group_id
    combined["GroupSize"] = group_id.map(group_id.value_counts())
    cabin = combined["Cabin"].fillna("Unknown/0/U").astype(str).str.split("/", expand=True)
    combined["Deck"] = cabin[0].fillna("Unknown")
    combined["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce").fillna(-1)
    combined["Side"] = cabin[2].fillna("Unknown")
    combined["Surname"] = combined["Name"].fillna("Unknown Unknown").astype(str).str.split().str[-1]
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        combined[col] = combined[col].fillna(0.0)
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())
    combined["TotalSpend"] = combined[spend_cols].sum(axis=1)
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["CryoSleep"] = combined["CryoSleep"].fillna(False)
    combined["VIP"] = combined["VIP"].fillna(False)
    combined["HomePlanet"] = combined["HomePlanet"].fillna("Unknown")
    combined["Destination"] = combined["Destination"].fillna("Unknown")

    features = [
        "HomePlanet",
        "Destination",
        "CryoSleep",
        "VIP",
        "Deck",
        "Side",
        "Age",
        "CabinNum",
        "GroupSize",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpend",
        "NoSpend",
    ]
    engineered = combined[features].copy()
    engineered["CryoSleep"] = engineered["CryoSleep"].astype(int)
    engineered["VIP"] = engineered["VIP"].astype(int)
    train_x = engineered.iloc[: len(train)].reset_index(drop=True)
    test_x = engineered.iloc[len(train) :].reset_index(drop=True)
    return train_x, test_x


def _spaceship_catboost(
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
            depth=8,
            iterations=700,
            learning_rate=0.04,
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
        depth=8,
        iterations=700,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    final_model.fit(train_x, y, cat_features=cat_idx, verbose=False)
    submission_preds = final_model.predict(test_x).reshape(-1).astype(bool)
    return float(np.mean(scores)), submission_preds


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
        cat_score, cat_preds = _spaceship_catboost(train_x.copy(), test_x.copy(), y, folds)
        benchmarks.append({"model": "catboost", "score": round(cat_score, 5)})
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


def benchmark_playground_telco(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Churn"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(train["Churn"]).astype(int)
    train_x, test_x = _playground_prepare_features(train, test)
    skf = StratifiedKFold(n_splits=min(max(3, folds), 5), shuffle=True, random_state=RANDOM_STATE)

    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}

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
        lgb_score = cross_val_score(lgb_model, train_x, y, cv=skf, scoring="roc_auc", n_jobs=1)
        benchmarks.append({"model": "lightgbm", "score": round(float(lgb_score.mean()), 5)})
        lgb_model.fit(train_x, y)
        trained_predictions["lightgbm"] = lgb_model.predict_proba(test_x)[:, 1]
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
        xgb_score = cross_val_score(xgb_model, train_x, y, cv=skf, scoring="roc_auc", n_jobs=1)
        benchmarks.append({"model": "xgboost", "score": round(float(xgb_score.mean()), 5)})
        xgb_model.fit(train_x, y)
        trained_predictions["xgboost"] = xgb_model.predict_proba(test_x)[:, 1]
    except ImportError:
        pass

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
    combined["TotalSF"] = combined[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].sum(axis=1)
    combined["TotalBath"] = (
        combined["FullBath"].fillna(0)
        + 0.5 * combined["HalfBath"].fillna(0)
        + combined["BsmtFullBath"].fillna(0)
        + 0.5 * combined["BsmtHalfBath"].fillna(0)
    )
    combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
    combined["RemodAge"] = combined["YrSold"] - combined["YearRemodAdd"]
    combined["HasGarage"] = combined["GarageArea"].fillna(0).gt(0).astype(int)
    combined["HasBsmt"] = combined["TotalBsmtSF"].fillna(0).gt(0).astype(int)
    combined["HasPool"] = combined["PoolArea"].fillna(0).gt(0).astype(int)
    combined["HasFireplace"] = combined["Fireplaces"].fillna(0).gt(0).astype(int)
    combined["HasSecondFloor"] = combined["2ndFlrSF"].fillna(0).gt(0).astype(int)
    combined["TotalPorchSF"] = (
        combined["WoodDeckSF"].fillna(0)
        + combined["OpenPorchSF"].fillna(0)
        + combined["EnclosedPorch"].fillna(0)
        + combined["3SsnPorch"].fillna(0)
        + combined["ScreenPorch"].fillna(0)
    )
    combined["QualSF"] = combined["OverallQual"].fillna(0) * combined["GrLivArea"].fillna(0)
    combined["TotalHomeQuality"] = combined["OverallQual"].fillna(0) + combined["OverallCond"].fillna(0)

    for col in ("LotArea", "GrLivArea", "TotalSF", "1stFlrSF", "2ndFlrSF", "MasVnrArea", "TotalBsmtSF"):
        if col in combined:
            combined[f"log_{col.lower()}"] = np.log1p(combined[col].clip(lower=0))

    train_x = combined.iloc[: len(train)].reset_index(drop=True)
    test_x = combined.iloc[len(train) :].reset_index(drop=True)
    return train_x, test_x


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

    cv = KFold(n_splits=min(max(3, folds), 5), shuffle=True, random_state=RANDOM_STATE)
    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}

    elastic = Pipeline(
        [
            ("scale", RobustScaler()),
            ("model", ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=20000, random_state=RANDOM_STATE)),
        ]
    )
    elastic_score = -cross_val_score(
        elastic,
        dense_train,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    benchmarks.append({"model": "elasticnet", "score": round(float(elastic_score.mean()), 5)})
    elastic.fit(dense_train, y)
    elastic_test_pred = np.expm1(elastic.predict(dense_test)).clip(min=0)
    trained_predictions["elasticnet"] = elastic_test_pred

    xgb_oof: np.ndarray | None = None
    xgb_test_pred: np.ndarray | None = None
    try:
        import lightgbm as lgb

        lgb_model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_score = -cross_val_score(
            lgb_model,
            dense_train,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        benchmarks.append({"model": "lightgbm", "score": round(float(lgb_score.mean()), 5)})
        lgb_model.fit(dense_train, y)
        trained_predictions["lightgbm"] = np.expm1(lgb_model.predict(dense_test)).clip(min=0)
    except ImportError:
        pass

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_score = -cross_val_score(
            xgb_model,
            dense_train,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        benchmarks.append({"model": "xgboost", "score": round(float(xgb_score.mean()), 5)})
        xgb_model.fit(dense_train, y)
        xgb_test_pred = np.expm1(xgb_model.predict(dense_test)).clip(min=0)
        trained_predictions["xgboost"] = xgb_test_pred
        xgb_oof = cross_val_predict(xgb_model, dense_train, y, cv=cv, n_jobs=1)
    except ImportError:
        pass

    if xgb_oof is not None and xgb_test_pred is not None:
        elastic_oof = cross_val_predict(elastic, dense_train, y, cv=cv, n_jobs=1)
        blend_oof = 0.5 * elastic_oof + 0.5 * xgb_oof
        blend_rmse = float(np.sqrt(np.mean((blend_oof - y.to_numpy()) ** 2)))
        benchmarks.append({"model": "elastic_xgb_blend", "score": round(blend_rmse, 5)})
        trained_predictions["elastic_xgb_blend"] = 0.5 * elastic_test_pred + 0.5 * xgb_test_pred

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
    best = min(benchmarks, key=lambda row: row["score"])

    submission_path = None
    if write_submission:
        submission_frame = _store_sales_prediction_frame(train, test)
        submission_path = _submission_dir("store-sales-time-series-forecasting") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame({"id": test["id"], "sales": submission_frame[best["model"]].clip(lower=0)}).to_csv(
            submission_path,
            index=False,
        )

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


def _deep_past_sentence_rows(sentences: pd.DataFrame, published_row: pd.Series) -> pd.DataFrame:
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
    published = pd.read_csv(data_dir / "published_texts.csv")
    sentences = pd.read_csv(data_dir / "Sentences_Oare_FirstWord_LinNum.csv")

    query = " ".join(test["transliteration"].astype(str))
    published_idx, published_score = _deep_past_best_match(published["transliteration"], query)
    published_row = published.iloc[published_idx]
    sentence_rows = _deep_past_sentence_rows(sentences, published_row)
    sentence_predictions = _deep_past_assign_sentences_to_rows(test, sentence_rows) if not sentence_rows.empty else []
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


def _march_team_features(results: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    team_games = _march_team_game_rows(results)
    recent = (
        team_games.groupby(["Season", "TeamID"], group_keys=False)
        .tail(10)
        .groupby(["Season", "TeamID"])
        .agg(
            recent_win_pct=("Win", "mean"),
            recent_margin=("Margin", "mean"),
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
        )
        .reset_index()
    )
    elo = _march_elo_features(results)
    features = season_features.merge(recent, on=["Season", "TeamID"], how="left")
    features = features.merge(elo, on=["Season", "TeamID"], how="left")

    seed_features = seeds.copy()
    seed_features["seed"] = seed_features["Seed"].map(_march_seed_number)
    seed_features = seed_features[["Season", "TeamID", "seed"]]
    features = features.merge(seed_features, on=["Season", "TeamID"], how="left")
    features["seed"] = features["seed"].fillna(20.0)
    features["elo"] = features["elo"].fillna(1500.0)
    features["recent_win_pct"] = features["recent_win_pct"].fillna(features["win_pct"])
    features["recent_margin"] = features["recent_margin"].fillna(features["avg_margin"])
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
    base_cols = [
        "games",
        "win_pct",
        "avg_score",
        "avg_allowed",
        "avg_margin",
        "recent_win_pct",
        "recent_margin",
        "elo",
        "seed",
    ]
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
        if include_target:
            row["target"] = 1 if team1 == team_a else 0
        for col in base_cols:
            value_1 = float(feat_1[col])
            value_2 = float(feat_2[col])
            row[f"{col}_1"] = value_1
            row[f"{col}_2"] = value_2
            row[f"{col}_diff"] = value_1 - value_2
        rows.append(row)

    return pd.DataFrame(rows)


def _march_build_models() -> dict[str, Any]:
    return {
        "lr": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=1.5)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def benchmark_march_mania(data_dir: Path, _folds: int, write_submission: bool) -> LabResult:
    regular_season = pd.concat(
        [
            pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv"),
            pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv"),
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

    features = _march_team_features(regular_season, seeds)
    train_df = _march_matchups(tournament, features, include_target=True)
    if train_df.empty:
        raise SystemExit("Failed to build March Mania training rows from downloaded competition files.")

    feature_cols = [col for col in train_df.columns if col not in {"target", "Team1", "Team2"}]
    holdout_seasons = sorted(season for season in train_df["Season"].unique() if season >= 2021)
    if not holdout_seasons:
        holdout_seasons = sorted(train_df["Season"].unique())[-5:]

    season_predictions: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {"lr": [], "hgb": []}
    for season in holdout_seasons:
        train_mask = train_df["Season"] < season
        valid_mask = train_df["Season"] == season
        if int(train_mask.sum()) == 0 or int(valid_mask.sum()) == 0:
            continue
        x_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "target"].astype(int)
        x_valid = train_df.loc[valid_mask, feature_cols]
        y_valid = train_df.loc[valid_mask, "target"].astype(int)
        for name, model in _march_build_models().items():
            model.fit(x_train, y_train)
            probs = model.predict_proba(x_valid)[:, 1]
            season_predictions[name].append((season, y_valid.to_numpy(), probs))

    benchmarks: list[dict[str, Any]] = []
    for name, rows in season_predictions.items():
        scores = [(season, brier_score_loss(y_true, probs)) for season, y_true, probs in rows]
        if scores:
            benchmarks.append({"model": name, "score": round(float(np.mean([score for _, score in scores])), 5)})

    if season_predictions["lr"] and season_predictions["hgb"]:
        ensemble_scores = []
        for (season_lr, y_lr, pred_lr), (season_hgb, _y_hgb, pred_hgb) in zip(
            season_predictions["lr"], season_predictions["hgb"]
        ):
            if season_lr != season_hgb:
                continue
            ensemble_scores.append((season_lr, brier_score_loss(y_lr, (pred_lr + pred_hgb) / 2.0)))
        if ensemble_scores:
            benchmarks.append(
                {
                    "model": "lr_hgb_ensemble",
                    "score": round(float(np.mean([score for _, score in ensemble_scores])), 5),
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
        models = _march_build_models()
        fitted: dict[str, Any] = {}
        for name, model in models.items():
            model.fit(train_x, train_y)
            fitted[name] = model

        if best["model"] == "lr_hgb_ensemble":
            preds = (fitted["lr"].predict_proba(submit_x)[:, 1] + fitted["hgb"].predict_proba(submit_x)[:, 1]) / 2.0
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
