#!/usr/bin/env python3
"""Benchmark local Kaggle competition baselines and optionally submit them.

Supports a small set of high-value competitions already used in this repo:
    - titanic
    - spaceship-titanic
    - nlp-getting-started

Examples
--------
    python -m kaggle_portfolio.notebooks.local_competition_lab titanic
    python -m kaggle_portfolio.notebooks.local_competition_lab titanic --write-submission
    python -m kaggle_portfolio.notebooks.local_competition_lab spaceship-titanic --submit
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


BENCHMARKS = {
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
