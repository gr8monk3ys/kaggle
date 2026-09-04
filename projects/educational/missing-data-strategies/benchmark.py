"""Benchmark seven missing-data strategies across three datasets.

Produces the numbers quoted in docs/discussions/discussion-drafts.md Draft 15.
Every imputer is fit *inside* the cross-validation fold, so no test-fold
statistic leaks into training. Run:

    python projects/educational/missing-data-strategies/benchmark.py

Writes results.json next to this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 42
MISSING_RATE = 0.20
N_SPLITS = 5


def _ordinal_encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Map non-numeric columns to integer codes, preserving NaN as NaN."""
    out = frame.copy()
    for col in out.columns:
        # pandas 3 gives text columns the `str` dtype, not `object`, so ask
        # "is it numeric?" rather than comparing against a specific dtype.
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.Categorical(out[col]).codes.astype(float)
            # pandas encodes NaN as -1; restore it so imputers can see it.
            out.loc[out[col] == -1, col] = np.nan
    return out


def _inject_mcar(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Blank out `rate` of cells completely at random."""
    rng = np.random.default_rng(seed)
    return frame.mask(rng.random(frame.shape) < rate)


def load_datasets() -> list[dict]:
    """Three binary-classification tasks: one with native missingness, two injected."""
    sets = []

    # 1. Native missingness — the honest headline case.
    path = REPO / "datasets" / "mental-health-tech" / "mental_health_tech.csv"
    frame = pd.read_csv(path)
    target = (frame.pop("treatment") == "Yes").astype(int)
    features = _ordinal_encode(frame)
    sets.append(
        {
            "name": "mental-health-tech",
            "missingness": "native",
            "X": features,
            "y": target,
            "rows": len(features),
        }
    )

    # 2. sklearn bundle — reproducible with no download at all.
    bunch = load_breast_cancer(as_frame=True)
    sets.append(
        {
            "name": "breast-cancer (sklearn)",
            "missingness": f"injected MCAR @ {MISSING_RATE:.0%}",
            "X": _inject_mcar(bunch.data, MISSING_RATE, SEED),
            "y": bunch.target,
            "rows": len(bunch.data),
        }
    )

    # 3. Repo's own fraud dataset, stratified subsample so KNN/MICE stay tractable.
    path = REPO / "datasets" / "credit-card-fraud" / "credit_card_transactions.csv"
    frame = pd.read_csv(path)
    target_full = frame.pop("Class")
    numeric = frame.select_dtypes(include="number")
    pos = target_full[target_full == 1].index
    rng = np.random.default_rng(SEED)
    neg = rng.choice(target_full[target_full == 0].index, size=2500, replace=False)
    keep = np.concatenate([pos.to_numpy(), neg])
    sets.append(
        {
            "name": "credit-card-fraud (subsample)",
            "missingness": f"injected MCAR @ {MISSING_RATE:.0%}",
            "X": _inject_mcar(numeric.loc[keep].reset_index(drop=True), MISSING_RATE, SEED),
            "y": target_full.loc[keep].reset_index(drop=True),
            "rows": len(keep),
        }
    )
    return sets


def build_imputer(strategy: str):
    """Return a transformer implementing one strategy, or None for row-dropping."""
    if strategy == "drop_rows":
        return None
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "constant_-999":
        return SimpleImputer(strategy="constant", fill_value=-999)
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    if strategy == "mice":
        return IterativeImputer(max_iter=10, random_state=SEED)
    if strategy == "missing_indicator":
        return SimpleImputer(strategy="median", add_indicator=True)
    if strategy == "rf_impute":
        # Deliberately small forest: a full-size one costs ~50x more for a
        # difference well inside the fold-to-fold spread reported below.
        return IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10, max_depth=8, random_state=SEED, n_jobs=-1
            ),
            max_iter=2,
            random_state=SEED,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def build_model(family: str):
    if family == "tree":
        return HistGradientBoostingClassifier(random_state=SEED)
    return Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )


def fold_aucs(strategy, X_tr, y_tr, X_te, y_te):
    """Fit the imputer on the training fold once, then score both model families.

    Imputation is the expensive half (MICE and RF-imputation dominate), so it is
    shared across the two models rather than repeated per family.
    """
    imputer = build_imputer(strategy)

    if imputer is None:
        # "Drop rows" is only definable on the training side: at predict time
        # every test row must receive a score, so it falls back to the median.
        complete = X_tr.notna().all(axis=1)
        if complete.sum() < 20 or y_tr[complete].nunique() < 2:
            return {}  # nothing survives the drop
        fallback = SimpleImputer(strategy="median").fit(X_tr[complete])
        tr_X, tr_y = fallback.transform(X_tr[complete]), y_tr[complete]
        te_X = fallback.transform(X_te)
    else:
        fitted = imputer.fit(X_tr)
        tr_X, tr_y = fitted.transform(X_tr), y_tr
        te_X = fitted.transform(X_te)

    scores = {}
    for family in ("tree", "linear"):
        model = build_model(family)
        model.fit(tr_X, tr_y)
        scores[family] = roc_auc_score(y_te, model.predict_proba(te_X)[:, 1])
    return scores


STRATEGIES = [
    "drop_rows",
    "median",
    "constant_-999",
    "knn",
    "mice",
    "missing_indicator",
    "rf_impute",
]


def main() -> None:
    datasets = load_datasets()
    results = []
    timings: dict[str, list[float]] = {}

    for spec in datasets:
        X, y = spec["X"], spec["y"]
        missing_pct = float(X.isna().to_numpy().mean())
        print(f"\n=== {spec['name']}  rows={spec['rows']}  cells missing={missing_pct:.1%} ===")
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        for strategy in STRATEGIES:
            started = time.perf_counter()
            per_family = {"tree": [], "linear": []}
            for tr_idx, te_idx in splitter.split(X, y):
                scores = fold_aucs(
                    strategy, X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx], y.iloc[te_idx]
                )
                for family, auc in scores.items():
                    per_family[family].append(auc)
            elapsed = time.perf_counter() - started

            for family, folds in per_family.items():
                results.append(
                    {
                        "dataset": spec["name"],
                        "strategy": strategy,
                        "model": family,
                        "auc_mean": round(float(np.mean(folds)), 4) if folds else None,
                        "auc_std": round(float(np.std(folds)), 4) if folds else None,
                        "folds_scored": len(folds),
                    }
                )
            tree_auc = per_family["tree"]
            lin_auc = per_family["linear"]
            fmt = lambda v: f"{np.mean(v):.4f}" if v else "n/a"
            print(
                f"  {strategy:<18} tree={fmt(tree_auc)}  linear={fmt(lin_auc)}  {elapsed:6.2f}s"
            )
            timings.setdefault(strategy, []).append(round(elapsed, 2))

    payload = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "injected_missing_rate": MISSING_RATE,
        "datasets": [
            {"name": d["name"], "rows": d["rows"], "missingness": d["missingness"]}
            for d in datasets
        ],
        "seconds_per_strategy": timings,
        "results": results,
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
