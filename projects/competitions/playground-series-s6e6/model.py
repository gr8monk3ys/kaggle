#!/usr/bin/env python3
"""Playground Series S6E6 — improved blend over the baseline.

Adds standard SDSS color-index features (u-g, g-r, r-i, i-z, ...) and blends a
HistGradientBoostingClassifier with XGBoost via averaged class probabilities.
Honest out-of-fold (OOF) cross-validation measures the blend, then both models
refit on all data and their test probabilities are averaged for the submission.

Usage:
    python model.py --data-dir /kaggle/input/playground-series-s6e6 --out submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

SEED = 42
TARGET = "class"
BASE_NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CATEGORICAL = ["spectral_type", "galaxy_population"]
# Standard SDSS color separators (adjacent + a few broad colors).
COLORS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("g", "i"), ("r", "z")]


def resolve_data_dir(explicit: str | None) -> Path:
    for c in [Path(explicit) if explicit else None,
              Path("/kaggle/input/playground-series-s6e6"), Path("/tmp/ps6e6"), Path(".")]:
        if c and (c / "train.csv").exists():
            return c
    raise FileNotFoundError("train.csv not found; pass --data-dir")


def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for a, b in COLORS:
        out[f"{a}_{b}"] = df[a] - df[b]
    return out


def build_matrix(df: pd.DataFrame, encoder: OrdinalEncoder, fit: bool):
    color_cols = [f"{a}_{b}" for a, b in COLORS]
    numeric = BASE_NUMERIC + color_cols
    num = df[numeric].to_numpy(dtype=float)
    cats = df[CATEGORICAL].astype(str)
    enc = encoder.fit_transform(cats) if fit else encoder.transform(cats)
    X = np.hstack([num, enc])
    cat_mask = [False] * len(numeric) + [True] * len(CATEGORICAL)
    return X, cat_mask


def hist_model(cat_mask):
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=SEED,
    )


def xgb_model(n_classes):
    return XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
        num_class=n_classes, n_jobs=-1, random_state=SEED, eval_metric="mlogloss",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data dir: {data_dir}")
    train = add_colors(pd.read_csv(data_dir / "train.csv"))
    test = add_colors(pd.read_csv(data_dir / "test.csv"))
    print(f"train={train.shape} test={test.shape}")

    le = LabelEncoder()
    y = le.fit_transform(train[TARGET].to_numpy())
    n_classes = len(le.classes_)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X, cat_mask = build_matrix(train, encoder, fit=True)
    X_test, _ = build_matrix(test, encoder, fit=False)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), n_classes))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        hm = hist_model(cat_mask).fit(X[tr], y[tr])
        xm = xgb_model(n_classes).fit(X[tr], y[tr])
        proba = 0.5 * hm.predict_proba(X[va]) + 0.5 * xm.predict_proba(X[va])
        oof[va] = proba
        fa = accuracy_score(y[va], proba.argmax(1))
        ff = f1_score(y[va], proba.argmax(1), average="macro")
        print(f"  fold {fold}: blend acc={fa:.5f} macro_f1={ff:.5f}")

    oof_pred = oof.argmax(1)
    print(f"OOF blend accuracy = {accuracy_score(y, oof_pred):.5f}")
    print(f"OOF blend macro-F1 = {f1_score(y, oof_pred, average='macro'):.5f}")

    # Refit on all data and average test probabilities.
    hm = hist_model(cat_mask).fit(X, y)
    xm = xgb_model(n_classes).fit(X, y)
    test_proba = 0.5 * hm.predict_proba(X_test) + 0.5 * xm.predict_proba(X_test)
    preds = le.inverse_transform(test_proba.argmax(1))

    sub = pd.DataFrame({"id": test["id"], TARGET: preds})
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(sub):,} rows)")
    print(sub[TARGET].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
