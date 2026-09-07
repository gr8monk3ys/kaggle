#!/usr/bin/env python3
"""Playground Series S6E6 — stellar object classification baseline.

Multi-class classification (GALAXY / QSO / STAR) from photometric features.
A HistGradientBoostingClassifier baseline with native categorical support and
honest stratified cross-validation. Reproducible: reads data from --data-dir
(defaults to the Kaggle mount, falls back to a local extract) and writes
submission.csv in the competition's id,class format.

Usage:
    python baseline.py --data-dir /kaggle/input/playground-series-s6e6 --out submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
TARGET = "class"
NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CATEGORICAL = ["spectral_type", "galaxy_population"]


def resolve_data_dir(explicit: str | None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        Path("/kaggle/input/playground-series-s6e6"),
        Path("/tmp/ps6e6"),
        Path("."),
    ]
    for c in candidates:
        if c and (c / "train.csv").exists():
            return c
    raise FileNotFoundError("train.csv not found; pass --data-dir")


def build_features(df: pd.DataFrame, encoder: OrdinalEncoder, fit: bool) -> np.ndarray:
    num = df[NUMERIC].to_numpy(dtype=float)
    cats = df[CATEGORICAL].astype(str)
    enc = encoder.fit_transform(cats) if fit else encoder.transform(cats)
    return np.hstack([num, enc])


def make_model() -> HistGradientBoostingClassifier:
    # The last two columns (encoded categoricals) are flagged categorical.
    cat_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        random_state=SEED,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data dir: {data_dir}")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"train={train.shape} test={test.shape}")
    print("class balance:\n", train[TARGET].value_counts(normalize=True).round(4))

    y = train[TARGET].to_numpy()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = build_features(train, encoder, fit=True)
    X_test = build_features(test, encoder, fit=False)

    # Honest stratified CV: report accuracy + macro-F1 before trusting the model.
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        model = make_model()
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx])
        acc = accuracy_score(y[va_idx], pred)
        f1 = f1_score(y[va_idx], pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        print(f"  fold {fold}: acc={acc:.5f} macro_f1={f1:.5f}")
    print(f"CV accuracy = {np.mean(accs):.5f} +/- {np.std(accs):.5f}")
    print(f"CV macro-F1 = {np.mean(f1s):.5f} +/- {np.std(f1s):.5f}")

    # Refit on all training data for the final submission.
    final = make_model()
    final.fit(X, y)
    preds = final.predict(X_test)

    submission = pd.DataFrame({"id": test["id"], TARGET: preds})
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(submission):,} rows)")
    print(submission[TARGET].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
