#!/usr/bin/env python3
"""Live-submission skeleton for Hull Tactical - Market Prediction.

This is the pattern your SUBMISSION notebook uses. Hull Tactical is scored
through Kaggle's `kaggle_evaluation` inference-server gateway: the gateway calls
your `predict()` once per trading day, streaming that day's feature row, and you
return the day's allocation to the S&P 500 (a single float bounded in [0, 2]).
You never see the full test set at once, so look-ahead is impossible.

This file CANNOT run here because the `kaggle_evaluation` package only ships
inside the competition's data bundle (download-gated until you accept the rules).
It is the correct, faithful wiring -- not a stub that pretends to score. On
Kaggle, the gateway and data are present and this runs unchanged.

Files the gateway provides (visible via `kaggle competitions files`):
  kaggle_evaluation/                      <- the gateway package (do NOT edit)
  kaggle_evaluation/default_inference_server.py
  kaggle_evaluation/default_gateway.py
  train.csv, test.csv

Usage on Kaggle (inside the submission notebook):
  1. Train / load your model once at import time (see `load_or_train_model`).
  2. Implement `predict(test_row)` -> float allocation in [0, 2].
  3. Hand `predict` to the gateway server and start it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Reuse the offline trainer's feature/target resolution + allocation link so the
# offline CV and the live server use IDENTICAL logic (no train/serve skew).
from baseline import (
    allocation_from_pred,
    feature_columns,
    fit_model,
    load_train,
)

# Path conventions on Kaggle. Adjust if the competition mounts data elsewhere.
KAGGLE_DATA_DIR = "/kaggle/input/hull-tactical-market-prediction"

_MODEL = None
_FEATS: list[str] = []
_SIGMA = 1.0


def load_or_train_model(data_dir: str = KAGGLE_DATA_DIR):
    """Train the model once from train.csv. Called at notebook import time."""
    global _MODEL, _FEATS, _SIGMA
    df, target = load_train(data_dir)
    _MODEL, _FEATS, _SIGMA = fit_model(df, target)
    return _MODEL


def predict(test_row: "pd.DataFrame") -> float:
    """Gateway callback: one trading day in, one allocation out.

    `test_row` is a single-row DataFrame of that day's features (same schema as
    train.csv minus the target). Return a float allocation in [0, 2].
    """
    if _MODEL is None:
        load_or_train_model()
    # Align to the exact training feature order. Fill any missing column with NaN
    # (NOT 0.0): the HistGradientBoosting model handles NaN natively as "missing",
    # matching training, whereas a hard 0.0 is an out-of-distribution value that
    # silently skews predictions.
    x = pd.DataFrame(index=test_row.index)
    for c in _FEATS:
        x[c] = test_row[c] if c in test_row.columns else np.nan
    x = x.replace([np.inf, -np.inf], np.nan).to_numpy()
    mu = _MODEL.predict(x)
    alloc = allocation_from_pred(mu, _SIGMA)
    return float(alloc[0])


def serve():
    """Start the competition gateway. Only works inside the Kaggle runtime where
    the `kaggle_evaluation` package is present."""
    try:
        import kaggle_evaluation.default_inference_server as kis
    except ImportError as e:  # pragma: no cover - only importable on Kaggle
        raise SystemExit(
            "kaggle_evaluation not found. This server runs inside the Kaggle "
            "competition notebook, where the gateway package ships with the "
            "data. Accept the rules and run there.\n"
            f"(import error: {e})"
        )
    load_or_train_model()
    server = kis.HullInferenceServer(predict)  # class name per the comp gateway
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        server.serve()  # scored run: stream the real (hidden) test set
    else:
        # Local debug run against the provided sample test set.
        server.run_local_gateway((os.path.join(KAGGLE_DATA_DIR, "test.csv"),))


if __name__ == "__main__":
    serve()
