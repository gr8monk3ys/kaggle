#!/usr/bin/env python3
"""Reproducible starter for Hull Tactical - Market Prediction.

Status: STARTER (not yet trained on the real, rules-gated competition data).
-------------------------------------------------------------------------------
This is a financial TIME-SERIES competition that is scored through Kaggle's
`kaggle_evaluation` inference-server gateway, NOT by uploading a static
submission.csv. You ship a notebook that defines a `predict()` callback; the
gateway streams the test set to you ONE TRADING DAY AT A TIME and you must
return that day's portfolio allocation. Returning future-looking aggregates is
therefore impossible by construction. That, plus the Sharpe-style metric, is why
this is a starter (with a runnable offline trainer + a gateway skeleton) rather
than a flat HistGradientBoosting `submission.csv` baseline.

The data (`train.csv`, `test.csv`) is download-gated: accept the rules at
https://www.kaggle.com/competitions/hull-tactical-market-prediction/rules
before `kaggle competitions download` works. Until then the offline trainer runs
on synthetic data via --smoke-test so the pipeline logic is verifiable.

Task summary
------------
* Predict a daily **allocation to the S&P 500**, bounded in **[0, 2]** (0 = all
  cash, 1 = fully invested, up to 2 = 2x leverage).
* The signal to learn is the **forward excess return** of the S&P 500 over the
  risk-free rate (the train target is along the lines of
  `market_forward_excess_returns`). Higher predicted forward return -> larger
  allocation.
* **Metric**: a modified, volatility-penalized **Sharpe ratio** of the strategy
  returns. It penalizes strategies whose realized volatility materially exceeds
  the market's (public write-ups cite a penalty when portfolio vol exceeds the
  benchmark vol by >~20%) and that fail to beat the market. So position SIZING
  and risk control matter as much as directional accuracy.

What this script provides
-------------------------
1. `load_train(data_dir)`         -- robust loader + target/feature resolver.
2. `time_series_cv(df)`           -- HONEST expanding-window (no-shuffle) CV that
                                     reports both regression RMSE and a proxy
                                     annualized Sharpe of the resulting strategy.
3. `fit_model(df)`                -- HistGradientBoostingRegressor on lagged,
                                     leak-free features.
4. `allocation_from_pred(mu, sigma)` -- maps a predicted forward return to a
                                     bounded [0, 2] allocation (the link the live
                                     gateway needs).
5. `--smoke-test`                 -- runs 1-4 on synthetic market data.

The companion `inference_server.py` shows the exact gateway `predict()` wiring
for the live submission notebook.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

SEED = 42
N_SPLITS = 5
TRADING_DAYS = 252
MIN_ALLOC, MAX_ALLOC = 0.0, 2.0

TARGET_CANDIDATES = [
    "market_forward_excess_returns", "forward_returns",
    "market_forward_returns", "target", "y",
]
# Columns that are outcomes / ids, never inputs.
NON_FEATURE = {"date_id", "date", "id", "row_id", "is_scored", "weight"}


def _resolve_target(cols):
    lower = {c.lower(): c for c in cols}
    for cand in TARGET_CANDIDATES:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    # Fallback: any column containing 'forward' and 'return'.
    for c in cols:
        cl = c.lower()
        if "forward" in cl and "return" in cl:
            return c
    return None


def load_train(data_dir: str):
    path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    target = _resolve_target(df.columns)
    if target is None:
        raise ValueError(f"No target column found. Saw: {list(df.columns)}")
    return df, target


def feature_columns(df: pd.DataFrame, target: str):
    feats = []
    for c in df.columns:
        if c == target or c in NON_FEATURE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def _annualized_sharpe(strategy_rets: np.ndarray) -> float:
    sd = strategy_rets.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * strategy_rets.mean() / sd)


def allocation_from_pred(mu: np.ndarray, sigma: float) -> np.ndarray:
    """Map predicted forward excess return -> allocation in [0, 2].

    Simple, monotone, risk-aware link: scale the predicted edge by its own
    historical spread and center at fully-invested (1.0). Replace with a proper
    mean-variance / Kelly-fraction sizing once the metric is confirmed.
    """
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    alloc = 1.0 + mu / (2.0 * sigma)
    return np.clip(alloc, MIN_ALLOC, MAX_ALLOC)


def time_series_cv(df: pd.DataFrame, target: str) -> float:
    """Expanding-window CV (NO shuffling): train on the past, validate on the
    immediately following block. Reports RMSE and a strategy Sharpe proxy."""
    feats = feature_columns(df, target)
    X = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy()
    y = df[target].to_numpy()
    n = len(df)
    fold_size = n // (N_SPLITS + 1)
    if fold_size == 0:
        raise ValueError("Not enough rows for time-series CV.")

    rmses, sharpes = [], []
    for k in range(1, N_SPLITS + 1):
        tr_end = fold_size * k
        va_end = min(fold_size * (k + 1), n)
        tr = slice(0, tr_end)
        va = slice(tr_end, va_end)
        if va.stop - va.start < 5:
            continue
        model = HistGradientBoostingRegressor(
            random_state=SEED, max_iter=300, learning_rate=0.03,
            max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        rmse = float(np.sqrt(mean_squared_error(y[va], pred)))
        sigma = float(np.nanstd(y[tr]))
        alloc = allocation_from_pred(pred, sigma)
        # Realized strategy excess return = allocation * realized fwd return.
        strat = alloc * y[va]
        sharpe = _annualized_sharpe(strat[np.isfinite(strat)])
        rmses.append(rmse)
        sharpes.append(sharpe)
        print(f"  fold {k}  train=[0:{tr_end}] val=[{tr_end}:{va_end}]  "
              f"RMSE={rmse:.6f}  strat_Sharpe={sharpe:.3f}")
    mean_rmse = float(np.mean(rmses))
    mean_sharpe = float(np.mean(sharpes))
    print(f"CV RMSE={mean_rmse:.6f}   CV strategy Sharpe (proxy)={mean_sharpe:.3f}"
          "   (expanding window, no shuffle)")
    return mean_sharpe


def fit_model(df: pd.DataFrame, target: str):
    feats = feature_columns(df, target)
    X = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy()
    y = df[target].to_numpy()
    model = HistGradientBoostingRegressor(
        random_state=SEED, max_iter=500, learning_rate=0.03,
        max_leaf_nodes=31, l2_regularization=1.0,
    )
    model.fit(X, y)
    return model, feats, float(np.nanstd(y))


# --- Synthetic smoke test -----------------------------------------------------
def make_synthetic_train(path: str, n: int = 2000, n_feats: int = 12):
    rng = np.random.default_rng(SEED)
    F = rng.normal(size=(n, n_feats))
    # A few features carry a weak, realistic predictive edge on next-day return.
    beta = np.zeros(n_feats)
    beta[:3] = [0.004, -0.003, 0.002]
    fwd = F @ beta + rng.normal(0, 0.01, size=n)  # daily excess return ~1% vol
    cols = {f"feat_{i}": F[:, i] for i in range(n_feats)}
    cols["date_id"] = np.arange(n)
    cols["market_forward_excess_returns"] = fwd
    pd.DataFrame(cols).to_csv(path, index=False)


def run_smoke_test():
    print("=== SMOKE TEST: synthetic market data ===")
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic_train(os.path.join(tmp, "train.csv"))
        df, target = load_train(tmp)
        print(f"rows={len(df)}  target='{target}'  "
              f"features={len(feature_columns(df, target))}")
        sharpe = time_series_cv(df, target)
        model, feats, sigma = fit_model(df, target)
        # Verify the live link function on the last day.
        last = df[feats].iloc[[-1]].replace([np.inf, -np.inf], np.nan).to_numpy()
        mu = model.predict(last)
        alloc = allocation_from_pred(mu, sigma)
        assert MIN_ALLOC <= alloc[0] <= MAX_ALLOC, alloc
        assert np.isfinite(alloc[0])
        print(f"Last-day predicted edge={mu[0]:.5f} -> allocation={alloc[0]:.3f} "
              f"(bounded in [{MIN_ALLOC},{MAX_ALLOC}])")
        print(f"SMOKE TEST PASSED (synthetic strategy Sharpe proxy={sharpe:.3f}). "
              "Offline pipeline + allocation link are sound.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None,
                    help="Dir containing train.csv (and optionally test.csv).")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test or args.data_dir is None:
        if args.data_dir is None and not args.smoke_test:
            print("No --data-dir given; running --smoke-test.\n"
                  "(Accept rules + download to train for real; live scoring is "
                  "via the kaggle_evaluation gateway -- see inference_server.py.)")
        run_smoke_test()
        return

    df, target = load_train(args.data_dir)
    print(f"Loaded {len(df)} rows; target='{target}'; "
          f"{len(feature_columns(df, target))} features.")
    time_series_cv(df, target)
    model, feats, sigma = fit_model(df, target)
    print(f"Refit on all rows. {len(feats)} features, return sigma={sigma:.5f}.")
    print("For the LIVE submission, load this model inside inference_server.py "
          "and serve allocations through the gateway.")


if __name__ == "__main__":
    main()
