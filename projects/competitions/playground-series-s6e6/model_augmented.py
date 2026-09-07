#!/usr/bin/env python3
"""Playground Series S6E6 — full PS-playbook model (catalog-validated).

Applies the recurring Playground-Series winning recipe to the stellar-class task:
  * augment training with the ORIGINAL SDSS17 dataset (synthetic data is generated
    from it; original lacks the synthetic-only categoricals -> encoded as missing),
  * blend HistGradientBoosting + XGBoost + LightGBM + CatBoost,
  * logistic-regression STACK over the four models' OOF probabilities,
  * macro-F1-optimal per-class decision weights (the competition metric).

Crucially, CV folds split the SYNTHETIC train only; the original rows go into every
TRAIN fold but never validation, so OOF still reflects the synthetic test distribution.

Run:  .venv/bin/python projects/competitions/playground-series-s6e6/model_augmented.py
"""
from __future__ import annotations

import glob
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[3]
COMP = ROOT / ".competition_lab" / "playground-series-s6e6"
ORIG = glob.glob(str(ROOT / ".competition_lab" / "sdss17" / "*.csv"))[0]
OUT = Path(__file__).resolve().parent / "submission.csv"
TARGET = "class"
BANDS = ["u", "g", "r", "i", "z"]
NUM = ["alpha", "delta", *BANDS, "redshift"]
CAT = ["spectral_type", "galaxy_population"]
COLORS = list(combinations(BANDS, 2))
REDSHIFT = ["rs_sq", "rs_near0", "rs_x_gr", "rs_x_ug", "rs_x_iz"]


def fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a, b in COLORS:
        df[f"{a}_{b}"] = df[a] - df[b]
    df["rs_sq"] = df["redshift"] ** 2
    df["rs_near0"] = (df["redshift"].abs() < 0.0025).astype(float)
    df["rs_x_gr"] = df["redshift"] * (df["g"] - df["r"])
    df["rs_x_ug"] = df["redshift"] * (df["u"] - df["g"])
    df["rs_x_iz"] = df["redshift"] * (df["i"] - df["z"])
    return df


def optimize_f1_weights(oof, y, n_classes):
    def neg_f1(w):
        return -f1_score(y, (oof * np.abs(w)).argmax(1), average="macro")
    best = minimize(neg_f1, np.ones(n_classes), method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
    w = np.abs(best.x)
    return w / w.mean(), -best.fun


def main() -> int:
    t0 = time.time()
    train = fe(pd.read_csv(COMP / "train.csv"))
    test = fe(pd.read_csv(COMP / "test.csv"))
    orig = fe(pd.read_csv(ORIG))
    for c in CAT:
        orig[c] = np.nan  # original lacks the synthetic-only categoricals
    print(f"synthetic train={train.shape} test={test.shape} | original={orig.shape}", flush=True)

    le = LabelEncoder().fit(train[TARGET].to_numpy())
    y = le.transform(train[TARGET].to_numpy())
    y_orig = le.transform(orig[TARGET].to_numpy())  # original shares GALAXY/QSO/STAR
    n_classes = len(le.classes_)

    numeric_feats = NUM + [f"{a}_{b}" for a, b in COLORS] + REDSHIFT
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    enc.fit(train[CAT].astype(str))

    def build(df):
        return np.hstack([df[numeric_feats].to_numpy(float), enc.transform(df[CAT].astype(str))])

    X, X_test, X_orig = build(train), build(test), build(orig)

    builders = {
        "hist": lambda: HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, random_state=SEED),
        "xgb": lambda: XGBClassifier(
            n_estimators=700, learning_rate=0.04, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", objective="multi:softprob", num_class=n_classes, n_jobs=-1,
            random_state=SEED, eval_metric="mlogloss"),
        "lgb": lambda: LGBMClassifier(
            objective="multiclass", num_class=n_classes, n_estimators=800, learning_rate=0.03,
            num_leaves=127, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=40, random_state=SEED, n_jobs=-1, verbose=-1),
        "cat": lambda: CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=8, loss_function="MultiClass",
            random_seed=SEED, verbose=0, allow_writing_files=False),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {k: np.zeros((len(y), n_classes)) for k in builders}
    test_pred = {k: np.zeros((len(X_test), n_classes)) for k in builders}
    for fold, (tri, vai) in enumerate(skf.split(X, y), 1):
        X_tr = np.vstack([X[tri], X_orig])          # augment train fold with ALL original
        y_tr = np.concatenate([y[tri], y_orig])
        tf = time.time()
        for name, make in builders.items():
            model = make().fit(X_tr, y_tr)
            p = model.predict_proba(X[vai])
            oof[name][vai] = p.reshape(len(vai), n_classes)
            test_pred[name] += model.predict_proba(X_test).reshape(len(X_test), n_classes) / 5
        blend = np.mean([oof[k][vai] for k in builders], axis=0)
        print(f"  fold {fold}: blend macroF1={f1_score(y[vai], blend.argmax(1), average='macro'):.5f} "
              f"({time.time() - tf:.0f}s)", flush=True)

    avg_oof = np.mean([oof[k] for k in builders], axis=0)
    avg_test = np.mean([test_pred[k] for k in builders], axis=0)
    avg_f1 = f1_score(y, avg_oof.argmax(1), average="macro")

    stack_X = np.hstack([oof[k] for k in builders])
    stack_test = np.hstack([test_pred[k] for k in builders])
    meta = LogisticRegression(max_iter=3000, C=1.0)
    stack_oof = cross_val_predict(meta, stack_X, y, cv=skf, method="predict_proba")
    stack_f1 = f1_score(y, stack_oof.argmax(1), average="macro")

    # F1-tune both candidates, keep the best.
    candidates = {}
    w_avg, f1_avg_t = optimize_f1_weights(avg_oof, y, n_classes)
    candidates["avg+f1"] = (f1_avg_t, (avg_test * w_avg))
    w_stk, f1_stk_t = optimize_f1_weights(stack_oof, y, n_classes)
    meta.fit(stack_X, y)
    candidates["stack+f1"] = (f1_stk_t, (meta.predict_proba(stack_test) * w_stk))

    print(f"\nper-model OOF macroF1: " +
          ", ".join(f"{k}={f1_score(y, oof[k].argmax(1), average='macro'):.5f}" for k in builders), flush=True)
    print(f"blend-avg OOF macroF1 = {avg_f1:.5f}  (F1-tuned {f1_avg_t:.5f})", flush=True)
    print(f"stack    OOF macroF1 = {stack_f1:.5f}  (F1-tuned {f1_stk_t:.5f})", flush=True)

    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_f1, best_test = candidates[best_name]
    print(f"\n=== BEST = {best_name}  OOF macroF1 = {best_f1:.5f}  (baseline 0.957) ===", flush=True)

    preds = le.inverse_transform(best_test.argmax(1))
    sub = pd.DataFrame({"id": test["id"], TARGET: preds})
    sub.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(sub):,} rows)  class mix: {sub[TARGET].value_counts().to_dict()}", flush=True)
    print(f"Total time: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
