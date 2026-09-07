"""Playground telco churn, with pseudo-labelling.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import (
    TargetEncoder,
)

from kaggle_portfolio.shared.layout import RepoLayout
from kaggle_portfolio.shared.errors import CommandError

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

ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = RepoLayout.resolve().lab_root
RANDOM_STATE = 42
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


def _playground_prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.drop(columns=["Churn"]), test], axis=0, ignore_index=True
    )
    combined["TotalCharges"] = pd.to_numeric(combined["TotalCharges"], errors="coerce")
    tenure = combined["tenure"].replace(0, np.nan)
    combined["ChargesPerTenure"] = combined["TotalCharges"] / tenure
    combined["MonthlyToTenureRatio"] = combined["MonthlyCharges"] / tenure
    combined["IsNewCustomer"] = combined["tenure"].fillna(0).le(6).astype(int)
    combined["HasFiber"] = (
        combined["InternetService"].fillna("").eq("Fiber optic").astype(int)
    )
    combined["HasAutoPay"] = (
        combined["PaymentMethod"]
        .fillna("")
        .str.contains("automatic", case=False, regex=False)
        .astype(int)
    )
    combined["HasStreaming"] = (
        combined["StreamingTV"].fillna("").eq("Yes")
        | combined["StreamingMovies"].fillna("").eq("Yes")
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
    oof = cross_val_predict(model, train_x, y, cv=cv, method="predict_proba", n_jobs=1)[
        :, 1
    ]
    score = float(roc_auc_score(y, oof))
    model.fit(train_x, y)
    test_pred = model.predict_proba(test_x)[:, 1]
    return score, oof, test_pred


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
        train[target]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(train[target])
        .astype(int)
    )
    orig[target] = (
        orig[target]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(orig[target])
        .astype(int)
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
        col: pd.concat([train[col], test[col], orig[col]], axis=0).value_counts(
            normalize=True
        )
        for col in num_cols
    }
    train = _concat_feature_block(
        train,
        {
            f"FREQ_{col}": train[col].map(freq_maps[col]).fillna(0).astype("float32")
            for col in num_cols
        },
    )
    test = _concat_feature_block(
        test,
        {
            f"FREQ_{col}": test[col].map(freq_maps[col]).fillna(0).astype("float32")
            for col in num_cols
        },
    )
    orig = _concat_feature_block(
        orig,
        {
            f"FREQ_{col}": orig[col].map(freq_maps[col]).fillna(0).astype("float32")
            for col in num_cols
        },
    )
    new_num_cols.extend([f"FREQ_{col}" for col in num_cols])

    all_num = pd.concat(
        [train[num_cols], test[num_cols], orig[num_cols]], axis=0, ignore_index=True
    )
    rank_updates_train: dict[str, Any] = {}
    rank_updates_test: dict[str, Any] = {}
    rank_updates_orig: dict[str, Any] = {}
    for col in num_cols:
        ranks = (
            all_num[col].rank(method="average", pct=True).astype("float32").to_numpy()
        )
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
            updates[f"INV1P_{col}"] = (1.0 / (1.0 + values.clip(lower=0))).astype(
                "float32"
            )
        return updates

    train = _concat_feature_block(train, _power_updates(train))
    test = _concat_feature_block(test, _power_updates(test))
    orig = _concat_feature_block(orig, _power_updates(orig))
    new_num_cols.extend([f"LOG1P_{col}" for col in num_cols])
    new_num_cols.extend([f"SQRT_{col}" for col in num_cols])
    new_num_cols.extend([f"INV1P_{col}" for col in num_cols])

    def _core_numeric_updates(df: pd.DataFrame) -> dict[str, Any]:
        charges_deviation = (
            df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]
        ).astype("float32")
        service_yes_count = (df[service_cols] == "Yes").sum(axis=1).astype("float32")
        return {
            "charges_deviation": charges_deviation,
            "abs_charges_dev": np.abs(charges_deviation).astype("float32"),
            "monthly_to_total_ratio": (
                df["MonthlyCharges"] / (df["TotalCharges"] + 1)
            ).astype("float32"),
            "total_to_monthly_ratio": (
                df["TotalCharges"] / (df["MonthlyCharges"] + 1)
            ).astype("float32"),
            "avg_monthly_charges": (df["TotalCharges"] / (df["tenure"] + 1)).astype(
                "float32"
            ),
            "tenure_x_monthly": (df["tenure"] * df["MonthlyCharges"]).astype("float32"),
            "tenure_x_total": (df["tenure"] * df["TotalCharges"]).astype("float32"),
            "service_yes_count": service_yes_count,
            "service_no_count": (df[service_cols] == "No")
            .sum(axis=1)
            .astype("float32"),
            "service_other_count": (
                df[service_cols]
                .isin(["No phone service", "No internet service"])
                .sum(axis=1)
                .astype("float32")
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
        pd.concat(
            [train["MonthlyCharges"], test["MonthlyCharges"], orig["MonthlyCharges"]]
        ),
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
            "tenure_bin": pd.cut(
                train["tenure"], bins=tenure_bins, include_lowest=True
            ).astype(str),
            "MonthlyCharges_bin": pd.cut(
                train["MonthlyCharges"], bins=monthly_bins, include_lowest=True
            ).astype(str),
            "TotalCharges_bin": pd.cut(
                train["TotalCharges"], bins=total_bins, include_lowest=True
            ).astype(str),
        },
    )
    test = _concat_feature_block(
        test,
        {
            "tenure_bin": pd.cut(
                test["tenure"], bins=tenure_bins, include_lowest=True
            ).astype(str),
            "MonthlyCharges_bin": pd.cut(
                test["MonthlyCharges"], bins=monthly_bins, include_lowest=True
            ).astype(str),
            "TotalCharges_bin": pd.cut(
                test["TotalCharges"], bins=total_bins, include_lowest=True
            ).astype(str),
        },
    )
    orig = _concat_feature_block(
        orig,
        {
            "tenure_bin": pd.cut(
                orig["tenure"], bins=tenure_bins, include_lowest=True
            ).astype(str),
            "MonthlyCharges_bin": pd.cut(
                orig["MonthlyCharges"], bins=monthly_bins, include_lowest=True
            ).astype(str),
            "TotalCharges_bin": pd.cut(
                orig["TotalCharges"], bins=total_bins, include_lowest=True
            ).astype(str),
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
        cat_feature_updates["train"][name] = (
            train[left].astype(str) + "|" + train[right].astype(str)
        )
        cat_feature_updates["test"][name] = (
            test[left].astype(str) + "|" + test[right].astype(str)
        )
        cat_feature_updates["orig"][name] = (
            orig[left].astype(str) + "|" + orig[right].astype(str)
        )
        new_cat_cols.append(name)

    for left, middle, right in (("Contract", "InternetService", "PaymentMethod"),):
        name = f"{left}__{middle}__{right}"
        cat_feature_updates["train"][name] = (
            train[left].astype(str)
            + "|"
            + train[middle].astype(str)
            + "|"
            + train[right].astype(str)
        )
        cat_feature_updates["test"][name] = (
            test[left].astype(str)
            + "|"
            + test[middle].astype(str)
            + "|"
            + test[right].astype(str)
        )
        cat_feature_updates["orig"][name] = (
            orig[left].astype(str)
            + "|"
            + orig[middle].astype(str)
            + "|"
            + orig[right].astype(str)
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
        cat_feature_updates["train"][name] = (
            train[left].astype(str) + "_" + train[right].astype(str)
        )
        cat_feature_updates["test"][name] = (
            test[left].astype(str) + "_" + test[right].astype(str)
        )
        cat_feature_updates["orig"][name] = (
            orig[left].astype(str) + "_" + orig[right].astype(str)
        )
        new_cat_cols.append(name)

    for left, middle, right in combinations(ngram_top_cols[:4], 3):
        name = f"TG_{left}_{middle}_{right}"
        cat_feature_updates["train"][name] = (
            train[left].astype(str)
            + "_"
            + train[middle].astype(str)
            + "_"
            + train[right].astype(str)
        )
        cat_feature_updates["test"][name] = (
            test[left].astype(str)
            + "_"
            + test[middle].astype(str)
            + "_"
            + test[right].astype(str)
        )
        cat_feature_updates["orig"][name] = (
            orig[left].astype(str)
            + "_"
            + orig[middle].astype(str)
            + "_"
            + orig[right].astype(str)
        )
        new_cat_cols.append(name)
    train = _concat_feature_block(train, cat_feature_updates["train"])
    test = _concat_feature_block(test, cat_feature_updates["test"])
    orig = _concat_feature_block(orig, cat_feature_updates["orig"])

    counted_cat_cols = cat_cols + new_cat_cols
    all_cat_frame = pd.concat(
        [train[counted_cat_cols], test[counted_cat_cols], orig[counted_cat_cols]],
        ignore_index=True,
    )
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
        orig_proba_updates_train[name] = (
            train[col].map(lookup).fillna(orig_global).astype("float32")
        )
        orig_proba_updates_test[name] = (
            test[col].map(lookup).fillna(orig_global).astype("float32")
        )
        orig_proba_updates_orig[name] = (
            orig[col].map(lookup).fillna(orig_global).astype("float32")
        )
        new_num_cols.append(name)
    train = _concat_feature_block(train, orig_proba_updates_train)
    test = _concat_feature_block(test, orig_proba_updates_test)
    orig = _concat_feature_block(orig, orig_proba_updates_orig)

    orig_churner_tc = orig.loc[orig[target] == 1, "TotalCharges"].to_numpy(
        dtype=np.float32
    )
    orig_nonchurner_tc = orig.loc[orig[target] == 0, "TotalCharges"].to_numpy(
        dtype=np.float32
    )
    orig_tc = orig["TotalCharges"].to_numpy(dtype=np.float32)
    orig_is_mc_mean = orig.groupby("InternetService", observed=False)[
        "MonthlyCharges"
    ].mean()
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
                np.abs(zscore_against(tc, orig_churner_tc))
                - np.abs(zscore_against(tc, orig_nonchurner_tc))
            ).astype(np.float32),
            "zscore_nonchurner_TC": zscore_against(tc, orig_nonchurner_tc),
            "pctrank_churn_gap_TC": (
                pctrank_against(tc, orig_churner_tc)
                - pctrank_against(tc, orig_nonchurner_tc)
            ).astype(np.float32),
            "resid_IS_MC": (
                df["MonthlyCharges"]
                - df["InternetService"]
                .map(orig_is_mc_mean)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            ).astype(np.float32),
        }
        cond_is_vals = np.zeros(len(df), dtype=np.float32)
        for cat_val in orig["InternetService"].dropna().astype(str).unique():
            mask = df["InternetService"].astype(str) == cat_val
            if not mask.any():
                continue
            ref = orig.loc[
                orig["InternetService"].astype(str) == cat_val, "TotalCharges"
            ].to_numpy(dtype=np.float32)
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
            ref = orig.loc[
                orig["Contract"].astype(str) == cat_val, "TotalCharges"
            ].to_numpy(dtype=np.float32)
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
    train_frame, test_frame, feature_cols, te_cols, drop_raw_cols = (
        _playground_advanced_feature_frames(
            train,
            test,
            orig,
        )
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
            x_train = (
                train_frame.iloc[train_idx][feature_cols + [target]]
                .reset_index(drop=True)
                .copy()
            )
            y_train = train_frame.iloc[train_idx][target].to_numpy()
            y_valid = train_frame.iloc[valid_idx][target].to_numpy()
            x_valid = (
                train_frame.iloc[valid_idx][feature_cols].reset_index(drop=True).copy()
            )
            x_test = test_frame[feature_cols].reset_index(drop=True).copy()
            inner_cv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed
            )

            te_stat_cols = [f"TE1_{col}_{stat}" for col in te_cols for stat in stats]
            x_train = _concat_feature_block(
                x_train, {name: np.nan for name in te_stat_cols}
            )

            for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
                x_inner_train = x_train.loc[
                    inner_train_idx, feature_cols + [target]
                ].copy()
                x_inner_valid = x_train.loc[inner_valid_idx, feature_cols].copy()
                for col in te_cols:
                    grouped = x_inner_train.groupby(col, observed=False)[target].agg(
                        stats
                    )
                    grouped.columns = [f"TE1_{col}_{stat}" for stat in stats]
                    x_inner_valid = x_inner_valid.merge(grouped, on=col, how="left")
                    for name in grouped.columns:
                        x_train.loc[inner_valid_idx, name] = x_inner_valid[
                            name
                        ].to_numpy(dtype="float32")

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
    lower_threshold = min(
        float(np.quantile(predictions, lower_quantile)), 1.0 - absolute_confidence
    )
    upper_threshold = max(
        float(np.quantile(predictions, upper_quantile)), absolute_confidence
    )
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
    train_frame, test_frame, feature_cols, te_cols, drop_raw_cols = (
        _playground_advanced_feature_frames(
            train,
            test,
            orig,
        )
    )
    n_splits = min(max(3, folds), 5)
    inner_splits = min(3, n_splits)
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
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
        x_train = (
            train_frame.iloc[train_idx][feature_cols + [target]]
            .reset_index(drop=True)
            .copy()
        )
        y_train = train_frame.iloc[train_idx][target].to_numpy()
        y_valid = train_frame.iloc[valid_idx][target].to_numpy()
        x_valid = (
            train_frame.iloc[valid_idx][feature_cols].reset_index(drop=True).copy()
        )
        x_test = test_frame[feature_cols].reset_index(drop=True).copy()
        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE
        )

        te_stat_cols = [f"TE1_{col}_{stat}" for col in te_cols for stat in stats]
        x_train = _concat_feature_block(
            x_train, {name: np.nan for name in te_stat_cols}
        )

        for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
            x_inner_train = x_train.loc[inner_train_idx, feature_cols + [target]].copy()
            x_inner_valid = x_train.loc[inner_valid_idx, feature_cols].copy()
            for col in te_cols:
                grouped = x_inner_train.groupby(col, observed=False)[target].agg(stats)
                grouped.columns = [f"TE1_{col}_{stat}" for stat in stats]
                x_inner_valid = x_inner_valid.merge(grouped, on=col, how="left")
                for name in grouped.columns:
                    x_train.loc[inner_valid_idx, name] = x_inner_valid[name].to_numpy(
                        dtype="float32"
                    )

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
        sample_weight = np.concatenate(
            [np.ones(len(y_train), dtype=float), pseudo_weights]
        )

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
            col.startswith(
                (
                    "FREQ_",
                    "RANK_",
                    "LOG1P_",
                    "SQRT_",
                    "INV1P_",
                    "ISYES_",
                    "ISNO_",
                    "ISOTHER_",
                )
            )
            or "__" in col
            or col.startswith(("BG_", "TG_"))
        ):
            selected_feature_cols.append(col)
            continue
        if col.startswith("ORIG_proba_"):
            source_col = col.removeprefix("ORIG_proba_")
            if (
                source_col in base_num_cols
                or source_col in base_cat_cols
                or source_col
                in {
                    "tenure_bin",
                    "MonthlyCharges_bin",
                    "TotalCharges_bin",
                }
            ):
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
    train_frame, test_frame, feature_cols, te_cols, _drop_raw_cols = (
        _playground_advanced_feature_frames(
            train,
            test,
            orig,
        )
    )
    selected_feature_cols = _playground_lightgbm_selected_features(feature_cols)
    selected_te_cols = [
        col
        for col in _playground_lightgbm_te_columns(te_cols)
        if col in selected_feature_cols
    ]
    train_model = train_frame[selected_feature_cols].copy()
    test_model = test_frame[selected_feature_cols].copy()
    y = train_frame[target].to_numpy()
    n_splits = min(max(3, folds), 5)
    inner_splits = min(3, n_splits)
    stats = ["mean", "std"]

    raw_cat_cols = [
        col
        for col in selected_feature_cols
        if str(train_model[col].dtype) == "category"
    ]
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
            inner_cv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed
            )

            te_stat_cols = [
                f"LGB_TE1_{col}_{stat}" for col in selected_te_cols for stat in stats
            ]
            x_train = _concat_feature_block(
                x_train, {name: np.nan for name in te_stat_cols}
            )

            for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
                x_inner_train = x_train.loc[inner_train_idx, selected_te_cols].copy()
                x_inner_train[target] = y_train[inner_train_idx]
                x_inner_valid = x_train.loc[inner_valid_idx, selected_te_cols].copy()
                for col in selected_te_cols:
                    grouped = x_inner_train.groupby(col, observed=False)[target].agg(
                        stats
                    )
                    grouped.columns = [f"LGB_TE1_{col}_{stat}" for stat in stats]
                    x_inner_valid = x_inner_valid.merge(grouped, on=col, how="left")
                    for name in grouped.columns:
                        x_train.loc[inner_valid_idx, name] = x_inner_valid[
                            name
                        ].to_numpy(dtype="float32")

            for col in selected_te_cols:
                grouped = (
                    pd.DataFrame({col: x_train[col], target: y_train})
                    .groupby(col, observed=False)[target]
                    .agg(stats)
                )
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
                            mean_encoder.fit_transform(
                                x_train[selected_te_cols], y_train
                            ),
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
    train_frame, test_frame, feature_cols, _te_cols, _drop_raw_cols = (
        _playground_advanced_feature_frames(
            train,
            test,
            orig,
        )
    )
    selected_feature_cols = _playground_catboost_selected_features(feature_cols)
    train_model = train_frame[selected_feature_cols].copy()
    test_model = test_frame[selected_feature_cols].copy()
    y = train_frame[target].to_numpy()
    n_splits = min(max(3, folds), 5)

    cat_cols = [
        col
        for col in selected_feature_cols
        if str(train_model[col].dtype) == "category"
    ]
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
            rank_oof_frames = {
                name: _rank_scale(predictions[name][0]) for name in subset
            }
            rank_test_frames = {
                name: _rank_scale(predictions[name][1]) for name in subset
            }

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
                    weights = {
                        name: raw_units[idx] / units for idx, name in enumerate(subset)
                    }
                    kind, score, pred = _consider(weights)
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        best_kind = kind
                        best_pred = pred

    if best_weights is None or best_pred is None:
        return None
    return best_kind, best_weights, best_score, best_pred


def benchmark_playground_telco(
    data_dir: Path, folds: int, write_submission: bool
) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = (
        train["Churn"]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(train["Churn"])
        .astype(int)
    )
    train_x, test_x = _playground_prepare_features(train, test)
    skf = StratifiedKFold(
        n_splits=min(max(3, folds), 5), shuffle=True, random_state=RANDOM_STATE
    )

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
        lgb_score, lgb_oof, lgb_pred = _playground_model_result(
            lgb_model, train_x, test_x, y, skf
        )
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
        xgb_score, xgb_oof, xgb_pred = _playground_model_result(
            xgb_model, train_x, test_x, y, skf
        )
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
            benchmarks.append(
                {"model": "lightgbm_te", "score": round(float(lgb_score), 5)}
            )
            trained_predictions["lightgbm_te"] = lgb_pred
            blend_inputs["lightgbm_te"] = (lgb_oof, lgb_pred)
        except (RuntimeError, ValueError):
            pass
        try:
            advanced_score, advanced_oof, advanced_pred = (
                _playground_advanced_xgboost_result(
                    train,
                    test,
                    pd.read_csv(original_path),
                    folds,
                )
            )
            benchmarks.append(
                {"model": "xgboost_te", "score": round(float(advanced_score), 5)}
            )
            trained_predictions["xgboost_te"] = advanced_pred
            blend_inputs["xgboost_te"] = (advanced_oof, advanced_pred)
        except (RuntimeError, ValueError):
            pass
        try:
            pseudo_score, pseudo_oof, pseudo_pred = (
                _playground_advanced_xgboost_pseudo_result(
                    train,
                    test,
                    pd.read_csv(original_path),
                    folds,
                )
            )
            benchmarks.append(
                {"model": "xgboost_te_pseudo", "score": round(float(pseudo_score), 5)}
            )
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
            benchmarks.append(
                {"model": "catboost_te", "score": round(float(cat_score), 5)}
            )
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
        raise CommandError("No Playground Series S6E3 models are available locally.")

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = _submission_dir("playground-series-s6e3") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame(
            {"id": test["id"], "Churn": trained_predictions[best["model"]]}
        ).to_csv(
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
