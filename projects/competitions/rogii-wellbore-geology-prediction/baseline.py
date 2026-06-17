#!/usr/bin/env python3
"""Reproducible starter baseline for ROGII - Wellbore Geology Prediction.

Status: STARTER (not yet trained on the real, rules-gated competition data).
-------------------------------------------------------------------------------
The competition data is download-gated: you must accept the rules at
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules
before `kaggle competitions download` works. Until then this script cannot be
run against real data, so the CV/LB numbers below are NOT filled in.

What this script DOES, end to end, the moment the data is present:
  1. Resolves a --data-dir containing train/ and test/ folders of per-well CSVs.
  2. For every well, joins its `<id>__horizontal_well.csv` (the drilled lateral,
     indexed by measured depth MD) with `<id>__typewell.csv` (the offset/pilot
     reference log, indexed by true vertical depth TVD) into ONE tabular frame.
  3. Engineers depth- and log-derived features (rolling stats, gradients, the
     nearest typewell log value at the current TVD, etc.).
  4. Trains a HistGradientBoostingRegressor with HONEST GroupKFold CV where the
     group is the WELL id -- so no rows from a training well leak into its own
     validation fold. (Random KFold here would be optimistic: rows within a
     well are autocorrelated.)
  5. Refits on all wells and writes a submission in the exact sample_submission
     column order.

The target is TVT (True Vertical Thickness): the vertical offset of the lateral
relative to the geological marker, predicted at each station along the wellbore.
The competition metric is an RMSE-style error on TVT (lower is better).

Run modes
---------
  # Verify the pipeline logic on synthetic data shaped like the real schema:
  python baseline.py --smoke-test

  # Real run (after accepting rules + downloading):
  kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
  cd data && unzip -q rogii-wellbore-geology-prediction.zip && cd ..
  python baseline.py --data-dir ./data --out /tmp/rogii_submission.csv

Because the exact log column names are only visible after download, the feature
builder is SCHEMA-ADAPTIVE: it discovers numeric log columns at runtime and
treats anything matching the depth / target name patterns specially. Adjust the
*_CANDIDATES constants below once you have seen the real headers.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

SEED = 42
N_SPLITS = 5

# --- Schema hints (refine after inspecting the real headers) -----------------
# Measured-depth column in the horizontal (lateral) log.
MD_CANDIDATES = ["md", "MD", "measured_depth", "depth", "DEPTH"]
# True-vertical-depth column in the typewell (offset/pilot) log.
TVD_CANDIDATES = ["tvd", "TVD", "tvdss", "true_vertical_depth"]
# The prediction target along the lateral.
TARGET_CANDIDATES = ["tvt", "TVT", "target", "dz", "vertical_offset"]
# Common petrophysical logs (used if present; missing ones are skipped).
LOG_CANDIDATES = ["gr", "GR", "gamma", "gamma_ray", "res", "resistivity",
                  "rhob", "density", "nphi", "porosity", "dt", "sonic", "pe",
                  "incl", "inclination", "azimuth", "tvd"]


def _first_present(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def list_well_ids(split_dir: str) -> list[str]:
    """Return well ids that have a horizontal_well.csv in `split_dir`."""
    pat = os.path.join(split_dir, "*__horizontal_well.csv")
    ids = []
    for p in sorted(glob.glob(pat)):
        base = os.path.basename(p)
        ids.append(base.replace("__horizontal_well.csv", ""))
    return ids


def build_well_table(split_dir: str, well_id: str) -> pd.DataFrame:
    """Join one well's lateral log with its typewell into a single frame.

    The typewell is the geological reference (logs vs. TVD). For each lateral
    station we look up the nearest typewell sample by TVD via a merge_asof, so
    the model can compare the lateral's logs against the expected formation
    response at the same vertical depth -- the core geosteering signal.
    """
    h_path = os.path.join(split_dir, f"{well_id}__horizontal_well.csv")
    t_path = os.path.join(split_dir, f"{well_id}__typewell.csv")
    hz = pd.read_csv(h_path)
    tw = pd.read_csv(t_path)

    md_col = _first_present(hz.columns, MD_CANDIDATES)
    if md_col is None:
        # Fall back to row order if no explicit depth column.
        hz = hz.reset_index().rename(columns={"index": "md"})
        md_col = "md"
    hz = hz.sort_values(md_col).reset_index(drop=True)

    # Attach nearest typewell logs by TVD if both sides expose a TVD column.
    tvd_h = _first_present(hz.columns, TVD_CANDIDATES)
    tvd_t = _first_present(tw.columns, TVD_CANDIDATES)
    if tvd_h is not None and tvd_t is not None:
        tw_sorted = tw.sort_values(tvd_t).reset_index(drop=True)
        tw_logs = [c for c in tw_sorted.columns if c != tvd_t]
        tw_sorted = tw_sorted.rename(columns={c: f"tw_{c}" for c in tw_logs})
        merged = pd.merge_asof(
            hz.sort_values(tvd_h),
            tw_sorted,
            left_on=tvd_h,
            right_on=tvd_t,
            direction="nearest",
        )
        merged = merged.sort_values(md_col).reset_index(drop=True)
    else:
        merged = hz

    merged["well_id"] = well_id
    merged["_md_col"] = md_col
    return merged


def add_features(df: pd.DataFrame, md_col: str) -> pd.DataFrame:
    """Depth- and log-derived features, computed within a single well."""
    df = df.copy()
    df["md_norm"] = (df[md_col] - df[md_col].min()) / (
        df[md_col].max() - df[md_col].min() + 1e-9
    )
    df["md_step"] = df[md_col].diff().fillna(0.0)

    log_cols = [c for c in df.columns
                if (_first_present([c], LOG_CANDIDATES) or c.startswith("tw_"))
                and pd.api.types.is_numeric_dtype(df[c])]
    for c in log_cols:
        df[f"{c}_grad"] = df[c].diff().fillna(0.0)
        df[f"{c}_roll5"] = df[c].rolling(5, min_periods=1).mean()
        df[f"{c}_rollstd5"] = df[c].rolling(5, min_periods=1).std().fillna(0.0)
    return df


def assemble(split_dir: str) -> tuple[pd.DataFrame, list[str], str]:
    well_ids = list_well_ids(split_dir)  # list once; the returned ids must match the frame
    frames = []
    md_col = None
    for wid in well_ids:
        wt = build_well_table(split_dir, wid)
        md_col = wt["_md_col"].iloc[0]
        wt = add_features(wt, md_col)
        frames.append(wt)
    if not frames:
        raise FileNotFoundError(f"No *__horizontal_well.csv under {split_dir}")
    full = pd.concat(frames, ignore_index=True)
    full = full.drop(columns=["_md_col"])
    return full, well_ids, md_col


def feature_matrix(df: pd.DataFrame, target_col: str | None):
    drop = {"well_id"}
    if target_col:
        drop.add(target_col)
    feats = [c for c in df.columns
             if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    return X, feats


def cross_validate(train: pd.DataFrame, target_col: str) -> float:
    X, feats = feature_matrix(train, target_col)
    y = train[target_col].to_numpy()
    groups = train["well_id"].to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(N_SPLITS, n_groups)
    gkf = GroupKFold(n_splits=splits)

    scores = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        model = HistGradientBoostingRegressor(
            random_state=SEED, max_iter=400, learning_rate=0.05,
            max_leaf_nodes=63, l2_regularization=1.0, early_stopping=True,
        )
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])
        rmse = float(np.sqrt(mean_squared_error(y[va], pred)))
        scores.append(rmse)
        print(f"  fold {fold}/{splits}  RMSE={rmse:.4f}  "
              f"(val wells={len(np.unique(groups[va]))})")
    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"CV RMSE: {mean:.4f} +/- {std:.4f}  (GroupKFold by well)")
    return mean


def fit_full_and_predict(train, test, target_col, feats):
    X = train[feats].replace([np.inf, -np.inf], np.nan)
    y = train[target_col].to_numpy()
    model = HistGradientBoostingRegressor(
        random_state=SEED, max_iter=600, learning_rate=0.05,
        max_leaf_nodes=63, l2_regularization=1.0,
    )
    model.fit(X, y)
    Xt = test[feats].replace([np.inf, -np.inf], np.nan)
    return model.predict(Xt)


def write_submission(test, preds, sample_path, out_path, target_col):
    test = test.copy()
    test["_pred"] = preds
    if os.path.exists(sample_path):
        sub = pd.read_csv(sample_path)
        matched = [c for c in sub.columns if _first_present([c], TARGET_CANDIDATES)]
        pred_col = matched[0] if matched else sub.columns[-1]
        # Align predictions to sample_submission by the id column(s) the two frames
        # SHARE, not positionally: `test` is in well/MD order while the sample may be
        # in a different (e.g. id-sorted) order, so `sub[pred_col] = preds` would
        # silently mis-assign every prediction. Fall back to positional only when no
        # shared unique key exists, and say so loudly.
        key_cols = [c for c in sub.columns if c in test.columns and c != pred_col]
        keyed_unique = bool(key_cols) and not test[key_cols].duplicated().any()
        if keyed_unique and len(sub) == len(test):
            merged = sub.drop(columns=[pred_col], errors="ignore").merge(
                test[key_cols + ["_pred"]], on=key_cols, how="left")
            sub[pred_col] = merged["_pred"].to_numpy()
            missing = int(pd.isna(sub[pred_col]).sum())
            if missing:
                fill = float(np.nanmean(preds))
                print(f"WARNING: {missing}/{len(sub)} submission rows had no matching "
                      f"prediction (id mismatch); filled with mean {fill:.4f}.",
                      file=sys.stderr)
                sub[pred_col] = sub[pred_col].fillna(fill)
        elif len(sub) == len(preds):
            print("WARNING: no shared unique id column between sample_submission and "
                  "test; using POSITIONAL alignment -- verify the row order matches!",
                  file=sys.stderr)
            sub[pred_col] = preds
        else:
            print(f"WARNING: sample rows ({len(sub)}) != preds ({len(preds)});"
                  " writing best-effort frame.", file=sys.stderr)
            sub = pd.DataFrame({pred_col: preds})
    else:
        sub = pd.DataFrame({"well_id": test["well_id"], target_col: preds})
    sub.to_csv(out_path, index=False)
    print(f"Wrote submission: {out_path}  ({len(sub)} rows, "
          f"cols={list(sub.columns)})")


# --- Synthetic smoke test (verifies the pipeline without the gated data) ------
def make_synthetic(root: str, n_wells: int, with_target: bool):
    rng = np.random.default_rng(SEED)
    split = os.path.join(root, "train" if with_target else "test")
    os.makedirs(split, exist_ok=True)
    for i in range(n_wells):
        wid = f"well{i:03d}"
        n = rng.integers(200, 400)
        md = np.cumsum(rng.uniform(0.4, 0.6, size=n)) + 1000
        tvd = 2000 + 50 * np.sin(md / 80) + rng.normal(0, 0.5, size=n)
        gr = 60 + 30 * np.sin(md / 40) + rng.normal(0, 5, size=n)
        res = np.exp(rng.normal(2.0, 0.3, size=n))
        hz = pd.DataFrame({"md": md, "tvd": tvd, "gr": gr, "resistivity": res})
        if with_target:
            # TVT as a smooth function of logs+depth so the model can learn it.
            hz["tvt"] = (0.02 * (gr - 60) - 1.5 * np.log(res)
                         + 0.01 * (tvd - 2000) + rng.normal(0, 0.3, size=n))
        hz.to_csv(os.path.join(split, f"{wid}__horizontal_well.csv"),
                  index=False)
        m = rng.integers(80, 150)
        tvd_t = np.linspace(1950, 2100, m)
        tw = pd.DataFrame({"tvd": tvd_t,
                           "gr": 60 + 30 * np.sin(tvd_t / 40),
                           "resistivity": np.exp(0.001 * tvd_t)})
        tw.to_csv(os.path.join(split, f"{wid}__typewell.csv"), index=False)


def run_smoke_test():
    print("=== SMOKE TEST: synthetic data shaped like the real schema ===")
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic(tmp, n_wells=12, with_target=True)
        make_synthetic(tmp, n_wells=4, with_target=False)
        train, _, md_col = assemble(os.path.join(tmp, "train"))
        test, _, test_md = assemble(os.path.join(tmp, "test"))
        target_col = _first_present(train.columns, TARGET_CANDIDATES)
        assert target_col, "target not found in synthetic train"
        print(f"train rows={len(train)}  test rows={len(test)}  "
              f"target='{target_col}'")
        cv = cross_validate(train, target_col)
        X, feats = feature_matrix(train, target_col)
        preds = fit_full_and_predict(train, test, target_col, feats)
        out = os.path.join(tmp, "sub.csv")
        write_submission(test, preds, "/nonexistent", out, target_col)
        assert len(preds) == len(test)
        assert np.isfinite(preds).all()

        # Verify id-based alignment: a sample_submission whose rows are SHUFFLED
        # relative to `test` must still receive each row's correct prediction.
        sample = test[["well_id", test_md]].copy()
        sample[target_col] = 0.0
        sample = sample.sample(frac=1.0, random_state=1).reset_index(drop=True)
        sample_path = os.path.join(tmp, "sample_submission.csv")
        sample.to_csv(sample_path, index=False)
        aligned = os.path.join(tmp, "aligned.csv")
        write_submission(test, preds, sample_path, aligned, target_col)
        written = pd.read_csv(aligned)
        expect = test[["well_id", test_md]].copy()
        expect["_p"] = preds
        check = written.merge(expect, on=["well_id", test_md])
        assert len(check) == len(test), "alignment merge lost/duplicated rows"
        assert np.allclose(check[target_col].to_numpy(), check["_p"].to_numpy()), (
            "submission predictions are MISALIGNED to sample_submission rows")
        print("Alignment check PASSED (predictions matched shuffled sample rows by id).")

        print(f"SMOKE TEST PASSED (synthetic CV RMSE={cv:.4f}, "
              f"{len(feats)} features). Pipeline logic is sound.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None,
                    help="Dir containing train/ and test/ well CSVs.")
    ap.add_argument("--out", default="/tmp/rogii_submission.csv")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run pipeline on synthetic data (no download needed).")
    args = ap.parse_args()

    if args.smoke_test or args.data_dir is None:
        if args.data_dir is None and not args.smoke_test:
            print("No --data-dir given; running --smoke-test instead.\n"
                  "(Accept the competition rules + download to train for real.)")
        run_smoke_test()
        return

    # Resolve train/test subdirs robustly.
    base = args.data_dir
    train_dir = next((d for d in (os.path.join(base, "train"), base)
                      if list_well_ids(d)), None)
    test_dir = next((d for d in (os.path.join(base, "test"), base)
                     if list_well_ids(d)), None)
    if not train_dir:
        sys.exit(f"No wells found under {base}/train or {base}.")
    sample_path = os.path.join(base, "sample_submission.csv")

    print(f"Loading train wells from {train_dir} ...")
    train, _, md_col = assemble(train_dir)
    target_col = _first_present(train.columns, TARGET_CANDIDATES)
    if not target_col:
        sys.exit(f"Could not locate target column. Saw: {list(train.columns)}")
    print(f"train rows={len(train)}  target='{target_col}'")

    cross_validate(train, target_col)

    X, feats = feature_matrix(train, target_col)
    if test_dir:
        test, _, _ = assemble(test_dir)
        preds = fit_full_and_predict(train, test, target_col, feats)
        write_submission(test, preds, sample_path, args.out, target_col)
    else:
        print("No test wells found; skipping submission.")


if __name__ == "__main__":
    main()
