#!/usr/bin/env python3
"""Build the data_leakage_cv_pitfalls.ipynb notebook."""
import sys as _sys
import os as _os


def _find_repo_root(start_dir):
    current = _os.path.abspath(start_dir)
    while True:
        if _os.path.exists(_os.path.join(current, "manage.sh")) and _os.path.isdir(_os.path.join(current, "kaggle_portfolio")):
            return current
        parent = _os.path.dirname(current)
        if parent == current:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        current = parent


_sys.path.insert(0, _find_repo_root(_os.path.dirname(_os.path.abspath(__file__))))
from kaggle_portfolio.shared.build_utils import md, code, write_notebook

cells = []

# ── Cell 1: Title ─────────────────────────────────────────────────────────────
cells.append(md(
'# <center>5 Ways Your Cross-Validation Lies to You</center>\n'
'\n'
'<center>\n'
'\n'
'![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)\n'
'![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)\n'
'![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy)\n'
'![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)\n'
'![License](https://img.shields.io/badge/License-MIT-red)\n'
'\n'
'</center>\n'
'\n'
'---\n'
'\n'
'**Author:** Lorenzo Scaturchio  \n'
'**Last Updated:** July 2026  \n'
'**Kernel Version:** 1.0\n'
'\n'
'---'
))

# ── Cell 2: TL;DR + TOC ───────────────────────────────────────────────────────
cells.append(md(
'## TL;DR\n'
'\n'
'The most expensive bug in a Kaggle pipeline is not a crash — it is a **local CV\n'
'score that looks great and means nothing.** Data leakage lets information from\n'
'the validation fold sneak into training, so your CV climbs while your\n'
'leaderboard score does not. This notebook builds **five leaks from scratch on\n'
'data with no real signal**, shows each one producing an impressive-but-fake\n'
'score, then fixes it and watches the score collapse back to honest. Every number\n'
'below is computed live — fork it and reproduce the collapse yourself.\n'
'\n'
'| # | The leak | Fake CV | Honest CV |\n'
'|---|----------|:-------:|:---------:|\n'
'| 1 | Feature selection on the full dataset | ~0.74 | ~0.50 |\n'
'| 2 | Target encoding without out-of-fold | ~0.84 | ~0.49 |\n'
'| 3 | Random KFold on grouped rows | **1.00** | ~0.50 |\n'
'| 4 | Shuffled CV on a time series | R² ~0.99 | R² < 0 |\n'
'| 5 | Duplicate rows split across folds | ~0.76 | ~0.55 |\n'
'\n'
'*(Ground truth in every case is chance — ~0.50 AUC / ~0 R². Any lift above that\n'
'is the leak talking.)*\n'
'\n'
'## Table of Contents\n'
'\n'
'1. [Objective](#1.-Objective)\n'
'2. [What Leakage Is, Precisely](#2.-What-Leakage-Is,-Precisely)\n'
'3. [Leak 1 — Preprocessing on the Full Dataset](#3.-Leak-1)\n'
'4. [Leak 2 — Target Encoding Without Out-of-Fold](#4.-Leak-2)\n'
'5. [Leak 3 — Random Splits on Grouped Data](#5.-Leak-3)\n'
'6. [Leak 4 — Shuffled CV on a Time Series](#6.-Leak-4)\n'
'7. [Leak 5 — Duplicate Rows Across Folds](#7.-Leak-5)\n'
'8. [The Damage, Side by Side](#8.-The-Damage,-Side-by-Side)\n'
'9. [A Leak-Proofing Checklist](#9.-A-Leak-Proofing-Checklist)\n'
'10. [Conclusion](#10.-Conclusion)'
))

# ── Cell 3: §1 objective ──────────────────────────────────────────────────────
cells.append(md(
'## 1. Objective\n'
'\n'
'Every Kaggler eventually lives the same horror story: local CV says 0.92, you\n'
'submit, the leaderboard says 0.78. The gap is almost always **leakage** — and\n'
'the insidious part is that a leaky pipeline runs without error and *feels* like\n'
'progress.\n'
'\n'
'By the end of this notebook you will be able to:\n'
'\n'
'- name the five leakage patterns that account for most "my CV lied" posts;\n'
'- **see each one inflate a score on pure-noise data**, so you trust the\n'
'  mechanism rather than taking it on faith;\n'
'- fix each with the correct scikit-learn construct (`Pipeline`, out-of-fold\n'
'  encoding, `GroupKFold`, `TimeSeriesSplit`, de-duplication);\n'
'- apply a pre-submission checklist that catches leakage before the leaderboard\n'
'  does.\n'
'\n'
'The trick used throughout: the data has **no real signal**. Targets are random,\n'
'or learnable only by cheating. So the honest score *must* be chance — and every\n'
'point above chance is the leak, measured.'
))

cells.append(code(
'import numpy as np\n'
'import pandas as pd\n'
'import matplotlib.pyplot as plt\n'
'from sklearn.model_selection import (\n'
'    cross_val_score, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit)\n'
'from sklearn.feature_selection import SelectKBest, f_classif\n'
'from sklearn.linear_model import LogisticRegression, Ridge\n'
'from sklearn.neighbors import KNeighborsClassifier\n'
'from sklearn.ensemble import GradientBoostingRegressor\n'
'from sklearn.preprocessing import OneHotEncoder\n'
'from sklearn.pipeline import Pipeline\n'
'\n'
'SEED = 0\n'
'rng = np.random.default_rng(SEED)\n'
'np.random.seed(SEED)\n'
'\n'
'# Every leak records (fake, honest) here for the summary chart in Section 8.\n'
'SCORES = {}\n'
'print("environment ready")'
))

# ── Cell 4: §2 what leakage is ────────────────────────────────────────────────
cells.append(md(
'## 2. What Leakage Is, Precisely\n'
'\n'
'> **Leakage:** any path by which information that would not be available at\n'
'> prediction time influences training or model selection.\n'
'\n'
'In a cross-validation setting it has one operational meaning: **the validation\n'
'fold touched the training process.** That contact can be direct (the same row\n'
'appears in both) or indirect (a scaler, a feature, or a selection step was fit\n'
'using the validation rows). The result is always the same — the model is\n'
'partly graded on data it has already seen, so the CV score is optimistic.\n'
'\n'
'The fix is always the same principle too: **every step that learns from data —\n'
'not just the model — must be fit inside the fold, on training rows only.** The\n'
'five sections below are five faces of that single rule.'
))

# ── Cell 5: §3 leak 1 ─────────────────────────────────────────────────────────
cells.append(md(
'## 3. Leak 1 — Preprocessing on the Full Dataset\n'
'\n'
'The classic. You select the top-K features (or fit a scaler, or a PCA) on the\n'
'whole dataset *before* cross-validating the model. The selection step already\n'
'peeked at every validation label to decide which features matter.\n'
'\n'
'**Setup:** 800 rows, 5,000 pure-noise features, a random binary target. There\n'
'is nothing to learn — honest AUC must be ~0.50.'
))

cells.append(code(
'n, p = 800, 5000\n'
'X = rng.standard_normal((n, p))\n'
'y = rng.integers(0, 2, n)\n'
'\n'
'# WRONG: pick the 20 "best" features using the whole y, then CV the model\n'
'X_pre = SelectKBest(f_classif, k=20).fit(X, y).transform(X)\n'
'fake = cross_val_score(LogisticRegression(max_iter=1000), X_pre, y,\n'
'                       cv=5, scoring="roc_auc").mean()\n'
'\n'
'# RIGHT: selection lives inside the pipeline, re-fit on each training fold\n'
'pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)),\n'
'                 ("clf", LogisticRegression(max_iter=1000))])\n'
'honest = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()\n'
'\n'
'SCORES["Feature selection\\non full data"] = (fake, honest)\n'
'print(f"leaked AUC  = {fake:.3f}   <- looks like real signal")\n'
'print(f"honest AUC  = {honest:.3f}   <- the truth: pure noise")'
))

cells.append(md(
'A ~0.24 AUC gap conjured from **random numbers**. Selecting 20 of 5,000 noise\n'
'columns by their correlation with the full target guarantees some will look\n'
'predictive on the validation rows they were chosen with. Put the selector in a\n'
'`Pipeline` and `cross_val_score` re-fits it per fold — the illusion vanishes.'
))

# ── Cell 6: §4 leak 2 ─────────────────────────────────────────────────────────
cells.append(md(
'## 4. Leak 2 — Target Encoding Without Out-of-Fold\n'
'\n'
'Target (mean) encoding replaces a category with the average target for that\n'
'category. Done naively — computing each row\'s encoding from a mean that\n'
'**includes that row** — it leaks the label straight into the feature. The\n'
'damage scales with cardinality: rare categories essentially memorise their own\n'
'targets.\n'
'\n'
'**Setup:** 1,500 rows, a 600-category column (~2-3 rows each), random target.'
))

cells.append(code(
'n = 1500\n'
'cat = rng.integers(0, 600, n)      # high-cardinality id\n'
'y = rng.integers(0, 2, n)\n'
'\n'
'# WRONG: encode each row with its category mean over ALL rows (includes itself)\n'
'te_full = pd.Series(y).groupby(cat).transform("mean").to_numpy().reshape(-1, 1)\n'
'fake = cross_val_score(LogisticRegression(max_iter=1000), te_full, y,\n'
'                       cv=5, scoring="roc_auc").mean()\n'
'\n'
'# RIGHT: out-of-fold encoding — each row encoded from OTHER folds only\n'
'oof = np.zeros(n)\n'
'for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(cat, y):\n'
'    fold_mean = pd.Series(y[tr]).groupby(cat[tr]).mean()\n'
'    oof[va] = pd.Series(cat[va]).map(fold_mean).fillna(y[tr].mean()).to_numpy()\n'
'honest = cross_val_score(LogisticRegression(max_iter=1000), oof.reshape(-1, 1), y,\n'
'                         cv=5, scoring="roc_auc").mean()\n'
'\n'
'SCORES["Target encoding\\nwithout OOF"] = (fake, honest)\n'
'print(f"leaked AUC  = {fake:.3f}   <- the feature is a disguised copy of y")\n'
'print(f"honest AUC  = {honest:.3f}   <- out-of-fold encoding tells the truth")'
))

cells.append(md(
'The naive encoding scored ~0.84 on a **random** target — because for a category\n'
'with two rows, "mean target of this category" is almost literally this row\'s\n'
'label. Out-of-fold encoding computes each row\'s value from folds that exclude\n'
'it, and the fake signal disappears. (Smoothing toward the global mean helps too,\n'
'but out-of-fold is the non-negotiable part.)'
))

# ── Cell 7: §5 leak 3 ─────────────────────────────────────────────────────────
cells.append(md(
'## 5. Leak 3 — Random Splits on Grouped Data\n'
'\n'
'When rows cluster into groups — multiple visits per patient, several photos per\n'
'user, repeated measurements per device — and the label is a property of the\n'
'**group**, a random KFold puts some of a group\'s rows in train and the rest in\n'
'validation. The model memorises the group identity and reads the answer off.\n'
'\n'
'**Setup:** 150 users x 10 rows; the label is constant within a user and random\n'
'across users; the *only* feature is the one-hot user id. Nothing generalises —\n'
'honest AUC is 0.50.'
))

cells.append(code(
'groups = np.repeat(np.arange(150), 10)\n'
'user_label = rng.integers(0, 2, 150)\n'
'y = user_label[groups]\n'
'X_id = OneHotEncoder(sparse_output=False).fit_transform(groups.reshape(-1, 1))\n'
'\n'
'# WRONG: random KFold — a user\'s rows land in both train and validation\n'
'fake = cross_val_score(LogisticRegression(max_iter=2000), X_id, y,\n'
'                       cv=KFold(5, shuffle=True, random_state=SEED),\n'
'                       scoring="roc_auc").mean()\n'
'\n'
'# RIGHT: GroupKFold keeps every user entirely on one side of the split\n'
'honest = cross_val_score(LogisticRegression(max_iter=2000), X_id, y,\n'
'                         groups=groups, cv=GroupKFold(5),\n'
'                         scoring="roc_auc").mean()\n'
'\n'
'SCORES["Random KFold\\non grouped rows"] = (fake, honest)\n'
'print(f"leaked AUC  = {fake:.3f}   <- perfect score by memorising user id")\n'
'print(f"honest AUC  = {honest:.3f}   <- unseen users are a coin flip")'
))

cells.append(md(
'A flawless **1.000** versus a coin-flip **0.500** — the widest gap in the\n'
'notebook, and the one that most often survives into real pipelines because the\n'
'code looks completely normal. If your rows have any entity behind them —\n'
'user, session, image series, molecule — ask whether that entity should be a\n'
'`groups=` argument. When in doubt, it should.\n'
'\n'
'The mechanism is worth quantifying, because it explains why the fake score was\n'
'not merely high but *perfect*: measure how many validation-fold users also\n'
'appear in the training fold under each splitter.'
))

cells.append(code(
'def straddling_users(cv, **kw):\n'
'    """Mean fraction of validation-fold users that also appear in the training fold."""\n'
'    fracs = []\n'
'    for tr, va in cv.split(X_id, y, **kw):\n'
'        overlap = np.intersect1d(np.unique(groups[tr]), np.unique(groups[va]))\n'
'        fracs.append(len(overlap) / len(np.unique(groups[va])))\n'
'    return float(np.mean(fracs))\n'
'\n'
'kf_frac = straddling_users(KFold(5, shuffle=True, random_state=SEED))\n'
'gk_frac = straddling_users(GroupKFold(5), groups=groups)\n'
'print(f"validation users also seen in training | KFold:      {kf_frac:.0%}")\n'
'print(f"validation users also seen in training | GroupKFold: {gk_frac:.0%}")'
))

cells.append(md(
'The observation that matters: under shuffled KFold **every single validation\n'
'user** also appears in training (with 10 rows per user, the odds of all 10\n'
'landing in one fold are negligible), so the model never has to generalise at\n'
'all. `GroupKFold` drives the overlap to exactly 0%. This two-line audit — count\n'
'entity overlap across your fold boundaries — is worth running on any dataset\n'
'where you even suspect a hidden grouping.'
))

# ── Cell 8: §6 leak 4 ─────────────────────────────────────────────────────────
cells.append(md(
'## 6. Leak 4 — Shuffled CV on a Time Series\n'
'\n'
'On temporal data, a shuffled KFold lets the model train on **future** points to\n'
'predict the **past** — it interpolates between validation neighbours instead of\n'
'forecasting. The score looks superb and is meaningless for anything you would\n'
'actually deploy (which only ever sees the past).\n'
'\n'
'**Setup:** a smooth seasonal + trend series with the time index as the only\n'
'feature. Shuffled CV can interpolate the curve; honest forward-chaining CV must\n'
'extrapolate it.'
))

cells.append(code(
'm = 1200\n'
't = np.arange(m)\n'
'y = np.sin(t / 30.0) * 5 + 0.01 * t + rng.standard_normal(m) * 0.4\n'
'X_t = t.reshape(-1, 1).astype(float)\n'
'\n'
'def gb():\n'
'    return GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=SEED)\n'
'\n'
'# WRONG: shuffled KFold — validation points sit between training points in time\n'
'fake = cross_val_score(gb(), X_t, y,\n'
'                       cv=KFold(5, shuffle=True, random_state=SEED),\n'
'                       scoring="r2").mean()\n'
'\n'
'# RIGHT: TimeSeriesSplit — always train on the past, validate on the future\n'
'honest = cross_val_score(gb(), X_t, y, cv=TimeSeriesSplit(5), scoring="r2").mean()\n'
'\n'
'SCORES["Shuffled CV\\non time series"] = (fake, honest)\n'
'print(f"shuffled R2 = {fake:.3f}   <- interpolating between future & past")\n'
'print(f"forward  R2 = {honest:.3f}   <- honest forecasting is far harder")'
))

cells.append(md(
'Shuffled CV reports a near-perfect R² ~0.99; the honest forward split reports a\n'
'**negative** R² — the model genuinely struggles to extrapolate the trend, which\n'
'is the real difficulty of forecasting. The shuffled number is not "slightly\n'
'optimistic," it is describing a task you will never actually face. Any data with\n'
'a time axis needs `TimeSeriesSplit` (or a manual date cutoff). The honest\n'
'scheme does carry a trade-off: early folds train on very little history, so\n'
'per-fold scores are noisier — prefer averaging over the later folds, or set\n'
'`test_size` so each fold is substantial.\n'
'\n'
'One picture makes the geometry of the two splits — and therefore the leak —\n'
'unmistakable:'
))

cells.append(code(
'tr_sh, va_sh = next(KFold(5, shuffle=True, random_state=SEED).split(X_t))\n'
'tr_ts, va_ts = list(TimeSeriesSplit(5).split(X_t))[-1]\n'
'\n'
'fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, sharey=True)\n'
'for ax, tr_i, va_i, name in [\n'
'    (axes[0], tr_sh, va_sh, "Shuffled KFold: every validation point (red) has training neighbours on both sides"),\n'
'    (axes[1], tr_ts, va_ts, "TimeSeriesSplit: validation (red) lies strictly in the future"),\n'
']:\n'
'    ax.plot(t[tr_i], y[tr_i], ".", ms=2, color="#9AA7B5", label="train")\n'
'    ax.plot(t[va_i], y[va_i], ".", ms=2, color="#D64550", label="validation")\n'
'    ax.set_title(name, loc="left", fontsize=10)\n'
'    ax.spines[["top", "right"]].set_visible(False)\n'
'axes[0].legend(frameon=False, markerscale=4, loc="upper left")\n'
'axes[1].set_xlabel("time index")\n'
'plt.tight_layout()\n'
'plt.show()'
))

# ── Cell 9: §7 leak 5 ─────────────────────────────────────────────────────────
cells.append(md(
'## 7. Leak 5 — Duplicate Rows Across Folds\n'
'\n'
'Duplicates and near-duplicates are everywhere: repeated records, augmented\n'
'copies, the same event logged twice. When a row and its copy land on opposite\n'
'sides of a random split, the model has literally seen the validation answer.\n'
'High-capacity models (KNN, trees) exploit it hardest.\n'
'\n'
'**Setup:** 500 random rows, each triplicated, shuffled. A 5-NN classifier finds\n'
'a row\'s own copies among its nearest neighbours.'
))

cells.append(code(
'n0 = 500\n'
'X_base = rng.standard_normal((n0, 20))\n'
'y_base = rng.integers(0, 2, n0)\n'
'X_dup = np.vstack([X_base, X_base, X_base])          # each row appears 3x\n'
'y_dup = np.concatenate([y_base, y_base, y_base])\n'
'order = rng.permutation(len(y_dup))\n'
'X_dup, y_dup = X_dup[order], y_dup[order]\n'
'\n'
'# WRONG: random split — copies of a row straddle the fold boundary\n'
'fake = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_dup, y_dup,\n'
'                       cv=KFold(5, shuffle=True, random_state=SEED),\n'
'                       scoring="roc_auc").mean()\n'
'\n'
'# RIGHT: de-duplicate before validating\n'
'_, uniq = np.unique(X_dup, axis=0, return_index=True)\n'
'honest = cross_val_score(KNeighborsClassifier(n_neighbors=5),\n'
'                         X_dup[uniq], y_dup[uniq], cv=5, scoring="roc_auc").mean()\n'
'\n'
'SCORES["Duplicate rows\\nacross folds"] = (fake, honest)\n'
'print(f"leaked AUC  = {fake:.3f}   <- neighbours are its own copies")\n'
'print(f"honest AUC  = {honest:.3f}   <- deduped, back to chance")'
))

cells.append(md(
'The duplicated set scores ~0.76 on noise because most validation rows have an\n'
'identical twin sitting in the training fold. De-duplicating drops it back to\n'
'~0.55 — essentially chance, since a 500-row noise AUC naturally wobbles a few\n'
'points around 0.50 (the leak, ~0.20 of AUC, is what actually mattered). Always\n'
'check `df.duplicated().sum()` — and be suspicious of near-duplicates from\n'
'augmentation or logging too. How exposed was the naive split, exactly? Count\n'
'the validation rows whose identical twin sits in the training fold:'
))

cells.append(code(
'twin_rates = []\n'
'for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X_dup):\n'
'    train_keys = {X_dup[i].tobytes() for i in tr}\n'
'    twin_rates.append(np.mean([X_dup[i].tobytes() in train_keys for i in va]))\n'
'\n'
'print(f"validation rows with an exact copy in the training fold: {np.mean(twin_rates):.1%}")\n'
'print("after de-duplication: 0.0% by construction")'
))

cells.append(md(
'Effectively **every** validation row had a twin on the training side — with\n'
'each row appearing three times, the chance that all copies land in the same\n'
'fold is tiny. One limitation of this audit worth stating plainly: `tobytes()`\n'
'catches only *exact* duplicates. Near-duplicates — the same image re-encoded,\n'
'the same reading logged a second apart — need a similarity check (hashing,\n'
'nearest-neighbour distance) and leak just as effectively.'
))

# ── Cell 10: §8 summary chart ─────────────────────────────────────────────────
cells.append(md(
'## 8. The Damage, Side by Side\n'
'\n'
'One chart from the five results above. Each pair shows the fake score the leak\n'
'produced against the honest score after the fix. The gap between them is,\n'
'literally, fiction — and it is exactly what evaporates on the leaderboard.'
))

cells.append(code(
'labels = list(SCORES.keys())\n'
'fake_scores = [SCORES[k][0] for k in labels]\n'
'honest_scores = [SCORES[k][1] for k in labels]\n'
'\n'
'y_pos = np.arange(len(labels))\n'
'h = 0.38\n'
'fig, ax = plt.subplots(figsize=(9, 5))\n'
'b1 = ax.barh(y_pos + h/2, fake_scores, height=h, color="#D64550",\n'
'             label="Leaked (fake)", zorder=3)\n'
'b2 = ax.barh(y_pos - h/2, honest_scores, height=h, color="#2E7CD6",\n'
'             label="Honest (fixed)", zorder=3)\n'
'ax.bar_label(b1, fmt="%.2f", padding=4, fontsize=9)\n'
'ax.bar_label(b2, fmt="%.2f", padding=4, fontsize=9)\n'
'ax.axvline(0.5, color="#555555", linewidth=1, linestyle="--", zorder=2)\n'
'ax.text(0.5, len(labels) - 0.4, " chance (AUC 0.5)", color="#666666", fontsize=9)\n'
'ax.set_yticks(y_pos, labels)\n'
'ax.invert_yaxis()\n'
'ax.set_xlabel("Cross-validation score (AUC; Leak 4 uses R\\u00b2 and is off this scale)")\n'
'ax.set_title("Five leaks: the fake score vs the honest score", loc="left")\n'
'ax.legend(loc="lower right", frameon=False)\n'
'ax.spines[["top", "right"]].set_visible(False)\n'
'ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)\n'
'plt.tight_layout()\n'
'plt.show()\n'
'\n'
'print("Leak                              fake     honest    gap")\n'
'for k in labels:\n'
'    f, hs = SCORES[k]\n'
'    tag = k.replace(chr(10), " ")\n'
'    print(f"{tag:<34s}{f:6.3f}   {hs:6.3f}   {f - hs:+.3f}")'
))

cells.append(md(
'Note the chart mixes AUC (Leaks 1-3, 5) with the time-series R² (Leak 4), so\n'
'read the time-series bar qualitatively — the point is direction and magnitude,\n'
'not a shared scale. Every red bar stands well above the 0.5 chance line while\n'
'its blue partner sits on it: **the entire height difference was leakage.**'
))

cells.append(md(
'### The one test that catches all five: shuffle the target\n'
'\n'
'Treat every good CV score as a **hypothesis to falsify**, not a result to\n'
'celebrate. The cheapest falsification: shuffle the target and re-run the exact\n'
'same pipeline. Real signal cannot survive shuffling, so any score above chance\n'
'is leakage — measured directly, with no theory required. To show it working in\n'
'both directions, this time we use data that **does** contain real signal:'
))

cells.append(code(
'# 600 rows, 3,000 features, of which the first two genuinely drive the target\n'
'n2, p2 = 600, 3000\n'
'X_sig = rng.standard_normal((n2, p2))\n'
'y_sig = (X_sig[:, 0] + X_sig[:, 1] + rng.standard_normal(n2) * 0.5 > 0).astype(int)\n'
'\n'
'pipe2 = Pipeline([("sel", SelectKBest(f_classif, k=20)),\n'
'                  ("clf", LogisticRegression(max_iter=1000))])\n'
'\n'
'real = cross_val_score(pipe2, X_sig, y_sig, cv=5, scoring="roc_auc").mean()\n'
'\n'
'y_shuf = rng.permutation(y_sig)   # break the X-y link; keep everything else\n'
'X_leaky = SelectKBest(f_classif, k=20).fit(X_sig, y_shuf).transform(X_sig)\n'
'leaky_shuf = cross_val_score(LogisticRegression(max_iter=1000), X_leaky, y_shuf,\n'
'                             cv=5, scoring="roc_auc").mean()\n'
'clean_shuf = cross_val_score(pipe2, X_sig, y_shuf, cv=5, scoring="roc_auc").mean()\n'
'\n'
'print(f"real target,     clean pipeline  AUC = {real:.3f}   <- genuine signal survives")\n'
'print(f"shuffled target, leaky selection AUC = {leaky_shuf:.3f}   <- \'signal\' surviving a shuffle = leak")\n'
'print(f"shuffled target, clean pipeline  AUC = {clean_shuf:.3f}   <- honest chance, as it must be")'
))

# ── Cell 11: §9 checklist ─────────────────────────────────────────────────────
cells.append(md(
'## 9. A Leak-Proofing Checklist\n'
'\n'
'Run this before you trust a CV number:\n'
'\n'
'- [ ] **Is every fitted step inside the fold?** Scalers, imputers, selectors,\n'
'      PCA, encoders — wrap them in a `Pipeline` so `cross_val_score` re-fits\n'
'      them per fold. Nothing that calls `.fit` should touch the full dataset.\n'
'- [ ] **Any target-derived feature computed out-of-fold?** Target/count/\n'
'      likelihood encodings must be built with an out-of-fold scheme, never a\n'
'      whole-column `groupby`.\n'
'- [ ] **Do rows have a hidden entity?** User, session, patient, image-series,\n'
'      molecule — if the label is a property of that entity, use `GroupKFold`\n'
'      (or `StratifiedGroupKFold`).\n'
'- [ ] **Is there a time axis?** Use `TimeSeriesSplit` or a hard date cutoff;\n'
'      never shuffle. Also confirm no feature (rolling mean, "days until X")\n'
'      secretly uses the future.\n'
'- [ ] **Duplicates checked?** `df.duplicated().sum()`, plus a look for\n'
'      near-duplicates from augmentation or double-logging.\n'
'- [ ] **Sanity test:** does a CV score above chance survive on a **shuffled\n'
'      target**? If a model "predicts" random labels, you have a leak. This one\n'
'      check would have caught all five leaks above.\n'
'- [ ] **Does local CV track the leaderboard?** If they move together across a\n'
'      few submissions, trust CV. If CV climbs while the board does not, stop and\n'
'      hunt for leakage before tuning anything else.\n'
'\n'
'The shuffled-target item is cheap enough to make permanent. Here it is as a\n'
'helper you can drop into any project\'s test suite:'
))

cells.append(code(
'def leak_smoke_test(estimator, X, y, cv=5, scoring="roc_auc", chance=0.5, tol=0.06, seed=SEED):\n'
'    """Fail loudly if `estimator` can score above chance on a shuffled target."""\n'
'    y_r = np.random.RandomState(seed).permutation(np.asarray(y))\n'
'    score = cross_val_score(estimator, X, y_r, cv=cv, scoring=scoring).mean()\n'
'    assert score < chance + tol, f"possible leak: shuffled-target score = {score:.3f}"\n'
'    return score\n'
'\n'
'ok = leak_smoke_test(pipe2, X_sig, y_sig)\n'
'print(f"clean pipeline passes the smoke test (shuffled-target AUC = {ok:.3f})")'
))

# ── Cell 12: §10 conclusion ───────────────────────────────────────────────────
cells.append(md(
'## 10. Conclusion\n'
'\n'
'**Takeaways**\n'
'\n'
'1. Leakage is one rule broken five ways: **something that learns from data saw\n'
'   the validation rows.** Fix it by fitting every such step inside the fold.\n'
'2. On data with zero real signal, each leak manufactured a convincing score —\n'
'   up to a perfect 1.00 — and each fix collapsed it back to chance. The gap was\n'
'   never skill; it was information bleed.\n'
'3. The **shuffled-target test** is the cheapest insurance you can buy: if a\n'
'   pipeline scores above chance on random labels, it leaks.\n'
'4. Trust CV only once it *moves with* the leaderboard. An honest 0.50 beats a\n'
'   fake 0.90, because you can actually improve on the honest one.\n'
'\n'
'**Next experiments to try on your own**\n'
'\n'
'- Add the shuffled-target check as an assertion in your own CV harness.\n'
'- Re-run Leak 3 with `StratifiedGroupKFold` to keep class balance *and* group\n'
'  integrity at once.\n'
'- Take a past competition where your CV and LB diverged and diagnose which of\n'
'  these five (or which leaky feature) was responsible.\n'
'\n'
'**Related notebooks in this series:**\n'
'\n'
'- Feature Engineering Cookbook: 50 Techniques\n'
'- Optuna Tuning: A Practical Kaggle Guide\n'
'- Polars on Kaggle: The Complete Speed Guide\n'
'\n'
'---\n'
'\n'
'**If this notebook saved you a leaderboard faceplant, please upvote!** Questions\n'
'and war stories welcome in the comments.\n'
'\n'
'*Lorenzo Scaturchio | July 2026*'
))

# ── Notebook assembly ─────────────────────────────────────────────────────────

write_notebook(cells, __file__, "data_leakage_cv_pitfalls.ipynb")
