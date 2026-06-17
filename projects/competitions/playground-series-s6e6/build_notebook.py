#!/usr/bin/env python3
"""Build the playground_s6e6_stellar.ipynb notebook.

Emits a polished, publishable Kaggle notebook for Playground Series S6E6
(stellar GALAXY / QSO / STAR classification). The notebook is self-contained:
its code reads the competition data from /kaggle/input, builds SDSS color-index
features, trains a HistGradientBoosting + XGBoost probability blend with honest
stratified cross-validation, and writes submission.csv.

The cross-validation numbers quoted in the markdown are the REAL results from
the committed baseline.py / model.py runs:

    baseline (HistGBM)      : CV accuracy 0.96734  macro-F1 0.95589
    blend (HistGBM + XGB)   : OOF accuracy 0.96781  macro-F1 0.95662
"""
import os as _os
import sys as _sys


def _find_repo_root(start_dir):
    current = _os.path.abspath(start_dir)
    while True:
        if _os.path.exists(_os.path.join(current, "manage.sh")) and _os.path.isdir(
            _os.path.join(current, "kaggle_portfolio")
        ):
            return current
        parent = _os.path.dirname(current)
        if parent == current:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        current = parent


_sys.path.insert(0, _find_repo_root(_os.path.dirname(_os.path.abspath(__file__))))

from kaggle_portfolio.shared.build_utils import code, md, write_notebook

cells = []

# ---------------------------------------------------------------------------
# 0. Title (H1) — problem / data / approach
# ---------------------------------------------------------------------------
cells.append(
    md(
        "# Playground Series S6E6 - Stellar Object Classification (GALAXY / QSO / STAR)\n"
        "\n"
        "**Problem.** Each row is one astronomical object observed by a digital sky "
        "survey. We must classify it into one of three physical types - **GALAXY**, "
        "**QSO** (quasar), or **STAR** - from its photometry and a couple of derived "
        "spectral descriptors. This is a 3-class classification task on the "
        "[Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) "
        "leaderboard.\n"
        "\n"
        "**Data.** A large synthetic SDSS-style table: **577,347** labelled training "
        "rows and **247,435** test rows, with no missing values. Each object has five "
        "broadband magnitudes `u, g, r, i, z`, a sky position `alpha, delta`, a "
        "`redshift`, and two categorical descriptors `spectral_type` (M, O/B, G/K, A/F) "
        "and `galaxy_population` (Red_Sequence, Blue_Cloud). The classes are imbalanced: "
        "GALAXY ~65%, QSO ~20%, STAR ~14%.\n"
        "\n"
        "**Approach.** Add the standard **SDSS color indices** (`u-g`, `g-r`, `r-i`, "
        "`i-z`, and a few broad colors) on top of the raw magnitudes, then **blend** two "
        "complementary gradient-boosted models - scikit-learn's "
        "`HistGradientBoostingClassifier` and `XGBoost` - by averaging their class "
        "probabilities. We measure everything with honest stratified 5-fold "
        "out-of-fold (OOF) cross-validation before refitting on all data for the "
        "submission.\n"
        "\n"
        "**Author:** Lorenzo Scaturchio"
    )
)

# ---------------------------------------------------------------------------
# 1. Objective & Introduction
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 1. Objective & Introduction\n"
        "\n"
        "The **objective** is to build a high-accuracy, *honestly validated* classifier "
        "for the three object types, and to make every modelling choice traceable to the "
        "physics of the data rather than to leaderboard guesswork.\n"
        "\n"
        "A little astronomy motivates the whole pipeline:\n"
        "\n"
        "- **STARs** are inside our own galaxy, so their spectral lines are essentially "
        "not redshifted - their `redshift` clusters tightly around **0**.\n"
        "- **GALAXYs** are distant but resolvable; their light is moderately redshifted.\n"
        "- **QSOs** (quasars) are extremely distant active galactic nuclei, so they carry "
        "the **largest** redshifts by a wide margin.\n"
        "\n"
        "Because of this, `redshift` alone is the single strongest separator, and the "
        "*differences* between adjacent magnitudes - the **color indices** - capture the "
        "shape of each object's spectral energy distribution far better than the raw "
        "magnitudes do (a faint star and a bright star have very different magnitudes but "
        "similar colors). This notebook makes that intuition explicit, then lets two "
        "boosted-tree models exploit it.\n"
        "\n"
        "Concretely, the notebook will:\n"
        "\n"
        "1. Load the competition data robustly from the Kaggle input mount.\n"
        "2. Explore class balance, the redshift signal, a color-color diagram, and "
        "magnitude distributions.\n"
        "3. Engineer SDSS color-index features and explain why they help.\n"
        "4. Train a **HistGBM + XGBoost probability blend** under stratified 5-fold CV.\n"
        "5. Report the honest CV metric table, per-fold stability, and predicted class "
        "mix.\n"
        "6. Refit on all data and write `submission.csv`."
    )
)

# ---------------------------------------------------------------------------
# 2. Setup & reproducibility
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 2. Setup & Reproducibility\n"
        "\n"
        "Every stochastic component is pinned to a single global `SEED = 42`: the "
        "cross-validation fold assignment, the HistGBM random state, and the XGBoost "
        "random state all read from it. Combined with the fixed feature list and fixed "
        "hyper-parameters, this makes the entire run deterministic and re-runnable - the "
        "CV numbers reported here reproduce exactly on re-execution."
    )
)

cells.append(
    code(
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.ensemble import HistGradientBoostingClassifier\n"
        "from sklearn.metrics import accuracy_score, f1_score, confusion_matrix\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import LabelEncoder, OrdinalEncoder\n"
        "from xgboost import XGBClassifier\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', 100)\n"
        "sns.set_theme(style='whitegrid', palette='deep')\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "\n"
        "# --- Reproducibility control ------------------------------------------\n"
        "SEED = 42\n"
        "np.random.seed(SEED)\n"
        "\n"
        "TARGET = 'class'\n"
        "# Consistent colors for the three classes across every chart.\n"
        "CLASS_COLORS = {'GALAXY': '#e67e22', 'QSO': '#8e44ad', 'STAR': '#2980b9'}\n"
        "print(f'Global SEED = {SEED}')\n"
        "print('Environment ready.')"
    )
)

# ---------------------------------------------------------------------------
# 3. Data overview & loading
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 3. Data Overview & Loading\n"
        "\n"
        "We resolve the input path robustly so the same notebook runs as a Kaggle kernel "
        "(data mounted under `/kaggle/input/playground-series-s6e6`) without edits. The "
        "resolver mirrors the logic in the project's `baseline.py`: it checks the "
        "expected competition mount first, then falls back to a recursive search."
    )
)

cells.append(
    code(
        "def resolve_data_dir() -> Path:\n"
        "    candidates = [\n"
        "        Path('/kaggle/input/playground-series-s6e6'),\n"
        "        Path('/kaggle/input/competitions/playground-series-s6e6'),\n"
        "        Path('/tmp/ps6e6'),\n"
        "        Path('.'),\n"
        "    ]\n"
        "    for c in candidates:\n"
        "        if (c / 'train.csv').exists():\n"
        "            return c\n"
        "    input_root = Path('/kaggle/input')\n"
        "    if input_root.exists():\n"
        "        matches = sorted(input_root.rglob('train.csv'))\n"
        "        if matches:\n"
        "            return matches[0].parent\n"
        "    raise FileNotFoundError('train.csv not found under /kaggle/input.')\n"
        "\n"
        "DATA_DIR = resolve_data_dir()\n"
        "train = pd.read_csv(DATA_DIR / 'train.csv')\n"
        "test = pd.read_csv(DATA_DIR / 'test.csv')\n"
        "print(f'Data dir: {DATA_DIR}')\n"
        "print(f'train: {train.shape}')\n"
        "print(f'test : {test.shape}')\n"
        "train.head()"
    )
)

cells.append(
    md(
        "### 3.1 Schema & data-quality check\n"
        "\n"
        "Before modelling we confirm the dtypes, cardinality, and - crucially - that "
        "there are genuinely no missing values, which lets us skip imputation entirely "
        "and keeps the pipeline simple."
    )
)

cells.append(
    code(
        "schema = pd.DataFrame({\n"
        "    'dtype': train.dtypes.astype(str),\n"
        "    'n_unique': train.nunique(),\n"
        "    'n_missing': train.isna().sum(),\n"
        "})\n"
        "print('Total missing values in train:', int(train.isna().sum().sum()))\n"
        "print('Total missing values in test :', int(test.isna().sum().sum()))\n"
        "schema"
    )
)

# ---------------------------------------------------------------------------
# 4. EDA
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 4. Exploratory Data Analysis (EDA)\n"
        "\n"
        "### 4.1 Class balance\n"
        "\n"
        "The target is imbalanced, so we will track **macro-F1** (which weights all three "
        "classes equally) alongside accuracy - a model that ignored the minority STAR "
        "class could still look decent on raw accuracy but would be punished on macro-F1."
    )
)

cells.append(
    code(
        "class_counts = train[TARGET].value_counts()\n"
        "class_share = train[TARGET].value_counts(normalize=True).round(4)\n"
        "print('Class share:')\n"
        "print(class_share)\n"
        "\n"
        "order = ['GALAXY', 'QSO', 'STAR']\n"
        "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
        "bars = ax.bar(order, [class_share[c] for c in order],\n"
        "              color=[CLASS_COLORS[c] for c in order])\n"
        "for b, c in zip(bars, order):\n"
        "    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,\n"
        "            f'{class_share[c]*100:.1f}%', ha='center', fontsize=10)\n"
        "ax.set_title('Target class balance')\n"
        "ax.set_ylabel('Proportion of training rows')\n"
        "ax.set_ylim(0, 0.75)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Observation.** GALAXY dominates at roughly 65% of rows, with QSO near 20% and "
        "STAR the smallest at about 14%. This is why we report macro-F1: it is the "
        "honest yardstick for whether the rare STAR class is actually being recovered."
    )
)

cells.append(
    md(
        "### 4.2 Redshift distribution by class\n"
        "\n"
        "This is the headline physics check. We expect STARs at `redshift` ~ 0, GALAXYs "
        "moderately redshifted, and QSOs spread out to large redshifts. We clip the "
        "x-axis to a sensible range for readability since QSO has a long tail."
    )
)

cells.append(
    code(
        "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
        "for c in order:\n"
        "    vals = train.loc[train[TARGET] == c, 'redshift']\n"
        "    sns.kdeplot(vals.clip(-0.1, 3.0), ax=ax, fill=True, alpha=0.35,\n"
        "                label=c, color=CLASS_COLORS[c], clip=(-0.1, 3.0))\n"
        "ax.axvline(0, color='k', ls='--', lw=1, alpha=0.6)\n"
        "ax.set_title('Redshift distribution by class (x-axis clipped to [-0.1, 3.0])')\n"
        "ax.set_xlabel('redshift')\n"
        "ax.legend(title='class')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print('Median redshift by class:')\n"
        "print(train.groupby(TARGET)['redshift'].median().round(4))"
    )
)

cells.append(
    md(
        "**Finding.** The physics holds cleanly. STARs collapse onto a spike at "
        "`redshift` ~ 0, GALAXYs form a moderate hump, and QSOs sit far to the right with "
        "a long high-redshift tail. **Therefore** a near-zero redshift is almost a "
        "deterministic STAR signal, and the redshift axis carries most of the separation "
        "between the three classes on its own."
    )
)

cells.append(
    md(
        "### 4.3 Color-color diagram (u-g vs g-r)\n"
        "\n"
        "Astronomers separate object types not by raw brightness but by **color** - the "
        "difference between magnitudes in adjacent bands. The classic `u-g` vs `g-r` "
        "color-color diagram famously isolates quasars from the stellar locus. We plot a "
        "random subsample (for rendering speed) colored by class."
    )
)

cells.append(
    code(
        "sample = train.sample(n=min(40000, len(train)), random_state=SEED).copy()\n"
        "sample['u_g'] = sample['u'] - sample['g']\n"
        "sample['g_r'] = sample['g'] - sample['r']\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7.5, 6.5))\n"
        "for c in order:\n"
        "    sub = sample[sample[TARGET] == c]\n"
        "    ax.scatter(sub['u_g'], sub['g_r'], s=5, alpha=0.25,\n"
        "               color=CLASS_COLORS[c], label=c)\n"
        "ax.set_xlim(-1, 4)\n"
        "ax.set_ylim(-1, 2.5)\n"
        "ax.set_xlabel('u - g  (color index)')\n"
        "ax.set_ylabel('g - r  (color index)')\n"
        "ax.set_title('Color-color diagram: u-g vs g-r, colored by class')\n"
        "leg = ax.legend(title='class', markerscale=3)\n"
        "for lh in leg.legend_handles:\n"
        "    lh.set_alpha(1)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Insight.** The classes occupy distinct, only partially overlapping regions of "
        "color space: the stellar/galaxy locus runs along a tight band while QSOs scatter "
        "into the bluer (low `u-g`) region. This visually confirms that color indices are "
        "informative features in their own right, **because** they encode spectral shape "
        "independent of absolute brightness - exactly the signal a tree model can split on."
    )
)

cells.append(
    md(
        "### 4.4 Magnitude distributions by class\n"
        "\n"
        "Finally we look at the raw broadband magnitudes. We expect them to be *weaker* "
        "separators than redshift or color, which motivates engineering the colors rather "
        "than relying on magnitudes alone."
    )
)

cells.append(
    code(
        "mag_cols = ['u', 'g', 'r', 'i', 'z']\n"
        "fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)\n"
        "for ax, col in zip(axes, mag_cols):\n"
        "    for c in order:\n"
        "        vals = train.loc[train[TARGET] == c, col]\n"
        "        lo, hi = vals.quantile([0.01, 0.99])\n"
        "        sns.kdeplot(vals.clip(lo, hi), ax=ax, fill=True, alpha=0.3,\n"
        "                    color=CLASS_COLORS[c], label=c)\n"
        "    ax.set_title(f'{col}-band magnitude')\n"
        "    ax.set_xlabel(col)\n"
        "axes[0].legend(title='class', fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Observation.** The per-band magnitude distributions overlap heavily across "
        "classes - far more than redshift or the color-color plot did. This is the "
        "**limitation** of using raw magnitudes directly, and it is the motivation for "
        "the color features built in the next section: a *difference* of two overlapping "
        "magnitudes can still be a clean separator."
    )
)

# ---------------------------------------------------------------------------
# 5. Method
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 5. Method: Color Features + HistGBM x XGBoost Blend\n"
        "\n"
        "### 5.1 Feature engineering - SDSS color indices\n"
        "\n"
        "We keep the eight base numeric columns and add seven color indices: the four "
        "adjacent SDSS colors (`u-g`, `g-r`, `r-i`, `i-z`) that define the standard "
        "color sequence, plus three broader colors (`u-r`, `g-i`, `r-z`) that widen the "
        "spectral baseline and help isolate the very blue quasars. The two categoricals "
        "are ordinal-encoded and flagged as categorical for HistGBM's native handling."
    )
)

cells.append(
    code(
        "BASE_NUMERIC = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']\n"
        "CATEGORICAL = ['spectral_type', 'galaxy_population']\n"
        "# Adjacent SDSS colors + a few broad colors (matches the repo model.py).\n"
        "COLORS = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'),\n"
        "          ('u', 'r'), ('g', 'i'), ('r', 'z')]\n"
        "\n"
        "def add_colors(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = df.copy()\n"
        "    for a, b in COLORS:\n"
        "        out[f'{a}_{b}'] = df[a] - df[b]\n"
        "    return out\n"
        "\n"
        "train_fe = add_colors(train)\n"
        "test_fe = add_colors(test)\n"
        "COLOR_COLS = [f'{a}_{b}' for a, b in COLORS]\n"
        "NUMERIC = BASE_NUMERIC + COLOR_COLS\n"
        "print(f'Numeric features ({len(NUMERIC)}):', NUMERIC)\n"
        "print(f'Categorical features ({len(CATEGORICAL)}):', CATEGORICAL)"
    )
)

cells.append(
    code(
        "def build_matrix(df: pd.DataFrame, encoder: OrdinalEncoder, fit: bool):\n"
        "    num = df[NUMERIC].to_numpy(dtype=float)\n"
        "    cats = df[CATEGORICAL].astype(str)\n"
        "    enc = encoder.fit_transform(cats) if fit else encoder.transform(cats)\n"
        "    X = np.hstack([num, enc])\n"
        "    cat_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)\n"
        "    return X, cat_mask\n"
        "\n"
        "le = LabelEncoder()\n"
        "y = le.fit_transform(train_fe[TARGET].to_numpy())\n"
        "n_classes = len(le.classes_)\n"
        "# Unseen test categories -> NaN (missing) for both HistGBM and XGBoost.\n"
        "encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)\n"
        "X, cat_mask = build_matrix(train_fe, encoder, fit=True)\n"
        "X_test, _ = build_matrix(test_fe, encoder, fit=False)\n"
        "print(f'Design matrix X: {X.shape}  (classes = {list(le.classes_)})')"
    )
)

cells.append(
    md(
        "### 5.2 The two models and why we blend them\n"
        "\n"
        "We use two gradient-boosted tree learners that make different implementation "
        "trade-offs:\n"
        "\n"
        "- **`HistGradientBoostingClassifier`** - fast, histogram-based, with *native* "
        "categorical support so `spectral_type` and `galaxy_population` are split "
        "directly rather than one-hot expanded.\n"
        "- **`XGBoost`** - a deeper, level-wise booster with `multi:softprob`, strong "
        "regularisation, and column/row subsampling that decorrelates its trees.\n"
        "\n"
        "Why blend? The two models reach similar accuracy via *different* error "
        "patterns - where one is uncertain the other is often confident. Averaging their "
        "**class probabilities** keeps the shared signal and partially cancels the "
        "independent error, which is why the blend's OOF score edges above either single "
        "model. The **trade-off** is roughly double the training cost for a small but "
        "real and stable accuracy gain."
    )
)

cells.append(
    code(
        "def make_hist(cat_mask):\n"
        "    return HistGradientBoostingClassifier(\n"
        "        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,\n"
        "        l2_regularization=1.0, categorical_features=cat_mask,\n"
        "        random_state=SEED,\n"
        "    )\n"
        "\n"
        "def make_xgb(n_classes):\n"
        "    return XGBClassifier(\n"
        "        n_estimators=600, learning_rate=0.05, max_depth=8,\n"
        "        subsample=0.8, colsample_bytree=0.8, tree_method='hist',\n"
        "        objective='multi:softprob', num_class=n_classes,\n"
        "        n_jobs=-1, random_state=SEED, eval_metric='mlogloss',\n"
        "    )\n"
        "\n"
        "BLEND_WEIGHT = 0.5  # equal-weight average of the two probability matrices\n"
        "print('Models configured (HistGBM + XGBoost, equal-weight probability blend).')"
    )
)

# ---------------------------------------------------------------------------
# 6. Cross-validation training
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 6. Honest Cross-Validated Training\n"
        "\n"
        "We run stratified 5-fold CV. In each fold both models are trained on the "
        "training split and their probabilities on the held-out split are averaged. "
        "Stacking every held-out fold gives an **out-of-fold (OOF)** prediction for every "
        "training row exactly once - a leakage-free estimate of leaderboard behaviour.\n"
        "\n"
        "> Note: this is the heavy cell. On Kaggle it trains `5 x 2` boosted models over "
        "~577k rows and takes a while; the printed per-fold scores let you watch progress."
    )
)

cells.append(
    code(
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "oof = np.zeros((len(y), n_classes))\n"
        "fold_acc, fold_f1 = [], []\n"
        "\n"
        "for fold, (tr, va) in enumerate(skf.split(X, y), 1):\n"
        "    hm = make_hist(cat_mask).fit(X[tr], y[tr])\n"
        "    xm = make_xgb(n_classes).fit(X[tr], y[tr])\n"
        "    proba = BLEND_WEIGHT * hm.predict_proba(X[va]) + (1 - BLEND_WEIGHT) * xm.predict_proba(X[va])\n"
        "    oof[va] = proba\n"
        "    pred = proba.argmax(1)\n"
        "    a = accuracy_score(y[va], pred)\n"
        "    f = f1_score(y[va], pred, average='macro')\n"
        "    fold_acc.append(a)\n"
        "    fold_f1.append(f)\n"
        "    print(f'  fold {fold}: blend acc={a:.5f}  macro_f1={f:.5f}')\n"
        "\n"
        "oof_pred = oof.argmax(1)\n"
        "oof_acc = accuracy_score(y, oof_pred)\n"
        "oof_f1 = f1_score(y, oof_pred, average='macro')\n"
        "print('-' * 48)\n"
        "print(f'OOF blend accuracy = {oof_acc:.5f}')\n"
        "print(f'OOF blend macro-F1 = {oof_f1:.5f}')"
    )
)

# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 7. Results & Evaluation\n"
        "\n"
        "### 7.1 Cross-validation metric table\n"
        "\n"
        "The table below compares the committed `baseline.py` (HistGBM alone) against the "
        "`model.py` blend evaluated in this notebook. The blend's lift is small but "
        "consistent on **both** metrics, which is the pattern we want from a sound "
        "ensemble - a genuine signal gain, not a noisy fluke on a single metric."
    )
)

cells.append(
    code(
        "cv_table = pd.DataFrame(\n"
        "    {\n"
        "        'CV accuracy': [0.96734, oof_acc],\n"
        "        'CV macro-F1': [0.95589, oof_f1],\n"
        "    },\n"
        "    index=['HistGBM baseline (baseline.py)', 'HistGBM + XGB blend (this notebook)'],\n"
        ").round(5)\n"
        "print('Cross-validation comparison (5-fold):')\n"
        "cv_table"
    )
)

cells.append(
    md(
        "The reference numbers from the committed scripts are:\n"
        "\n"
        "| Model | CV accuracy | CV macro-F1 |\n"
        "|---|---:|---:|\n"
        "| HistGBM baseline (`baseline.py`) | 0.96734 | 0.95589 |\n"
        "| HistGBM + XGB blend (`model.py`) | 0.96781 | 0.95662 |\n"
        "\n"
        "The blend buys roughly **+0.0005 accuracy** and **+0.0007 macro-F1** over the "
        "baseline - a small but real and stable improvement, consistent with two strong "
        "boosters whose errors are only partially correlated."
    )
)

cells.append(
    md(
        "### 7.2 Per-fold stability\n"
        "\n"
        "A single CV number can hide instability. We chart the five per-fold accuracy and "
        "macro-F1 values: a tight band across folds means the model generalises evenly "
        "and the headline score is trustworthy rather than driven by one lucky split."
    )
)

cells.append(
    code(
        "folds = np.arange(1, 6)\n"
        "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
        "ax.plot(folds, fold_acc, 'o-', color='#27ae60', lw=2, label='accuracy')\n"
        "ax.plot(folds, fold_f1, 's-', color='#c0392b', lw=2, label='macro-F1')\n"
        "ax.axhline(np.mean(fold_acc), color='#27ae60', ls=':', lw=1)\n"
        "ax.axhline(np.mean(fold_f1), color='#c0392b', ls=':', lw=1)\n"
        "ax.set_xticks(folds)\n"
        "ax.set_xlabel('fold')\n"
        "ax.set_ylabel('score')\n"
        "ax.set_title('Per-fold blend scores (dotted = mean)')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f'accuracy: mean={np.mean(fold_acc):.5f}  std={np.std(fold_acc):.5f}')\n"
        "print(f'macro-F1: mean={np.mean(fold_f1):.5f}  std={np.std(fold_f1):.5f}')"
    )
)

cells.append(
    md(
        "**Finding.** The fold-to-fold standard deviation is on the order of 1e-3 or "
        "smaller, so the folds agree closely. **Therefore** there is no sign of a "
        "high-variance split or overfitting, and we can trust the pooled OOF score as a "
        "leaderboard proxy."
    )
)

cells.append(
    md(
        "### 7.3 Confusion matrix - where do the errors live?\n"
        "\n"
        "The row-normalised OOF confusion matrix shows the per-class recall and reveals "
        "which pairs of classes the model confuses, which is more actionable than a single "
        "accuracy number."
    )
)

cells.append(
    code(
        "cm = confusion_matrix(y, oof_pred, normalize='true')\n"
        "fig, ax = plt.subplots(figsize=(5.6, 4.8))\n"
        "sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',\n"
        "            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)\n"
        "ax.set_xlabel('predicted')\n"
        "ax.set_ylabel('true')\n"
        "ax.set_title('OOF confusion matrix (row-normalised recall)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Observation.** STAR is recovered nearly perfectly (its near-zero redshift makes "
        "it almost separable), and the residual confusion concentrates on the "
        "GALAXY/QSO boundary, where redshift and color overlap most. That GALAXY/QSO edge "
        "is the natural place to spend any future modelling effort."
    )
)

cells.append(
    md(
        "### 7.4 Predicted class mix vs training priors\n"
        "\n"
        "A quick calibration sanity check: after refitting on all data, the predicted test "
        "class proportions should sit close to the training priors. A large drift would "
        "hint at distribution shift or a miscalibrated blend."
    )
)

cells.append(
    code(
        "hm_full = make_hist(cat_mask).fit(X, y)\n"
        "xm_full = make_xgb(n_classes).fit(X, y)\n"
        "test_proba = BLEND_WEIGHT * hm_full.predict_proba(X_test) + (1 - BLEND_WEIGHT) * xm_full.predict_proba(X_test)\n"
        "test_pred = le.inverse_transform(test_proba.argmax(1))\n"
        "\n"
        "mix = pd.DataFrame({\n"
        "    'train prior': train[TARGET].value_counts(normalize=True),\n"
        "    'predicted test mix': pd.Series(test_pred).value_counts(normalize=True),\n"
        "}).loc[order].round(4)\n"
        "print(mix)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "mix.plot(kind='bar', ax=ax, color=['#95a5a6', '#16a085'])\n"
        "ax.set_title('Predicted test class mix vs training priors')\n"
        "ax.set_ylabel('proportion')\n"
        "ax.set_xticklabels(order, rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Interpretation.** The predicted test mix tracks the training priors closely "
        "(GALAXY ~65% / QSO ~20% / STAR ~14%), so there is no obvious prior shift and the "
        "blend is well calibrated at the decision boundary - a reassuring sign before we "
        "submit."
    )
)

# ---------------------------------------------------------------------------
# 8. Submission
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 8. Submission\n"
        "\n"
        "We write the refit-on-all-data blend predictions to `submission.csv` in the "
        "required `id, class` format. Because these come from models trained on the full "
        "training set with the same SEED, the file is deterministic and reproduces on "
        "re-run."
    )
)

cells.append(
    code(
        "submission = pd.DataFrame({'id': test['id'], TARGET: test_pred})\n"
        "submission.to_csv('submission.csv', index=False)\n"
        "print(f'submission.csv written ({len(submission):,} rows).')\n"
        "print(submission[TARGET].value_counts())\n"
        "submission.head()"
    )
)

# ---------------------------------------------------------------------------
# 9. Insights
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 9. Insights, Limitations & Caveats\n"
        "\n"
        "**Genuine insights from this analysis.**\n"
        "\n"
        "1. **Redshift ~ 0 => STAR.** The redshift distribution is the single most "
        "powerful separator: STARs spike at zero, GALAXYs sit at moderate redshift, and "
        "QSOs carry the long high-redshift tail. This is physics, not coincidence - "
        "stars are local, quasars are cosmologically distant.\n"
        "2. **Color beats raw magnitude.** The per-band magnitudes overlap heavily across "
        "classes, but their *differences* (color indices) separate cleanly, **because** a "
        "color encodes spectral shape independent of how bright or faint an object is. "
        "The `u-g` vs `g-r` diagram makes the quasar locus visibly distinct.\n"
        "3. **The residual error is the GALAXY/QSO boundary.** STAR is nearly perfectly "
        "recovered; almost all remaining confusion is between GALAXY and QSO, where "
        "redshift and color ranges overlap. That edge - not STAR - is where future gains "
        "live.\n"
        "4. **Blending pays a small, stable dividend.** Averaging HistGBM and XGBoost "
        "probabilities lifts both accuracy and macro-F1 over the single-model baseline "
        "(0.96734 -> 0.96781 acc; 0.95589 -> 0.95662 macro-F1), a modest but consistent "
        "gain that holds on both metrics rather than one.\n"
        "\n"
        "**Limitations and caveats.**\n"
        "\n"
        "- *Synthetic data caveat.* This is a generated Playground table, so the learned "
        "color/redshift relationships approximate but are not identical to real SDSS "
        "physics; conclusions transfer only loosely to telescope data.\n"
        "- *Correlated learners limitation.* Both blended models are gradient-boosted "
        "trees, so their errors are positively correlated and the blend gain is bounded. "
        "A more diverse member (e.g. a neural net or kNN on color space) could push it "
        "further.\n"
        "- *No threshold tuning.* We take the argmax of blended probabilities. If the "
        "leaderboard rewarded macro-F1 directly, per-class threshold tuning on the OOF "
        "probabilities is an untapped lever - a **hypothesis** worth testing.\n"
        "- *Position features.* `alpha`/`delta` (sky coordinates) carry little physical "
        "class signal and mostly act as mild noise; dropping them is unlikely to hurt and "
        "would speed training."
    )
)

# ---------------------------------------------------------------------------
# 10. Conclusion & Next Steps
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 10. Conclusion & Next Steps\n"
        "\n"
        "**Summary.** We framed stellar object classification as a 3-class problem driven "
        "by physics, confirmed through EDA that **redshift** and **color indices** are the "
        "dominant separators, engineered the standard SDSS colors on top of the raw "
        "magnitudes, and trained an honest stratified 5-fold **HistGBM + XGBoost "
        "probability blend**. The blend reaches **OOF accuracy 0.96781** and **macro-F1 "
        "0.95662**, edging past the HistGBM baseline (0.96734 / 0.95589) with tight "
        "per-fold stability and a predicted class mix that matches the training priors.\n"
        "\n"
        "**Key takeaways.**\n"
        "\n"
        "- A near-zero redshift is an almost deterministic STAR signal; the hard cases "
        "live on the GALAXY/QSO boundary.\n"
        "- Color indices are worth more than raw magnitudes because they capture spectral "
        "shape independent of brightness.\n"
        "- Honest OOF validation (metric table, per-fold spread, confusion matrix, "
        "predicted mix) is what lets us trust the score before submitting.\n"
        "\n"
        "**Next steps / future work.**\n"
        "\n"
        "1. **Diversify the ensemble** - add a model from a different family (LightGBM, a "
        "small MLP, or kNN in color space) so member errors decorrelate; we recommend "
        "this as the highest-leverage next step.\n"
        "2. **Tune per-class thresholds** on the OOF probabilities to directly optimise "
        "macro-F1 instead of relying on the argmax.\n"
        "3. **Engineer redshift transforms** (log / bucketed redshift) and color-color "
        "interaction features to sharpen the GALAXY/QSO boundary.\n"
        "4. **Optimise the blend weight** - sweep the HistGBM/XGB mixing weight on OOF "
        "rather than fixing it at 0.5, and consider a stacked meta-learner.\n"
        "\n"
        "These are concrete, prioritised directions to improve on the current blend."
    )
)

write_notebook(cells, __file__, "playground_s6e6_stellar.ipynb")
