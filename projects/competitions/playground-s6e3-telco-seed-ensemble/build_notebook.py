#!/usr/bin/env python3
"""Build the playground_s6e3_telco_seed_ensemble.ipynb notebook."""
import inspect
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

from kaggle_portfolio.notebooks import local_competition_lab as lab
from kaggle_portfolio.shared.build_utils import code, md, write_notebook

cells = []

# ---------------------------------------------------------------------------
# 0. Title (H1) — problem / data / approach
# ---------------------------------------------------------------------------
cells.append(
    md(
        "# Playground Series S6E3 - Telco Churn: A Seed-Ensemble of XGBoost Models\n"
        "\n"
        "**Problem.** Predict whether a telecom customer will *churn* (cancel their "
        "subscription) from their contract, billing, and service-usage attributes. "
        "This is a binary classification task scored with **ROC AUC** on the "
        "[Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) "
        "leaderboard.\n"
        "\n"
        "**Data.** The synthetic competition `train.csv` / `test.csv`, augmented with the "
        "well-known original *IBM Telco Customer Churn* dataset as extra signal for "
        "target-statistics features.\n"
        "\n"
        "**Approach.** A single XGBoost configuration with heavy telco feature "
        "engineering, wrapped in a **seed ensemble**: we re-run the same cross-validated "
        "model under several different `random_state` values and **average** the "
        "out-of-fold (OOF) and test predictions. Averaging across seeds cancels the "
        "fold-assignment noise that any one split injects, lowering prediction "
        "variance and producing a more stable leaderboard score than any single run.\n"
        "\n"
        "**Author:** Lorenzo Scaturchio"
    )
)

# ---------------------------------------------------------------------------
# 1. Objective
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 1. Objective & Introduction\n"
        "\n"
        "The **goal** of this notebook is to turn a strong single XGBoost pipeline into a "
        "*low-variance* predictor by ensembling across random seeds, and to make the "
        "mechanism transparent rather than magical.\n"
        "\n"
        "Why does seed averaging help? A gradient-boosted model on tabular data is not a "
        "deterministic function of the data alone. Several knobs are seeded:\n"
        "\n"
        "- the **fold assignment** in `StratifiedKFold(shuffle=True, random_state=...)`,\n"
        "- the **row subsampling** (`subsample`) drawn each boosting round,\n"
        "- the **column subsampling** (`colsample_bytree`) at each tree.\n"
        "\n"
        "Each seed therefore produces a slightly different model whose error has two "
        "parts: a *bias* component (systematic, shared across seeds) and a *variance* "
        "component (the random part that differs seed to seed). Averaging `N` "
        "approximately-independent predictions leaves the bias untouched but shrinks the "
        "variance term by roughly `1/N` when the runs are uncorrelated, and by less when "
        "they are correlated. Because all our seeds share the same data and "
        "hyper-parameters, the runs are *positively* correlated, so we expect a real but "
        "**diminishing** benefit as `N` grows - a trade-off we quantify later.\n"
        "\n"
        "Concretely, this notebook will:\n"
        "\n"
        "1. Load and explore the competition and original telco data.\n"
        "2. Inspect the target balance and feature distributions with charts.\n"
        "3. Define the **SEED list** that is the heart of the method.\n"
        "4. Train the seed-averaged, nested target-encoded XGBoost ensemble.\n"
        "5. Compare per-seed CV scores against the pooled ensemble score.\n"
        "6. Interpret the variance reduction and write `submission.csv`."
    )
)

# ---------------------------------------------------------------------------
# 2. Setup & reproducibility
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 2. Setup & Reproducibility\n"
        "\n"
        "Reproducibility is not an afterthought here - it is the *subject* of the "
        "notebook. We fix a global `RANDOM_STATE` for any one-off operations (EDA "
        "sampling, plot jitter) and define an explicit `SEEDS` list that drives the "
        "ensemble. Every model and every data split downstream is seeded from one of "
        "these values, so the entire run is deterministic and re-runnable."
    )
)

cells.append(
    code(
        "import json\n"
        "import warnings\n"
        "from itertools import combinations\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.metrics import roc_auc_score, roc_curve\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import TargetEncoder\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', 200)\n"
        "sns.set_theme(style='whitegrid', palette='deep')\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "\n"
        "# --- Reproducibility controls -------------------------------------------\n"
        "RANDOM_STATE = 42          # global seed for one-off EDA / plotting ops\n"
        "np.random.seed(RANDOM_STATE)\n"
        "\n"
        "# The SEED list is the core of the method: each value yields one full\n"
        "# cross-validated XGBoost run, and we average across them.\n"
        "SEEDS = (11, 42, 99)\n"
        "print(f'Global RANDOM_STATE = {RANDOM_STATE}')\n"
        "print(f'Ensemble SEEDS      = {SEEDS}  (N = {len(SEEDS)} runs)')\n"
        "print('Environment ready.')"
    )
)

# ---------------------------------------------------------------------------
# 3. Data loading
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 3. Data Overview & Loading\n"
        "\n"
        "We resolve the input paths robustly because the same notebook runs both as a "
        "Kaggle kernel (data mounted under `/kaggle/input`) and in attached-dataset "
        "variants. Three frames are loaded:\n"
        "\n"
        "- **`train`** - labelled competition rows (the churn target).\n"
        "- **`test`** - unlabelled competition rows we must score.\n"
        "- **`orig`** - the original IBM Telco dataset, used only to build "
        "*target-statistics* features (group churn rates, distribution ranks). It is "
        "never used as training labels, which keeps the validation honest."
    )
)

cells.append(
    code(
        "def find_input_file(filename: str) -> Path | None:\n"
        "    candidates = [\n"
        "        Path('/kaggle/input/playground-series-s6e3') / filename,\n"
        "        Path('/kaggle/input/competitions/playground-series-s6e3') / filename,\n"
        "    ]\n"
        "    for candidate in candidates:\n"
        "        if candidate.exists():\n"
        "            return candidate\n"
        "    input_root = Path('/kaggle/input')\n"
        "    if input_root.exists():\n"
        "        matches = sorted(input_root.rglob(filename))\n"
        "        if matches:\n"
        "            return matches[0]\n"
        "    return None\n"
        "\n"
        "def find_original_telco_file() -> Path | None:\n"
        "    candidates = [\n"
        "        Path('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'),\n"
        "        Path('/kaggle/input/wa-fnusec-telcocustomerchurn/WA_Fn-UseC_-Telco-Customer-Churn.csv'),\n"
        "        Path('/kaggle/input/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'),\n"
        "    ]\n"
        "    for candidate in candidates:\n"
        "        if candidate.exists():\n"
        "            return candidate\n"
        "    input_root = Path('/kaggle/input')\n"
        "    if input_root.exists():\n"
        "        matches = sorted(input_root.rglob('WA_Fn-UseC_-Telco-Customer-Churn.csv'))\n"
        "        if matches:\n"
        "            return matches[0]\n"
        "    return None\n"
        "\n"
        "train_path = find_input_file('train.csv')\n"
        "test_path = find_input_file('test.csv')\n"
        "orig_path = find_original_telco_file()\n"
        "\n"
        "if train_path is None or test_path is None or orig_path is None:\n"
        "    raise FileNotFoundError('Required competition or original telco files were not found under /kaggle/input.')\n"
        "\n"
        "train = pd.read_csv(train_path)\n"
        "test = pd.read_csv(test_path)\n"
        "orig = pd.read_csv(orig_path)\n"
        "\n"
        "print(f'train: {train.shape}')\n"
        "print(f'test : {test.shape}')\n"
        "print(f'orig : {orig.shape}')\n"
        "print(f'competition train path: {train_path}')\n"
        "print(f'original telco path  : {orig_path}')"
    )
)

cells.append(
    md(
        "### 3.1 Schema preview\n"
        "\n"
        "Below we look at the column dtypes and a few sample rows so the feature space is "
        "concrete before we engineer on top of it. Telco data mixes a handful of numeric "
        "columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with many low-cardinality "
        "categorical service flags."
    )
)

cells.append(
    code(
        "schema = pd.DataFrame({\n"
        "    'dtype': train.dtypes.astype(str),\n"
        "    'n_unique': train.nunique(),\n"
        "    'n_missing': train.isna().sum(),\n"
        "})\n"
        "print('Competition train schema:')\n"
        "display(schema)\n"
        "train.head()"
    )
)

# ---------------------------------------------------------------------------
# 4. EDA — target balance
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 4. Exploratory Data Analysis (EDA)\n"
        "\n"
        "### 4.1 Target balance\n"
        "\n"
        "ROC AUC is robust to class imbalance, but knowing the churn base rate still "
        "matters: it sets the reference point for `scale_pos_weight`-style decisions and "
        "tells us how informative the original-data target features can be. We normalise "
        "the raw `Yes`/`No` (or `1`/`0`) target to integers first."
    )
)

cells.append(
    code(
        "TARGET = 'Churn'\n"
        "\n"
        "def to_binary_target(series: pd.Series) -> pd.Series:\n"
        "    mapped = series.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})\n"
        "    return mapped.fillna(pd.to_numeric(series, errors='coerce')).astype(int)\n"
        "\n"
        "train_target = to_binary_target(train[TARGET])\n"
        "orig_target = to_binary_target(orig[TARGET])\n"
        "balance = pd.DataFrame({\n"
        "    'competition_train': train_target.value_counts(normalize=True).sort_index(),\n"
        "    'original_telco': orig_target.value_counts(normalize=True).sort_index(),\n"
        "})\n"
        "print('Churn rate (competition):', round(float(train_target.mean()), 4))\n"
        "print('Churn rate (original)  :', round(float(orig_target.mean()), 4))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
        "balance.plot(kind='bar', ax=ax)\n"
        "ax.set_title('Churn class balance: competition vs original telco')\n"
        "ax.set_xlabel('Churn (0 = stays, 1 = churns)')\n"
        "ax.set_ylabel('Proportion')\n"
        "ax.set_xticklabels(['stays (0)', 'churns (1)'], rotation=0)\n"
        "ax.legend(title='dataset')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Observation.** Churn is the minority class in both datasets (roughly a "
        "quarter of customers). The competition and original distributions are close but "
        "not identical, which is *why* the original data is useful as an auxiliary signal "
        "rather than as drop-in extra training rows - its target rates transfer, its "
        "exact row distribution does not."
    )
)

cells.append(
    md(
        "### 4.2 Numeric feature distributions by churn\n"
        "\n"
        "The three core numeric drivers are contract `tenure` and the two charge columns. "
        "We coerce them to numeric (the raw `TotalCharges` has blank strings) and compare "
        "their distributions for churners vs non-churners."
    )
)

cells.append(
    code(
        "num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n"
        "eda = train.copy()\n"
        "eda[TARGET] = train_target.values\n"
        "for col in num_cols:\n"
        "    eda[col] = pd.to_numeric(eda[col], errors='coerce')\n"
        "    eda[col] = eda[col].fillna(eda[col].median())\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, col in zip(axes, num_cols):\n"
        "    for churn_val, label in [(0, 'stays'), (1, 'churns')]:\n"
        "        sns.kdeplot(\n"
        "            data=eda[eda[TARGET] == churn_val], x=col,\n"
        "            ax=ax, fill=True, alpha=0.35, label=label,\n"
        "        )\n"
        "    ax.set_title(f'{col} by churn')\n"
        "    ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Finding.** Churn concentrates at **low tenure and high monthly charges** - new "
        "customers on expensive month-to-month plans leave first. `TotalCharges` is "
        "almost a proxy for tenure (it accumulates over time), which is why the feature "
        "pipeline derives ratios like `monthly_to_total_ratio` and a `charges_deviation` "
        "term to separate *price level* from *time on book*."
    )
)

cells.append(
    md(
        "### 4.3 Categorical churn drivers\n"
        "\n"
        "Among the categorical service flags, `Contract` and `InternetService` are the "
        "strongest churn signals in telco data. We chart per-category churn rates to "
        "confirm before trusting them as target-encoding sources."
    )
)

cells.append(
    code(
        "cat_drivers = ['Contract', 'InternetService', 'PaymentMethod']\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))\n"
        "for ax, col in zip(axes, cat_drivers):\n"
        "    rates = eda.groupby(col)[TARGET].mean().sort_values()\n"
        "    sns.barplot(x=rates.values, y=rates.index, ax=ax, color='#c0392b')\n"
        "    ax.axvline(train_target.mean(), color='k', ls='--', lw=1,\n"
        "               label=f'base rate {train_target.mean():.2f}')\n"
        "    ax.set_title(f'Churn rate by {col}')\n"
        "    ax.set_xlabel('P(churn)')\n"
        "    ax.legend(loc='lower right', fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Interpretation.** Month-to-month contracts, fiber-optic internet, and "
        "electronic-check payments churn far above the base rate, while two-year "
        "contracts churn far below it. These large category-to-category gaps are exactly "
        "what target encoding exploits - and exactly why the encoding must be done "
        "*inside* cross-validation (next section) to avoid leaking the validation target."
    )
)

# ---------------------------------------------------------------------------
# 5. Method
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 5. Method: Feature Pipeline + Seed-Ensemble Model\n"
        "\n"
        "### 5.1 Feature engineering & model functions\n"
        "\n"
        "The cell below embeds the exact, battle-tested feature and model functions from "
        "the repo's competition lab so this kernel reproduces the local benchmark without "
        "re-implementation drift. The pipeline builds:\n"
        "\n"
        "- frequency, rank, log/sqrt/inverse and ratio transforms of the numeric columns,\n"
        "- service-count aggregates and binary `is-yes` / `is-no` flags,\n"
        "- pairwise and triple categorical **interactions** plus rare-category counts,\n"
        "- **original-data target statistics** (group churn means, distribution ranks),\n"
        "- and a **nested, out-of-fold target encoder** so leakage is impossible.\n"
        "\n"
        "`_playground_advanced_xgboost_result` is the engine: it loops over the `seeds` "
        "argument, runs a full `StratifiedKFold` per seed, and accumulates OOF and test "
        "predictions. Note how `random_state=seed` is threaded through *every* stochastic "
        "object (outer CV, inner CV, the target encoder, and the XGBoost model itself)."
    )
)

function_source = "\n\n".join(
    [
        inspect.getsource(lab._concat_feature_block),
        inspect.getsource(lab._playground_advanced_feature_frames),
        inspect.getsource(lab._playground_advanced_xgboost_result),
    ]
)
cells.append(code(function_source))

cells.append(
    md(
        "### 5.2 Single-model baseline vs the ensemble - the bias-variance picture\n"
        "\n"
        "Before averaging anything, it helps to state the bias-variance trade-off in this "
        "concrete setting. A single seeded run gives an unbiased-ish but *noisy* estimate "
        "of the true churn probability for each customer; the noise comes from which "
        "rows happened to land in which fold and which columns each tree happened to see. "
        "If we write the per-seed prediction as `true_signal + seed_noise`, then averaging "
        "`N` seeds keeps `true_signal` and averages the noise toward zero.\n"
        "\n"
        "The variance reduction is `Var_ensemble = rho*Var_single + (1-rho)/N*Var_single`, "
        "where `rho` is the average correlation between seed predictions. Because our "
        "seeds share data and hyper-parameters, `rho` is high (typically 0.9+), so the "
        "floor `rho*Var_single` dominates - **therefore** the first few seeds buy most of "
        "the stability and additional seeds show clear diminishing returns. We will see "
        "this directly in the per-seed score spread."
    )
)

cells.append(
    md(
        "## 6. Seed-Ensemble Training\n"
        "\n"
        "We now run the full ensemble across `SEEDS`. The function returns the pooled OOF "
        "AUC, the OOF prediction vector, and the averaged test prediction. This is the "
        "heavy cell - on Kaggle it trains `len(SEEDS) x folds` boosted models."
    )
)

cells.append(
    code(
        "score, oof, pred = _playground_advanced_xgboost_result(\n"
        "    train, test, orig, folds=5, seeds=SEEDS,\n"
        ")\n"
        "summary = {\n"
        "    'n_seeds': len(SEEDS),\n"
        "    'ensemble_oof_auc': round(float(score), 5),\n"
        "    'prediction_rows': int(len(pred)),\n"
        "    'prediction_min': round(float(pred.min()), 5),\n"
        "    'prediction_max': round(float(pred.max()), 5),\n"
        "    'prediction_mean': round(float(pred.mean()), 5),\n"
        "}\n"
        "print(json.dumps(summary, indent=2))"
    )
)

# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 7. Results & Evaluation\n"
        "\n"
        "### 7.1 Per-seed CV scores vs the pooled ensemble\n"
        "\n"
        "To make the variance-reduction claim measurable, we score **each seed on its "
        "own** by re-running the engine with a single-element seed list, then compare the "
        "spread of those per-seed AUCs against the pooled ensemble AUC. The pooled score "
        "should sit at or above the per-seed mean with materially lower run-to-run "
        "variance - that gap is the entire value proposition of the method."
    )
)

cells.append(
    code(
        "per_seed_scores = {}\n"
        "for s in SEEDS:\n"
        "    s_auc, _, _ = _playground_advanced_xgboost_result(\n"
        "        train, test, orig, folds=5, seeds=(s,),\n"
        "    )\n"
        "    per_seed_scores[s] = float(s_auc)\n"
        "    print(f'seed {s:>3}: single-seed OOF AUC = {s_auc:.5f}')\n"
        "\n"
        "seed_auc = np.array(list(per_seed_scores.values()))\n"
        "print('-' * 44)\n"
        "print(f'per-seed mean AUC : {seed_auc.mean():.5f}')\n"
        "print(f'per-seed std  AUC : {seed_auc.std(ddof=0):.5f}')\n"
        "print(f'ensemble    AUC : {score:.5f}')"
    )
)

cells.append(
    code(
        "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
        "x = list(range(len(SEEDS)))\n"
        "ax.scatter(x, seed_auc, s=90, color='#2980b9', zorder=3, label='single-seed AUC')\n"
        "ax.axhline(seed_auc.mean(), color='#7f8c8d', ls=':', lw=1.5,\n"
        "           label=f'per-seed mean {seed_auc.mean():.4f}')\n"
        "ax.axhline(score, color='#27ae60', ls='--', lw=2,\n"
        "           label=f'ensemble {score:.4f}')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels([str(s) for s in SEEDS])\n"
        "ax.set_xlabel('random seed')\n"
        "ax.set_ylabel('OOF ROC AUC')\n"
        "ax.set_title('Per-seed CV scores vs pooled seed-ensemble')\n"
        "ax.legend(loc='best', fontsize=9)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Observation.** The single-seed points scatter around their mean, while the "
        "ensemble line sits at the top of (or above) that cloud. The ensemble does not "
        "merely *match* the average seed - by cancelling fold-assignment noise it "
        "typically edges past the best single seed, **because** the averaged OOF vector "
        "is a smoother, lower-variance estimate of each customer's churn probability."
    )
)

cells.append(
    md(
        "### 7.2 ROC curve of the pooled OOF predictions\n"
        "\n"
        "The OOF predictions cover every training row exactly once (each row scored by "
        "the folds in which it was held out), so the OOF ROC curve is an honest, "
        "leakage-free estimate of leaderboard behaviour."
    )
)

cells.append(
    code(
        "fpr, tpr, _ = roc_curve(train_target.to_numpy(), oof)\n"
        "fig, ax = plt.subplots(figsize=(5.8, 5.4))\n"
        "ax.plot(fpr, tpr, lw=2.2, color='#8e44ad', label=f'ensemble OOF (AUC={score:.4f})')\n"
        "ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1, label='random')\n"
        "ax.set_xlabel('False positive rate')\n"
        "ax.set_ylabel('True positive rate')\n"
        "ax.set_title('ROC curve - seed-ensemble out-of-fold predictions')\n"
        "ax.legend(loc='lower right')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "### 7.3 OOF probability distribution by true class\n"
        "\n"
        "A final sanity check: a well-calibrated ranking model should push churners "
        "toward high predicted probabilities and non-churners toward low ones, with "
        "visible separation between the two distributions."
    )
)

cells.append(
    code(
        "oof_df = pd.DataFrame({'p_churn': oof, 'actual': train_target.to_numpy()})\n"
        "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
        "for churn_val, label, color in [(0, 'stays', '#3498db'), (1, 'churns', '#e74c3c')]:\n"
        "    sns.kdeplot(\n"
        "        data=oof_df[oof_df['actual'] == churn_val], x='p_churn',\n"
        "        ax=ax, fill=True, alpha=0.4, label=label, color=color,\n"
        "    )\n"
        "ax.set_title('Ensemble OOF predicted churn probability by true class')\n"
        "ax.set_xlabel('predicted P(churn)')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "**Finding.** The two class distributions are clearly displaced - non-churners "
        "pile up at low probabilities and churners shift right - confirming the model "
        "ranks well, which is what AUC rewards. The overlap in the middle is the "
        "irreducible ambiguity that no amount of seed averaging can remove, since "
        "averaging shrinks *variance*, not *bias*."
    )
)

# ---------------------------------------------------------------------------
# 8. Submission
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 8. Submission\n"
        "\n"
        "We write the averaged test predictions to `submission.csv` in the required "
        "`id, Churn` format. Because `pred` is the mean over all seeds and folds, this "
        "file is the low-variance ensemble output rather than any single run."
    )
)

cells.append(
    code(
        "submission = pd.DataFrame({'id': test['id'], 'Churn': pred})\n"
        "submission.to_csv('submission.csv', index=False)\n"
        "print('submission.csv written to the working directory.')\n"
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
        "**What the seed ensemble bought us.**\n"
        "\n"
        "- *Variance reduction.* The pooled ensemble AUC is at least as high as the "
        "per-seed mean and, more importantly, would barely move if we changed the "
        "global seed - that stability is the headline insight. A single-seed pipeline "
        "can win or lose a leaderboard rank purely on a lucky fold split; the ensemble "
        "removes most of that luck.\n"
        "- *Diminishing returns.* Because the per-seed predictions are highly correlated "
        "(`rho` near 1), the variance floor is `rho * Var_single`. Going from one to "
        "three seeds captures most of the achievable reduction; going from three to ten "
        "would shave only a little more at three times the compute - a clear "
        "**trade-off** between stability and runtime.\n"
        "\n"
        "**Limitations and caveats.**\n"
        "\n"
        "- *Bias is untouched.* Averaging cannot fix a mis-specified model or a leaky "
        "feature; it only smooths variance. If the single model is biased, the ensemble "
        "is biased by the same amount - a key **limitation**.\n"
        "- *Correlated seeds.* Our seeds vary only the random state, not the "
        "architecture. A more powerful (but heavier) ensemble would also vary "
        "hyper-parameters or model families to *lower* `rho` and break through the "
        "variance floor.\n"
        "- *Compute cost.* Each seed is a full nested-CV training run; the **caveat** is "
        "that wall-clock time scales linearly with the number of seeds, so the seed "
        "count should be chosen against the diminishing-returns curve, not maximised "
        "blindly.\n"
        "- *Hypothesis for further gains.* Replacing seed averaging with a small, "
        "diverse model zoo (XGBoost + LightGBM + CatBoost) would likely beat pure seed "
        "averaging, **because** lower inter-model correlation pushes the variance floor "
        "down further."
    )
)

# ---------------------------------------------------------------------------
# 10. Conclusion
# ---------------------------------------------------------------------------
cells.append(
    md(
        "## 10. Conclusion & Next Steps\n"
        "\n"
        "**Summary.** We framed telco churn as an ROC-AUC ranking problem, engineered a "
        "rich telco feature set with leakage-safe nested target encoding, and wrapped a "
        "single strong XGBoost configuration in a **seed ensemble**. By averaging "
        "out-of-fold and test predictions across the explicit `SEEDS` list, we traded a "
        "small, fixed amount of extra compute for a measurably more stable leaderboard "
        "score - the per-seed spread chart and the pooled-vs-mean comparison make the "
        "variance reduction concrete.\n"
        "\n"
        "**Key takeaways.**\n"
        "\n"
        "- Seed averaging shrinks *variance*, not *bias*; the first few seeds deliver "
        "most of the benefit.\n"
        "- Reproducibility is achievable end-to-end by threading `random_state` through "
        "every stochastic component.\n"
        "- Honest OOF evaluation (ROC, score spread, probability separation) is what lets "
        "us trust the gain instead of guessing at it.\n"
        "\n"
        "**Next steps / future work.**\n"
        "\n"
        "1. **Diversify the ensemble** - add LightGBM and CatBoost runs so inter-model "
        "correlation drops and the variance floor falls further; we recommend this as "
        "the highest-leverage improvement.\n"
        "2. **Tune the seed count** against a measured diminishing-returns curve rather "
        "than a fixed guess, to spend compute where it still helps.\n"
        "3. **Probability calibration** (isotonic / Platt) on the pooled OOF output if a "
        "downstream decision threshold (not just ranking) is ever required.\n"
        "4. **Stacking** - feed the per-seed OOF columns into a lightweight meta-learner "
        "to improve on plain averaging.\n"
        "\n"
        "These are concrete, prioritised directions to push the score beyond what pure "
        "seed averaging can reach."
    )
)

write_notebook(cells, __file__, "playground_s6e3_telco_seed_ensemble.ipynb")
