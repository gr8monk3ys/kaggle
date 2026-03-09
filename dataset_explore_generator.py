#!/usr/bin/env python3
"""Generate rich EDA explore notebooks for dataset directories.

Reads each dataset's CSV + dataset-metadata.json to produce a ~20-25 cell
notebook with matplotlib/seaborn visualizations, replacing the generic 5-cell
template that was auto-generated.

Skips datasets that have hand-crafted build_notebook.py scripts (spotify-tracks,
mental-health-tech) since those generate domain-specific visualizations.

Usage
-----
    python3 dataset_explore_generator.py --dir datasets/credit-card-fraud
    python3 dataset_explore_generator.py --all
    python3 dataset_explore_generator.py --all --push

Invoked by: ./manage.sh build-explore-notebooks [--push]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_utils import code, md, write_notebook
from dataset_optimizer import analyze_csv, analyze_parquet
from kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"

# Datasets with custom build_notebook.py — skip these.
SKIP_DIRS = {"spotify-tracks", "mental-health-tech"}

# Common target column names to detect automatically.
TARGET_CANDIDATES = [
    "target", "label", "class", "Class", "is_fraud", "is_churned",
    "converted", "outcome", "survived", "treatment", "rating",
    "popularity", "grade", "score", "salary", "price",
]


# ---------------------------------------------------------------------------
# Column classification helpers
# ---------------------------------------------------------------------------

def _classify_columns(analysis: dict) -> dict:
    """Classify columns from analyze_csv output into numeric, categorical, etc."""
    numeric = []
    categorical = []
    high_cardinality = []
    id_like = []
    target = None

    for col in analysis.get("columns", []):
        name = col["name"]
        dtype = col["dtype"]
        n_unique = col["n_unique"]
        total = col["total"]

        # Detect ID-like columns
        if name.lower().endswith("_id") or name.lower() == "id":
            id_like.append(name)
            continue

        if dtype in ("integer", "float"):
            numeric.append(name)
        elif dtype == "boolean":
            categorical.append(name)
        elif dtype == "string":
            if n_unique <= 30:
                categorical.append(name)
            else:
                high_cardinality.append(name)

    # Detect target column
    for candidate in TARGET_CANDIDATES:
        lower_candidate = candidate.lower()
        for col in analysis.get("columns", []):
            if col["name"].lower() == lower_candidate:
                target = col["name"]
                break
        if target:
            break

    return {
        "numeric": numeric,
        "categorical": categorical,
        "high_cardinality": high_cardinality,
        "id_like": id_like,
        "target": target,
    }


def _find_csv_file(ds_dir: Path) -> Path | None:
    """Find the primary CSV (or Parquet) in a dataset directory."""
    csvs = sorted(ds_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    parquets = sorted(ds_dir.glob("*.parquet"))
    if parquets:
        return parquets[0]
    return None


# ---------------------------------------------------------------------------
# Cell generators
# ---------------------------------------------------------------------------

def _cell_title(meta: dict, analysis: dict) -> list[dict]:
    """Generate title + TOC cells."""
    title = meta.get("title", "Dataset Explorer")
    subtitle = meta.get("subtitle", "")
    dataset_id = meta.get("id", "")

    rows = analysis.get("rows", 0)
    n_cols = len(analysis.get("columns", []))
    file_name = analysis.get("file", "data.csv")

    header = f"# {title} — Complete EDA"
    if subtitle:
        header += f"\n> **{subtitle}**"
    if dataset_id:
        header += f" | [Dataset](https://www.kaggle.com/datasets/{dataset_id})"
    header += f"\n\n**{rows:,} rows · {n_cols} columns · `{file_name}`**"

    header += """

## Table of Contents
1. [Objective & Evaluation Plan](#objective)
2. [Setup & Data Loading](#setup)
3. [Data Overview](#overview)
4. [Missing Data Analysis](#missing)
5. [Numeric Distributions](#distributions)
6. [Categorical Analysis](#categorical)
7. [Correlation Analysis](#correlations)
8. [Target Analysis](#target)
9. [Evaluation Readiness](#evaluation)
10. [Key Findings](#findings)"""

    return [md(header)]


def _infer_modeling_plan(analysis: dict, classified: dict) -> tuple[str, str, str, str]:
    """Infer a lightweight modeling recommendation from dataset analysis."""
    target = classified["target"]
    if target:
        target_meta = next((col for col in analysis.get("columns", []) if col["name"] == target), {})
        dtype = str(target_meta.get("dtype", "")).lower()
        n_unique = int(target_meta.get("n_unique", 0) or 0)
        if dtype in {"integer", "float"} and n_unique > 20:
            return (
                "regression",
                "RMSE / MAE",
                "time-aware split or K-fold validation depending on row ordering",
                f"`{target}` behaves like a continuous target, so a regression baseline is the clean starting point.",
            )
        return (
            "classification",
            "F1 / ROC-AUC for imbalance, accuracy for balanced classes",
            "stratified train/validation split with class-ratio checks",
            f"`{target}` looks like a discrete target, so classification is the most defensible objective.",
        )

    if classified["numeric"]:
        return (
            "unsupervised exploration + candidate regression/classification framing",
            "define metric after target selection",
            "hold out a small validation slice once a target is chosen",
            "There is no obvious target yet, so the immediate goal is to surface hypotheses and shortlist modeling targets.",
        )

    return (
        "text/categorical exploration",
        "define metric after labeling a downstream task",
        "label a pilot sample before choosing validation",
        "This bundle is strongest for discovery, retrieval, and taxonomy design before supervised modeling.",
    )


def _cell_objective_and_evaluation(meta: dict, analysis: dict, classified: dict) -> list[dict]:
    """Generate notebook framing for objective, evaluation, and hypotheses."""
    rows = analysis.get("rows", 0)
    target = classified["target"] or "not yet fixed"
    task, metric, validation, framing = _infer_modeling_plan(analysis, classified)

    return [
        md(
            "## 1. Objective & Evaluation Plan <a id='objective'></a>\n\n"
            f"**Objective:** turn `{meta.get('title', 'this dataset')}` into a reproducible `{task}` workflow.\n\n"
            f"**Evaluation / validation:** use **{metric}** with **{validation}**.\n\n"
            f"**Candidate target:** `{target}`.\n\n"
            f"**Working hypothesis:** {framing}\n\n"
            "This section makes the modeling goal explicit so later findings can be interpreted in terms of metrics, "
            "trade-offs, and deployment limitations rather than isolated charts."
        ),
        code(
            f"""TARGET_COL = {target!r}
MODELING_TASK = {task!r}
PRIMARY_METRIC = {metric!r}
VALIDATION_PLAN = {validation!r}

print("Objective framing")
print("-" * 60)
print(f"Rows available      : {rows:,}")
print(f"Candidate target    : {{TARGET_COL}}")
print(f"Modeling task       : {{MODELING_TASK}}")
print(f"Primary metric      : {{PRIMARY_METRIC}}")
print(f"Validation approach : {{VALIDATION_PLAN}}")"""
        ),
    ]


def _cell_setup(ds_dir: Path, analysis: dict) -> list[dict]:
    """Generate setup + data loading cells."""
    file_name = analysis.get("file", "data.csv")
    dataset_sources = []

    # Read dataset_sources from kernel-metadata.json if available
    kernel_meta_path = ds_dir / "kernel-metadata.json"
    if kernel_meta_path.exists():
        try:
            km = json.loads(kernel_meta_path.read_text(encoding="utf-8"))
            dataset_sources = km.get("dataset_sources", [])
        except (json.JSONDecodeError, OSError):
            pass

    if dataset_sources:
        input_slug = dataset_sources[0]
        data_dir_line = f"DATA_DIR = '/kaggle/input/{input_slug.split('/')[-1]}'"
    else:
        data_dir_line = "DATA_DIR = '/kaggle/input'"

    read_func = "pd.read_csv" if file_name.endswith(".csv") else "pd.read_parquet"

    return [
        md("## 2. Setup & Data Loading <a id='setup'></a>"),
        code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

import os
{data_dir_line}
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

df = {read_func}(f'{{DATA_DIR}}/{file_name}')
print(f"Shape: {{df.shape[0]:,}} rows x {{df.shape[1]}} columns")
print(f"Memory: {{df.memory_usage(deep=True).sum() / 1e6:.1f}} MB")
df.head()"""),
    ]


def _cell_overview(analysis: dict) -> list[dict]:
    """Generate data overview cells."""
    return [
        md("## 3. Data Overview <a id='overview'></a>"),
        code("""print("Column Types:")
print(df.dtypes.value_counts().to_string())
print(f"\\nDuplicate rows: {df.duplicated().sum():,}")
print(f"Total missing values: {df.isnull().sum().sum():,}")
print()
df.describe().round(2)"""),
    ]


def _cell_missing_data() -> list[dict]:
    """Generate missing data visualization cells."""
    return [
        md("## 4. Missing Data Analysis <a id='missing'></a>"),
        code("""missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
missing_df = missing_df[missing_df['count'] > 0].sort_values('percent', ascending=False)

if len(missing_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
    colors = ['#e74c3c' if p > 20 else '#f39c12' if p > 5 else '#2ecc71'
              for p in missing_df['percent']]
    bars = ax.barh(missing_df.index, missing_df['percent'], color=colors, alpha=0.85)
    ax.set_xlabel('Missing (%)')
    ax.set_title('Missing Data by Column', fontweight='bold')
    for bar, pct in zip(bars, missing_df['percent']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found — dataset is complete!")"""),
    ]


def _cell_distributions(classified: dict) -> list[dict]:
    """Generate numeric distribution cells."""
    numeric = classified["numeric"]
    if not numeric:
        return [
            md("## 5. Numeric Distributions <a id='distributions'></a>"),
            md("*No numeric columns detected for distribution plots.*"),
        ]

    # Limit to first 12 numeric columns for readability
    cols_str = str(numeric[:12])

    return [
        md("## 5. Numeric Distributions <a id='distributions'></a>"),
        code(f"""NUMERIC_COLS = {cols_str}

n = len(NUMERIC_COLS)
ncols = min(3, n)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
if n == 1:
    axes = [axes]
else:
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for i, col in enumerate(NUMERIC_COLS):
    ax = axes[i]
    ax.hist(df[col].dropna(), bins=40, color='#3498db', alpha=0.8, edgecolor='white', linewidth=0.3)
    mean_val = df[col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {{mean_val:.2f}}')
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Numeric Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""),
        code(f"""# Box plots for outlier detection
fig, axes = plt.subplots(1, min(len(NUMERIC_COLS), 6), figsize=(min(len(NUMERIC_COLS), 6) * 3, 5))
if min(len(NUMERIC_COLS), 6) == 1:
    axes = [axes]

for ax, col in zip(axes if hasattr(axes, '__iter__') else [axes], NUMERIC_COLS[:6]):
    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.set_title(col, fontweight='bold', fontsize=10)

plt.suptitle('Box Plots — Outlier Detection', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""),
    ]


def _cell_categorical(classified: dict) -> list[dict]:
    """Generate categorical analysis cells."""
    categorical = classified["categorical"]
    if not categorical:
        return [
            md("## 6. Categorical Analysis <a id='categorical'></a>"),
            md("*No low-cardinality categorical columns detected.*"),
        ]

    cols_str = str(categorical[:8])

    return [
        md("## 6. Categorical Analysis <a id='categorical'></a>"),
        code(f"""CAT_COLS = {cols_str}

n = len(CAT_COLS)
ncols = min(2, n)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
if n == 1:
    axes = [axes]
else:
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
           '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

for i, col in enumerate(CAT_COLS):
    ax = axes[i]
    counts = df[col].value_counts().head(10)
    bars = ax.barh(counts.index.astype(str), counts.values,
                   color=palette[:len(counts)], alpha=0.85)
    ax.set_title(f'{{col}} (top {{min(10, len(counts))}})', fontweight='bold')
    ax.set_xlabel('Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{{val:,}}', va='center', fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""),
    ]


def _cell_correlations(classified: dict) -> list[dict]:
    """Generate correlation heatmap cells."""
    numeric = classified["numeric"]
    if len(numeric) < 2:
        return [
            md("## 7. Correlation Analysis <a id='correlations'></a>"),
            md("*Need at least 2 numeric columns for correlation analysis.*"),
        ]

    cols_str = str(numeric[:15])

    return [
        md("## 7. Correlation Analysis <a id='correlations'></a>"),
        code(f"""CORR_COLS = {cols_str}

corr = df[CORR_COLS].corr()

fig, ax = plt.subplots(figsize=(min(12, len(CORR_COLS) + 2), min(10, len(CORR_COLS) + 1)))
mask = np.triu(np.ones_like(corr), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, mask=mask,
            annot_kws={{'size': 9}}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()"""),
        code("""# Top absolute correlations (excluding self-correlations)
corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("Top 10 Feature Correlations:")
print(f"{'Feature A':<25} {'Feature B':<25} {'Correlation':>12}")
print("-" * 65)
for a, b, r in corr_pairs[:10]:
    arrow = '+' if r > 0 else '-'
    print(f"{a:<25} {b:<25} {arrow}{abs(r):>11.3f}")"""),
    ]


def _cell_target(classified: dict) -> list[dict]:
    """Generate target analysis cells (only if a target column is detected)."""
    target = classified["target"]
    if not target:
        return [
            md("## 8. Target Analysis <a id='target'></a>"),
            md("*No standard target column detected. Explore potential targets manually.*"),
        ]

    return [
        md(f"## 8. Target Analysis <a id='target'></a>\n\nDetected target column: **`{target}`**"),
        code(f"""target_col = '{target}'

print(f"Target: {{target_col}}")
print(f"Unique values: {{df[target_col].nunique()}}")
print(f"Null: {{df[target_col].isnull().sum()}}")
print()

if df[target_col].nunique() <= 20:
    print("Value counts:")
    print(df[target_col].value_counts().to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    counts = df[target_col].value_counts()
    ax1.bar(counts.index.astype(str), counts.values, color='#e74c3c', alpha=0.85)
    ax1.set_title(f'{{target_col}} Distribution', fontweight='bold')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Count')
    for x, y in zip(range(len(counts)), counts.values):
        ax1.text(x, y + max(counts) * 0.01, f'{{y:,}}', ha='center', fontsize=9)

    if len(counts) == 2:
        ax2.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
        ax2.set_title('Class Balance', fontweight='bold')
    else:
        pcts = (counts / counts.sum() * 100).round(1)
        ax2.barh(pcts.index.astype(str), pcts.values, color='#3498db', alpha=0.85)
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Class Proportions', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print(df[target_col].describe())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[target_col].dropna(), bins=50, color='#e74c3c', alpha=0.8, edgecolor='white')
    ax.axvline(df[target_col].median(), color='navy', linestyle='--',
               label=f"Median: {{df[target_col].median():.2f}}")
    ax.set_title(f'{{target_col}} Distribution', fontweight='bold')
    ax.set_xlabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.show()"""),
    ]


def _cell_evaluation_readiness(analysis: dict, classified: dict) -> list[dict]:
    """Generate a pre-modeling evaluation checklist."""
    task, metric, validation, _ = _infer_modeling_plan(analysis, classified)
    target = classified["target"] or "manual selection required"
    high_cardinality = classified["high_cardinality"][:5]
    text_hint = ", ".join(high_cardinality) if high_cardinality else "none flagged"

    return [
        md(
            "## 9. Evaluation Readiness <a id='evaluation'></a>\n\n"
            "Before modeling, translate the EDA into a validation plan. This keeps metric choices, leakage checks, "
            "and leaderboard expectations aligned with the problem structure."
        ),
        code(
            f"""TARGET_COL = {target!r}
MODELING_TASK = {task!r}
PRIMARY_METRIC = {metric!r}
VALIDATION_PLAN = {validation!r}
HIGH_CARD_TEXT = {text_hint!r}

print("Evaluation readiness checklist")
print("-" * 60)
print(f"Target candidate         : {{TARGET_COL}}")
print(f"Modeling task            : {{MODELING_TASK}}")
print(f"Primary metric           : {{PRIMARY_METRIC}}")
print(f"Validation plan          : {{VALIDATION_PLAN}}")
print(f"High-cardinality columns : {{HIGH_CARD_TEXT}}")
print("\\nRecommended baseline stack:")
if MODELING_TASK == "classification":
    print("- LogisticRegression / LightGBMClassifier with stratified validation")
elif MODELING_TASK == "regression":
    print("- Ridge / LightGBMRegressor with error analysis on residual tails")
else:
    print("- Clustering, retrieval, or weak-label experiments before supervised modeling")
print("\\nRisk checks:")
print("- Verify no leakage columns encode the target directly")
print("- Compare train/validation distributions before reading leaderboard movement")
print("- Document trade-offs, caveats, and limitations before feature expansion")"""
        ),
    ]


def _cell_quality_summary(analysis: dict) -> list[dict]:
    """Generate a data quality summary cell."""
    return [
        md("## 10. Key Findings <a id='findings'></a>"),
        code("""# Data quality summary
print("=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)
print(f"  Rows:            {len(df):,}")
print(f"  Columns:         {df.shape[1]}")
print(f"  Duplicates:      {df.duplicated().sum():,}")
print(f"  Total missing:   {df.isnull().sum().sum():,}")
print(f"  Complete rows:   {df.dropna().shape[0]:,} ({df.dropna().shape[0]/len(df)*100:.1f}%)")
print()

numeric_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"  Numeric columns:     {len(numeric_cols)}")
print(f"  Categorical columns: {len(cat_cols)}")
print()

if len(numeric_cols) > 0:
    print("Potential outliers (values > 3 std from mean):")
    for col in numeric_cols[:10]:
        z = (df[col] - df[col].mean()) / df[col].std()
        outliers = (z.abs() > 3).sum()
        if outliers > 0:
            print(f"  {col}: {outliers:,} rows ({outliers/len(df)*100:.1f}%)")"""),
        md("""### Insights, Trade-offs, and Next Steps
- **Insight:** the strongest opportunity usually comes from the small set of columns with the clearest business interpretation.
- **Observation:** missingness, skew, and high-cardinality text fields often drive the most useful feature engineering decisions.
- **Trade-off:** richer feature sets can improve metrics, but they also increase leakage risk and maintenance cost.
- **Limitation:** EDA alone cannot prove causality, so validation and error analysis should confirm every major hypothesis.
- **Hypothesis:** targeted feature engineering on the most informative fields should outperform a naive all-columns baseline.

### Next Steps
- Feature engineering: create interaction terms from correlated features
- Handle missing values: imputation strategy depends on missingness pattern (MCAR/MAR/MNAR)
- Scale numeric features before modeling (StandardScaler or RobustScaler for outliers)
- Encode categoricals: one-hot for low cardinality, target encoding for high cardinality
- Try baseline models: LogisticRegression/Ridge for linear, XGBoost/LightGBM for tree-based"""),
    ]


# ---------------------------------------------------------------------------
# Main notebook generator
# ---------------------------------------------------------------------------

def generate_explore_notebook(ds_dir: Path) -> list[dict]:
    """Generate a rich EDA notebook for a dataset directory.

    Returns the list of cells (empty list on failure).
    """
    meta_path = ds_dir / "dataset-metadata.json"
    if not meta_path.exists():
        print(f"  {YELLOW}SKIP{RESET} {ds_dir.name}: no dataset-metadata.json")
        return []

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  {RED}FAIL{RESET} {ds_dir.name}: bad dataset-metadata.json ({exc})")
        return []

    csv_path = _find_csv_file(ds_dir)
    if csv_path is None:
        print(f"  {YELLOW}SKIP{RESET} {ds_dir.name}: no CSV/Parquet files")
        return []

    # Analyze the CSV
    if csv_path.suffix == ".csv":
        analysis = analyze_csv(csv_path)
    else:
        analysis = analyze_parquet(csv_path)

    if "error" in analysis:
        print(f"  {RED}FAIL{RESET} {ds_dir.name}: {analysis['error']}")
        return []

    classified = _classify_columns(analysis)

    # Build cells
    cells: list[dict] = []
    cells.extend(_cell_title(meta, analysis))
    cells.extend(_cell_objective_and_evaluation(meta, analysis, classified))
    cells.extend(_cell_setup(ds_dir, analysis))
    cells.extend(_cell_overview(analysis))
    cells.extend(_cell_missing_data())
    cells.extend(_cell_distributions(classified))
    cells.extend(_cell_categorical(classified))
    cells.extend(_cell_correlations(classified))
    cells.extend(_cell_target(classified))
    cells.extend(_cell_evaluation_readiness(analysis, classified))
    cells.extend(_cell_quality_summary(analysis))

    return cells


def build_explore(ds_dir: Path, push: bool = False) -> bool:
    """Generate explore.ipynb for a dataset directory and optionally push."""
    print(f"  {BLUE}{ds_dir.name}{RESET}...")

    cells = generate_explore_notebook(ds_dir)
    if not cells:
        return False

    # Write notebook using build_utils
    out = write_notebook(cells, str(ds_dir / "build_stub.py"), "explore.ipynb")
    print(f"  {GREEN}OK{RESET} {len(cells)} cells → {out}")

    if push:
        import subprocess

        cli = kaggle_command()
        result = subprocess.run(
            [*cli, "kernels", "push", "-p", str(ds_dir)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  {GREEN}PUSHED{RESET}")
        else:
            msg = summarize_subprocess_error(result.stdout, result.stderr)
            print(f"  {RED}PUSH FAILED{RESET}: {msg}")
            return False

    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate rich EDA explore notebooks for datasets."
    )
    parser.add_argument("--dir", type=Path, default=None,
                        help="Single dataset directory to process.")
    parser.add_argument("--all", action="store_true",
                        help="Process all dataset directories.")
    parser.add_argument("--push", action="store_true",
                        help="Push notebooks to Kaggle after generating.")
    args = parser.parse_args(argv)

    if not args.all and not args.dir:
        parser.error("Specify --dir or --all")

    print(f"{BLUE}=== Dataset Explore Notebook Generator ==={RESET}\n")

    if args.dir:
        target = args.dir if args.dir.is_absolute() else ROOT / args.dir
        dirs = [target]
    else:
        dirs = sorted(
            d for d in DATASETS_DIR.iterdir()
            if d.is_dir() and d.name not in SKIP_DIRS
        )

    success = 0
    failed = 0
    skipped = 0

    for ds_dir in dirs:
        if ds_dir.name in SKIP_DIRS:
            print(f"  {YELLOW}SKIP{RESET} {ds_dir.name} (has custom build_notebook.py)")
            skipped += 1
            continue
        if build_explore(ds_dir, push=args.push):
            success += 1
        else:
            failed += 1

    print(f"\n{BLUE}=== Done ==={RESET}  "
          f"Built: {GREEN}{success}{RESET}  "
          f"Failed: {RED}{failed}{RESET}  "
          f"Skipped: {YELLOW}{skipped}{RESET}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
