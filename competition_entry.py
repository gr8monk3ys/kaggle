#!/usr/bin/env python3
"""Generate a complete competition entry scaffold from a competition slug.

Creates a directory with kernel-metadata.json and a starter EDA notebook,
ready for push to Kaggle.

Usage
-----
    python3 competition_entry.py spaceship-titanic
    python3 competition_entry.py spaceship-titanic --gpu
    python3 competition_entry.py spaceship-titanic --push

Invoked by: ./manage.sh create-competition-entry <slug> [--gpu] [--push]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from build_utils import code, md, write_notebook
from kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).parent
OWNER = "lorenzoscaturchio"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"

MAX_SLUG_LEN = 34

# Category detection keywords
NLP_KEYWORDS = {"nlp", "text", "language", "tweet", "sentiment", "bert", "llm",
                "translation", "ner", "qa", "question", "answer", "disaster"}
CV_KEYWORDS = {"image", "vision", "cnn", "segmentation", "detection", "x-ray",
               "medical", "radiology", "photo", "pixel", "digit", "mnist"}
TS_KEYWORDS = {"time series", "forecast", "sales", "stock", "temporal",
               "demand", "energy", "weather"}
TABULAR_KEYWORDS = {"tabular", "classification", "regression", "house",
                    "price", "titanic", "spaceship", "fraud"}


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

def make_slug(title: str) -> str:
    """Generate a Kaggle-compatible slug from a title (max MAX_SLUG_LEN chars).

    Slugifies by lowercasing, replacing non-alphanumeric with hyphens,
    collapsing runs of hyphens, and truncating at a word boundary.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) <= MAX_SLUG_LEN:
        return slug
    # Truncate at word boundary
    truncated = slug[:MAX_SLUG_LEN]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > MAX_SLUG_LEN // 2:
        truncated = truncated[:last_hyphen]
    return truncated.rstrip("-")


def detect_category(title: str) -> str:
    """Detect competition category from title keywords.

    Returns one of: 'nlp', 'cv', 'timeseries', 'tabular'.
    """
    title_lower = title.lower()
    if any(kw in title_lower for kw in NLP_KEYWORDS):
        return "nlp"
    if any(kw in title_lower for kw in CV_KEYWORDS):
        return "cv"
    if any(kw in title_lower for kw in TS_KEYWORDS):
        return "timeseries"
    return "tabular"


# ---------------------------------------------------------------------------
# Competition metadata fetching
# ---------------------------------------------------------------------------

def fetch_competition_info(slug: str) -> dict | None:
    """Fetch competition info from Kaggle CLI.

    Returns a dict with keys: ref, title, deadline, teamCount, category, etc.
    Returns None on failure.
    """
    import csv
    import io

    try:
        cli = kaggle_command()
        result = subprocess.run(
            [*cli, "competitions", "list", "--csv", "--page-size", "100",
             "--sort-by", "latestDeadline"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None

        for row in csv.DictReader(io.StringIO(result.stdout)):
            ref = row.get("ref", "")
            if slug in ref:
                return row

    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Notebook cell generators
# ---------------------------------------------------------------------------

def _generate_cells(slug: str, title: str, category: str, gpu: bool) -> list[dict]:
    """Generate starter EDA notebook cells based on competition category."""
    cells: list[dict] = []

    # Title cell
    cells.append(md(f"""# {title}
> Competition entry — [kaggle.com/competitions/{slug}](https://www.kaggle.com/competitions/{slug})

## Table of Contents
1. [Setup & Data Loading](#setup)
2. [Data Overview](#overview)
3. [Missing Data](#missing)
4. [Feature Distributions](#distributions)
5. [Correlation Analysis](#correlations)
6. [Baseline Model](#baseline)
7. [Submission](#submission)"""))

    # Setup cell — varies by category
    cells.append(md("## 1. Setup & Data Loading <a id='setup'></a>"))

    if category == "nlp":
        cells.append(code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

import os
DATA_DIR = '/kaggle/input/{slug}'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

train = pd.read_csv(f'{{DATA_DIR}}/train.csv')
test = pd.read_csv(f'{{DATA_DIR}}/test.csv')
print(f"Train: {{train.shape}}, Test: {{test.shape}}")
train.head()"""))
    elif category == "cv":
        cells.append(code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
{'import torch' if gpu else ''}
{'import torch.nn as nn' if gpu else ''}
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import os
DATA_DIR = '/kaggle/input/{slug}'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

# List available files
for root, dirs, files in os.walk(DATA_DIR):
    for f in files[:20]:
        print(os.path.join(root, f))"""))
    elif category == "timeseries":
        cells.append(code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import os
DATA_DIR = '/kaggle/input/{slug}'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

train = pd.read_csv(f'{{DATA_DIR}}/train.csv')
test = pd.read_csv(f'{{DATA_DIR}}/test.csv')
print(f"Train: {{train.shape}}, Test: {{test.shape}}")
train.head()"""))
    else:  # tabular
        cells.append(code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import os
DATA_DIR = '/kaggle/input/{slug}'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

train = pd.read_csv(f'{{DATA_DIR}}/train.csv')
test = pd.read_csv(f'{{DATA_DIR}}/test.csv')
print(f"Train: {{train.shape}}, Test: {{test.shape}}")
train.head()"""))

    # Data overview
    cells.append(md("## 2. Data Overview <a id='overview'></a>"))
    cells.append(code("""print("Column Types:")
print(train.dtypes.value_counts().to_string())
print(f"\\nDuplicate rows: {train.duplicated().sum():,}")
print(f"Total missing: {train.isnull().sum().sum():,}")
print()
train.describe().round(2)"""))

    # Missing data
    cells.append(md("## 3. Missing Data <a id='missing'></a>"))
    cells.append(code("""missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(1)
missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
missing_df = missing_df[missing_df['count'] > 0].sort_values('percent', ascending=False)

if len(missing_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
    colors = ['#e74c3c' if p > 20 else '#f39c12' if p > 5 else '#2ecc71'
              for p in missing_df['percent']]
    ax.barh(missing_df.index, missing_df['percent'], color=colors, alpha=0.85)
    ax.set_xlabel('Missing (%)')
    ax.set_title('Missing Data by Column', fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("No missing values — dataset is complete!")"""))

    # Distributions
    cells.append(md("## 4. Feature Distributions <a id='distributions'></a>"))
    cells.append(code("""numeric_cols = train.select_dtypes(include=['number']).columns.tolist()[:12]
n = len(numeric_cols)
if n > 0:
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.hist(train[col].dropna(), bins=40, color='#3498db', alpha=0.8,
                edgecolor='white', linewidth=0.3)
        ax.set_title(col, fontweight='bold')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found")"""))

    # Correlations
    cells.append(md("## 5. Correlation Analysis <a id='correlations'></a>"))
    cells.append(code("""numeric_cols = train.select_dtypes(include=['number']).columns.tolist()
if len(numeric_cols) >= 2:
    corr = train[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(min(12, len(numeric_cols) + 2),
                                     min(10, len(numeric_cols) + 1)))
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(corr, annot=len(numeric_cols) <= 15, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax, mask=mask)
    ax.set_title('Feature Correlations', fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric columns for correlation analysis")"""))

    # Baseline model placeholder
    cells.append(md("## 6. Baseline Model <a id='baseline'></a>"))
    cells.append(code("""# TODO: Implement baseline model
# Steps:
# 1. Identify target column
# 2. Handle missing values
# 3. Encode categoricals
# 4. Train/val split
# 5. Fit baseline model
# 6. Evaluate
print("Baseline model — implement based on competition requirements")"""))

    # Submission scaffold
    cells.append(md("## 7. Submission <a id='submission'></a>"))
    cells.append(code(f"""# TODO: Generate predictions and create submission
# submission = pd.DataFrame({{'id': test['id'], 'target': predictions}})
# submission.to_csv('submission.csv', index=False)
# print(f"Submission shape: {{submission.shape}}")
print("Submission scaffold — fill in after baseline model is complete")"""))

    return cells


# ---------------------------------------------------------------------------
# Directory + metadata creation
# ---------------------------------------------------------------------------

def make_kernel_metadata(slug: str, title: str, gpu: bool) -> dict:
    """Create kernel-metadata.json content for a competition entry."""
    nb_slug = make_slug(title)
    kernel_id = f"{OWNER}/{nb_slug}"

    return {
        "id": kernel_id,
        "title": title,
        "code_file": f"{slug.replace('/', '-')}_eda.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": gpu,
        "enable_internet": True,
        "keywords": ["eda", "competition", "beginner"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [slug],
    }


def create_entry(slug: str, *, gpu: bool = False, push: bool = False) -> bool:
    """Create a complete competition entry directory."""
    # Try to fetch competition info
    info = fetch_competition_info(slug)
    if info:
        raw_title = info.get("title", slug)
        title = f"{raw_title}: EDA & Baseline"
        print(f"  Found competition: {raw_title}")
    else:
        title = f"{slug.replace('-', ' ').title()}: EDA & Baseline"
        print(f"  {YELLOW}Competition not found via API — using slug as title{RESET}")

    # Detect category
    category = detect_category(title)
    print(f"  Category: {category}")

    # Create directory
    entry_dir = ROOT / slug
    if entry_dir.exists():
        print(f"  {YELLOW}Directory {slug}/ already exists — updating notebook{RESET}")
    else:
        entry_dir.mkdir(parents=True)
        print(f"  Created {slug}/")

    # Write kernel-metadata.json
    meta = make_kernel_metadata(slug, title, gpu)
    meta_path = entry_dir / "kernel-metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  {GREEN}kernel-metadata.json{RESET} created")
    else:
        print(f"  {YELLOW}kernel-metadata.json already exists — skipping{RESET}")
        # Re-read existing meta for code_file name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Generate notebook
    code_file = meta.get("code_file", f"{slug}_eda.ipynb")
    cells = _generate_cells(slug, title, category, gpu)
    write_notebook(cells, str(entry_dir / "stub.py"), code_file)
    print(f"  {GREEN}{code_file}{RESET} — {len(cells)} cells")

    if push:
        cli = kaggle_command()
        result = subprocess.run(
            [*cli, "kernels", "push", "-p", str(entry_dir)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  {GREEN}PUSHED{RESET}")
        else:
            msg = summarize_subprocess_error(result.stdout, result.stderr)
            print(f"  {RED}PUSH FAILED{RESET}: {msg}")
            print(f"  You may need to accept competition rules at:")
            print(f"  https://www.kaggle.com/competitions/{slug}")
            return False

    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a competition entry scaffold."
    )
    parser.add_argument("slug", help="Competition slug (e.g., spaceship-titanic)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU in kernel metadata.")
    parser.add_argument("--push", action="store_true",
                        help="Push to Kaggle after generating.")
    args = parser.parse_args(argv)

    print(f"{BLUE}=== Competition Entry Generator ==={RESET}\n")
    ok = create_entry(args.slug, gpu=args.gpu, push=args.push)

    if ok:
        print(f"\n{GREEN}Entry created!{RESET}")
        print(f"  Directory: {args.slug}/")
        print(f"  Next: edit the notebook, then run:")
        print(f"  ./manage.sh push {args.slug}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
