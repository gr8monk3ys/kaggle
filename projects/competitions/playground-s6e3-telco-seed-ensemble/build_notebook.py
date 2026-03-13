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

cells.append(
    md(
        "# Playground Series S6E3: Telco Seed-Ensemble XGBoost\n"
        "\n"
        "**Competition:** [Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)  \n"
        "**Goal:** run the exact multi-seed telco XGBoost path from the repo on Kaggle so the heavy benchmark does not depend on local CPU.  \n"
        "**Author:** Lorenzo Scaturchio\n"
        "\n"
        "---\n"
        "\n"
        "## Plan\n"
        "\n"
        "1. Load the competition train/test and the original IBM telco churn dataset.\n"
        "2. Build the same advanced telco feature frame used in the local competition lab.\n"
        "3. Train the seed-averaged nested target-encoded XGBoost ensemble.\n"
        "4. Report the full OOF AUC and write `submission.csv`.\n"
        "\n"
        "This notebook is intentionally narrow: it is for leaderboard movement, not tutorial exposition."
    )
)

cells.append(md("## 1. Setup"))

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
        "from sklearn.metrics import roc_auc_score\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.preprocessing import TargetEncoder\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', 200)\n"
        "RANDOM_STATE = 42\n"
        "print('Environment ready.')"
    )
)

cells.append(md("## 2. Data Loading"))

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

function_source = "\n\n".join(
    [
        inspect.getsource(lab._concat_feature_block),
        inspect.getsource(lab._playground_advanced_feature_frames),
        inspect.getsource(lab._playground_advanced_xgboost_result),
    ]
)

cells.append(md("## 3. Embedded Competition-Lab Functions"))
cells.append(code(function_source))

cells.append(md("## 4. Seed Ensemble Training"))

cells.append(
    code(
        "score, oof, pred = _playground_advanced_xgboost_result(train, test, orig, folds=5)\n"
        "summary = {\n"
        "    'oof_auc': round(float(score), 5),\n"
        "    'prediction_rows': int(len(pred)),\n"
        "    'prediction_min': float(pred.min()),\n"
        "    'prediction_max': float(pred.max()),\n"
        "    'prediction_mean': float(pred.mean()),\n"
        "}\n"
        "print(json.dumps(summary, indent=2))"
    )
)

cells.append(md("## 5. Submission"))

cells.append(
    code(
        "submission = pd.DataFrame({'id': test['id'], 'Churn': pred})\n"
        "submission.to_csv('submission.csv', index=False)\n"
        "print('submission.csv written to the working directory.')\n"
        "submission.head()"
    )
)

write_notebook(cells, __file__, "playground_s6e3_telco_seed_ensemble.ipynb")
