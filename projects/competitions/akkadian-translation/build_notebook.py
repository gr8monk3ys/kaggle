#!/usr/bin/env python3
"""Build a competition-compliant starter notebook for Akkadian submission."""

import os as _os
import sys as _sys
from pathlib import Path


def _find_repo_root(start_dir: str) -> str:
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


cells: list[dict] = []

cells.append(
    md(
        """# Akkadian Translation Starter Submission
**Competition:** [Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
**Runtime:** competition-safe starter with internet disabled

## What this notebook does
- loads the real competition files from `/kaggle/input`
- verifies the train/test/sample schema quickly
- writes a valid `submission.csv`

This starter uses the organizer-provided sample submission as the first valid baseline so the notebook can be committed and submitted reliably inside the code-competition rules."""
    )
)

cells.append(md("## 1. Setup"))

cells.append(
    code(
        """from pathlib import Path

import pandas as pd

candidates = [
    Path('/kaggle/input/deep-past-initiative-machine-translation'),
    Path('/tmp/akkadian-live/extracted'),
    Path('.'),
]
data_dir = next((path for path in candidates if (path / 'train.csv').exists()), None)
if data_dir is None:
    raise FileNotFoundError('Competition files not found in /kaggle/input or local fallback paths')

print(f'Using data directory: {data_dir}')"""
    )
)

cells.append(md("## 2. Load Competition Files"))

cells.append(
    code(
        """train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample = pd.read_csv(data_dir / 'sample_submission.csv')

print('Train shape:', train.shape)
print('Test shape :', test.shape)
print('Sample shape:', sample.shape)
print('\\nTrain columns:', list(train.columns))
print('Test columns :', list(test.columns))
print('Sample columns:', list(sample.columns))

train.head(2)"""
    )
)

cells.append(md("## 3. Write Submission"))

cells.append(
    code(
        """# First valid baseline: use the organizer-provided sample submission.
# This keeps the notebook commit fast and competition-compliant.
submission = sample.copy()
submission.to_csv('submission.csv', index=False)

print('submission.csv written')
print(submission.head().to_string(index=False))"""
    )
)

cells.append(
    md(
        """## Next Step
After the commit finishes on Kaggle, use the competition notebook submit flow to send `submission.csv` to the leaderboard."""
    )
)

write_notebook(cells, __file__, "akkadian_submission_baseline.ipynb")
