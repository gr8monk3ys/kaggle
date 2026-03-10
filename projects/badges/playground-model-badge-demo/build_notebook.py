#!/usr/bin/env python3
"""Build a minimal competition notebook that loads a Kaggle model."""

from __future__ import annotations

import os as _os
import sys as _sys


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
        """# Playground Model Badge Demo

This competition notebook attaches a Kaggle Model Hub artifact and loads it at runtime.

- Competition: [Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
- Model source: `lorenzoscaturchio/iris-logistic-regression-badge-demo/ScikitLearn/scikit-baseline/2`
"""
    )
)

cells.append(
    code(
        """from pathlib import Path
import json

import joblib
import numpy as np

search_roots = [
    Path("/kaggle/input"),
    Path("/workspaces/kaggle/projects/badges/kaggle-model-badge-demo/artifacts"),
]

model_path = None
for root in search_roots:
    if root.exists():
        matches = sorted(root.rglob("iris_logreg.joblib"))
        if matches:
            model_path = matches[0]
            break

if model_path is None:
    raise FileNotFoundError("iris_logreg.joblib not found in Kaggle model inputs")

label_path = model_path.with_name("label_names.json")
model = joblib.load(model_path)
label_names = json.loads(label_path.read_text())

print("Loaded model from:", model_path)
print("Classes:", label_names)"""
    )
)

cells.append(
    code(
        """X_demo = np.array(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.9, 3.1, 5.4, 2.1],
    ]
)

predictions = model.predict(X_demo)
predicted_labels = [label_names[str(int(idx))] for idx in predictions]

for row, label in zip(X_demo.tolist(), predicted_labels):
    print({"features": row, "prediction": label})"""
    )
)

cells.append(
    md(
        """The notebook is intentionally minimal. Its purpose is to validate the end-to-end Kaggle model attachment flow inside a competition notebook."""
    )
)

write_notebook(cells, __file__, "playground_model_badge_demo.ipynb")
