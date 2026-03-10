#!/usr/bin/env python3
"""Build a tiny scikit-learn model artifact for Kaggle model badge flows."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(iris.data, iris.target)

    joblib.dump(model, out_dir / "iris_logreg.joblib")
    (out_dir / "label_names.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(iris.target_names)}, indent=2) + "\n"
    )
    (out_dir / "README.md").write_text(
        "# Iris Logistic Regression Artifact\n\n"
        "Tiny scikit-learn logistic regression model trained on the iris dataset.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
