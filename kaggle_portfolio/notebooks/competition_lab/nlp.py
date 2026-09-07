"""NLP disaster tweets: text classification.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


from kaggle_portfolio.notebooks.competition_lab.runner import (  # noqa: F401
    LabResult,
    _benchmark_dir,
    _concat_feature_block,
    _ensure_data,
    _print_benchmarks,
    _safe_slug,
    _save_summary,
    _submission_dir,
    _submit,
)

RANDOM_STATE = 42
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


def benchmark_nlp(data_dir: Path, folds: int, write_submission: bool) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["target"].astype(int)
    text_train = train["text"].fillna("")
    text_test = test["text"].fillna("")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    candidates: list[tuple[str, Pipeline]] = [
        (
            "word_lr",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            min_df=2,
                            max_features=60000,
                            sublinear_tf=True,
                        ),
                    ),
                    ("model", LogisticRegression(max_iter=2000, C=4.0)),
                ]
            ),
        ),
        (
            "char_lr",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=2,
                            max_features=90000,
                            sublinear_tf=True,
                        ),
                    ),
                    ("model", LogisticRegression(max_iter=2000, C=3.0)),
                ]
            ),
        ),
        (
            "cnb",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            min_df=2,
                            max_features=70000,
                            sublinear_tf=True,
                        ),
                    ),
                    ("model", ComplementNB(alpha=0.4)),
                ]
            ),
        ),
    ]

    benchmarks: list[dict[str, Any]] = []
    trained_predictions: dict[str, np.ndarray] = {}
    for name, pipe in candidates:
        scores = cross_val_score(pipe, text_train, y, cv=skf, scoring="f1", n_jobs=1)
        benchmarks.append({"model": name, "score": round(float(scores.mean()), 5)})
        pipe.fit(text_train, y)
        trained_predictions[name] = pipe.predict(text_test)

    best = max(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        submission_path = (
            _submission_dir("nlp-getting-started")
            / f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame(
            {"id": test["id"], "target": trained_predictions[best["model"]].astype(int)}
        ).to_csv(
            submission_path,
            index=False,
        )

    return LabResult(
        competition="nlp-getting-started",
        metric_name="f1",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
