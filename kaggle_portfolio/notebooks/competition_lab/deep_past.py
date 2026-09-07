"""Deep past: sequence modelling.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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


def _deep_past_normalize(text: str) -> str:
    normalized = str(text or "").lower()
    for old, new in {
        "…": " ",
        "...": " ",
        "„": " ",
        "“": " ",
        "”": " ",
        '"': " ",
        "'": " ",
        "`": " ",
        "´": " ",
        "{": " ",
        "}": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "/": " ",
        "\\": " ",
        ",": " ",
        ".": " ",
        ";": " ",
        ":": " ",
        "!": " ",
        "?": " ",
    }.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _deep_past_best_match(corpus: pd.Series, query: str) -> tuple[int, float]:
    normalized_corpus = corpus.fillna("").map(_deep_past_normalize)
    normalized_query = _deep_past_normalize(query)

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=1)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)

    char_matrix = char_vec.fit_transform(normalized_corpus)
    word_matrix = word_vec.fit_transform(normalized_corpus)
    char_score = linear_kernel(char_vec.transform([normalized_query]), char_matrix)[0]
    word_score = linear_kernel(word_vec.transform([normalized_query]), word_matrix)[0]
    scores = 0.7 * char_score + 0.3 * word_score

    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])


def _deep_past_display_name_candidates(row: pd.Series) -> list[str]:
    candidates: list[str] = []
    for value in [row.get("label", ""), row.get("aliases", ""), row.get("note", "")]:
        raw = str(value or "").strip()
        if not raw or raw.lower() == "nan":
            continue
        parts = [part.strip() for part in raw.split("|")]
        for part in parts:
            if not part:
                continue
            candidates.append(part)
            stripped = re.sub(
                r"^cuneiform\s+(tablet|envelope)\s+", "", part, flags=re.IGNORECASE
            ).strip()
            if stripped and stripped != part:
                candidates.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _deep_past_optional_csv(
    data_dir: Path,
    file_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    path = data_dir / file_name
    if not path.exists():
        return pd.DataFrame(columns=required_columns)
    frame = pd.read_csv(path)
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def _deep_past_sentence_rows(
    sentences: pd.DataFrame, published_row: pd.Series
) -> pd.DataFrame:
    if sentences.empty or "display_name" not in sentences:
        return sentences.iloc[0:0]
    display_names = sentences["display_name"].astype(str).str.strip()
    best = sentences.iloc[0:0]
    for candidate in _deep_past_display_name_candidates(published_row):
        matches = sentences.loc[display_names == candidate]
        if len(matches) > len(best):
            best = matches
    if best.empty:
        return best
    return (
        best.loc[:, ["line_number", "translation"]]
        .dropna(subset=["line_number", "translation"])
        .sort_values("line_number")
        .reset_index(drop=True)
    )


def _deep_past_assign_sentences_to_rows(
    test: pd.DataFrame, sentence_rows: pd.DataFrame
) -> list[str]:
    ordered_test = test.sort_values(["line_start", "line_end"]).reset_index(drop=True)
    ordered_sentences = sentence_rows.sort_values("line_number").reset_index(drop=True)
    predictions: list[str] = []

    for idx, row in ordered_test.iterrows():
        start = int(row["line_start"])
        next_start = (
            int(ordered_test.loc[idx + 1, "line_start"])
            if idx + 1 < len(ordered_test)
            else None
        )
        if next_start is None:
            mask = ordered_sentences["line_number"] >= start
        else:
            mask = (ordered_sentences["line_number"] >= start) & (
                ordered_sentences["line_number"] < next_start
            )
        translation = " ".join(
            ordered_sentences.loc[mask, "translation"].astype(str)
        ).strip()
        predictions.append(translation)

    return predictions


def _deep_past_split_translation_by_rows(text: str, test: pd.DataFrame) -> list[str]:
    weights = (
        (
            test.sort_values(["line_start", "line_end"])["line_end"]
            .fillna(test["line_start"])
            .astype(int)
            - test.sort_values(["line_start", "line_end"])["line_start"].astype(int)
            + 1
        )
        .clip(lower=1)
        .tolist()
    )
    words = str(text or "").split()
    if not words:
        return ["" for _ in weights]

    total_weight = sum(weights) or len(weights)
    chunks: list[str] = []
    position = 0
    for idx, weight in enumerate(weights):
        remaining_words = len(words) - position
        remaining_groups = len(weights) - idx
        if idx == len(weights) - 1:
            take = remaining_words
        else:
            take = max(1, round(len(words) * weight / total_weight))
            take = min(take, remaining_words - (remaining_groups - 1))
        chunks.append(" ".join(words[position : position + take]).strip())
        position += take
    return chunks


def _deep_past_train_retrieval(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> tuple[list[str], float]:
    query = " ".join(
        test.sort_values(["line_start", "line_end"])["transliteration"].astype(str)
    )
    best_idx, best_score = _deep_past_best_match(train["transliteration"], query)
    best_translation = str(train.iloc[best_idx]["translation"])
    predictions = _deep_past_split_translation_by_rows(best_translation, test)
    fallback = sample.sort_values("id")["translation"].astype(str).tolist()
    completed = [pred.strip() or fallback[idx] for idx, pred in enumerate(predictions)]
    return completed, best_score


def benchmark_deep_past(
    data_dir: Path, _folds: int, write_submission: bool
) -> LabResult:
    train = pd.read_csv(data_dir / "train.csv")
    test = (
        pd.read_csv(data_dir / "test.csv")
        .sort_values(["line_start", "line_end"])
        .reset_index(drop=True)
    )
    sample = (
        pd.read_csv(data_dir / "sample_submission.csv")
        .sort_values("id")
        .reset_index(drop=True)
    )
    published = _deep_past_optional_csv(
        data_dir, "published_texts.csv", ["transliteration", "label", "aliases", "note"]
    )
    sentences = _deep_past_optional_csv(
        data_dir,
        "Sentences_Oare_FirstWord_LinNum.csv",
        ["display_name", "line_number", "translation"],
    )

    query = " ".join(test["transliteration"].astype(str))
    published_available = (
        not published.empty
        and "transliteration" in published
        and published["transliteration"].notna().any()
    )
    sentences_available = (
        not sentences.empty
        and "display_name" in sentences
        and "translation" in sentences
        and "line_number" in sentences
    )
    published_score = 0.0
    published_row = pd.Series(dtype="object")
    sentence_rows = sentences.iloc[0:0]
    sentence_predictions: list[str] = []
    if published_available:
        published_idx, published_score = _deep_past_best_match(
            published["transliteration"], query
        )
        published_row = published.iloc[published_idx]
        if sentences_available:
            sentence_rows = _deep_past_sentence_rows(sentences, published_row)
            if not sentence_rows.empty:
                sentence_predictions = _deep_past_assign_sentences_to_rows(
                    test, sentence_rows
                )
    train_predictions, train_score = _deep_past_train_retrieval(train, test, sample)
    sentence_coverage = (
        sum(1 for pred in sentence_predictions if pred.strip()) / len(test)
        if len(sentence_predictions) == len(test)
        else 0.0
    )
    published_decision_score = min(1.0, published_score + 0.15 * sentence_coverage)

    benchmarks: list[dict[str, Any]] = [
        {
            "model": "published_sentence_match",
            "score": round(published_decision_score, 5),
            "source_label": str(published_row.get("label", "")),
            "available": bool(published_available and sentences_available),
        },
        {
            "model": "train_retrieval",
            "score": round(train_score, 5),
        },
    ]

    chosen_model = "train_retrieval"
    chosen_score = train_score
    predictions = train_predictions
    if (
        published_score >= 0.6
        and len(sentence_predictions) == len(test)
        and all(pred.strip() for pred in sentence_predictions)
        and published_decision_score >= train_score
    ):
        chosen_model = "published_sentence_match"
        chosen_score = published_decision_score
        predictions = sentence_predictions

    submission_path = None
    if write_submission:
        submission_path = (
            _submission_dir("deep-past-initiative-machine-translation")
            / f"submission_{_safe_slug(chosen_model)}_{int(chosen_score * 100000)}.csv"
        )
        pd.DataFrame({"id": test["id"], "translation": predictions}).to_csv(
            submission_path, index=False
        )

    return LabResult(
        competition="deep-past-initiative-machine-translation",
        metric_name="decision_score",
        best_model=chosen_model,
        best_score=float(chosen_score),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
