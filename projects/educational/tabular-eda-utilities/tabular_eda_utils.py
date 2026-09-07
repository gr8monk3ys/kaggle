"""Reusable helpers for tabular Kaggle notebooks and dataset EDA."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


def load_csv_rows(csv_path: str, limit: int | None = None) -> list[dict[str, str]]:
    """Load CSV rows into dictionaries for quick notebook prototyping."""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    return rows if limit is None else rows[:limit]


def detect_numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    """Return columns whose first non-empty value parses as float."""
    if not rows:
        return []
    numeric: list[str] = []
    for key in rows[0]:
        for row in rows:
            value = row.get(key, "").strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                break
            numeric.append(key)
            break
    return numeric


def numeric_profile(rows: list[dict[str, str]], columns: list[str]) -> dict[str, dict[str, float]]:
    """Return count/min/max/mean for a selected list of numeric columns."""
    profile: dict[str, dict[str, float]] = {}
    for column in columns:
        values = [float(row[column]) for row in rows if row.get(column, "").strip()]
        if not values:
            continue
        profile[column] = {
            "count": float(len(values)),
            "min": min(values),
            "mean": mean(values),
            "max": max(values),
        }
    return profile


def missing_rate(rows: list[dict[str, str]]) -> dict[str, float]:
    """Return per-column missing-value rate for a loaded CSV."""
    if not rows:
        return {}
    total = len(rows)
    rates: dict[str, float] = {}
    for key in rows[0]:
        missing = sum(1 for row in rows if not row.get(key, "").strip())
        rates[key] = missing / total
    return rates


def csv_shape(csv_path: str) -> tuple[int, int]:
    """Return (rows, columns) for a CSV file."""
    rows = load_csv_rows(csv_path)
    if not rows:
        return 0, 0
    return len(rows), len(rows[0])


if __name__ == "__main__":
    print("Tabular EDA utility script ready. Mark this Kaggle script as a utility script in the UI.")
