"""The competition-lab harness: data in, submission out.

Everything here is competition-agnostic. Each competition's model lives in its
own module beside this one — before the split they shared a single 4,500-line
namespace and a ~60-function private-helper prefix convention was doing the work
a module boundary should.
"""

from __future__ import annotations

#!/usr/bin/env python3


import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import KaggleError
from kaggle_portfolio.shared.layout import RepoLayout
from kaggle_portfolio.shared.errors import CommandError

# One lab root, set at the CLI edge by set_layout(). A module-level value rather
# than a parameter because benchmarks are called as
# BENCHMARKS[slug](data_dir, folds, write_submission) — threading a layout
# through them would change that documented interface. It was previously worse:
# _ensure_data read deps.layout while _submission_dir read a separately-resolved
# global, so data and submissions could land under different roots.
_LAYOUT: RepoLayout | None = None


def layout() -> RepoLayout:
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = RepoLayout.resolve()
    return _LAYOUT


def set_layout(new: RepoLayout | None) -> None:
    """Point the lab at a different repo root. Called from main(); used by tests."""
    global _LAYOUT
    _LAYOUT = new


RANDOM_STATE = 42
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


@dataclass(frozen=True)
class LabResult:
    competition: str
    metric_name: str
    best_model: str
    best_score: float
    benchmark_rows: list[dict[str, Any]]
    submission_path: Path | None = None


def _ensure_data(deps: Deps, slug: str, force_download: bool = False) -> Path:
    set_layout(deps.layout)
    data_dir = layout().lab_root / slug / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files and not force_download:
        return data_dir

    try:
        deps.client.download_competition(slug, data_dir)
    except KaggleError as exc:
        raise CommandError(f"Failed to download {slug}: {exc}") from exc

    zip_files = list(data_dir.glob("*.zip"))
    for archive in zip_files:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    return data_dir


def _submission_dir(slug: str) -> Path:
    out = layout().lab_root / slug / "submissions"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _benchmark_dir(slug: str) -> Path:
    out = layout().lab_root / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")


def _save_summary(result: LabResult) -> None:
    out_dir = _benchmark_dir(result.competition)
    payload = {
        "competition": result.competition,
        "metric_name": result.metric_name,
        "best_model": result.best_model,
        "best_score": round(result.best_score, 6),
        "benchmarks": result.benchmark_rows,
        "submission_path": str(result.submission_path)
        if result.submission_path
        else None,
    }
    (out_dir / "latest_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_benchmarks(result: LabResult) -> None:
    print(f"{BLUE}=== {result.competition} local benchmark ==={RESET}")
    print(f"{'Model':<20} {result.metric_name:>10}")
    print("-" * 33)
    for row in result.benchmark_rows:
        color = GREEN if row["model"] == result.best_model else RESET
        print(f"{color}{row['model']:<20}{RESET} {row['score']:>10.5f}")
    print("")
    print(
        f"Best: {GREEN}{result.best_model}{RESET} "
        f"({result.metric_name}={result.best_score:.5f})"
    )
    if result.submission_path:
        print(f"Submission file: {result.submission_path}")


def _submit(deps: Deps, slug: str, submission_path: Path, message: str) -> None:
    outcome = deps.client.submit(slug, submission_path, message)
    if not outcome.ok:
        raise CommandError(f"Submission failed: {outcome.detail}")
    print(outcome.detail.strip() or "Submission accepted.")


def _concat_feature_block(df: pd.DataFrame, updates: dict[str, Any]) -> pd.DataFrame:
    if not updates:
        return df
    block = pd.DataFrame(updates, index=df.index)
    return pd.concat([df, block], axis=1).copy()
