import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

from kaggle_portfolio.ops import stale_content_detector as scd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_notebook(path: Path, cells: list[dict] | None = None) -> None:
    """Write a minimal .ipynb file."""
    if cells is None:
        cells = [
            {"cell_type": "markdown", "metadata": {}, "source": "# Hello"},
            {"cell_type": "code", "metadata": {}, "source": "import numpy as np",
             "outputs": [], "execution_count": None},
        ]
    payload = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_kernel_metadata(dir_path: Path, code_file: str = "guide.ipynb") -> None:
    """Write a minimal kernel-metadata.json."""
    meta = {
        "id": "user/test-notebook",
        "title": "Test Notebook",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
    }
    (dir_path / "kernel-metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _write_dataset_metadata(dir_path: Path) -> None:
    """Write a minimal dataset-metadata.json."""
    meta = {
        "id": "user/test-dataset",
        "title": "Test Dataset Title Here",
    }
    (dir_path / "dataset-metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _set_mtime_days_ago(path: Path, days: int) -> None:
    """Set a file's mtime to N days ago from 2026-03-02 (the tests' frozen today)."""
    # Anchor to the same fixed date the tests pass as `today`, at local noon so
    # DST shifts cannot move the resulting calendar date.
    ref = time.mktime(date(2026, 3, 2).timetuple()) + 43200
    target_ts = ref - (days * 86400)
    os.utime(path, (target_ts, target_ts))


# ---------------------------------------------------------------------------
# Stale notebook detection
# ---------------------------------------------------------------------------


def test_stale_notebook_flagged(tmp_path):
    """A notebook older than the threshold should be flagged."""
    nb_dir = tmp_path / "old-notebook"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    nb_file = nb_dir / "guide.ipynb"
    _write_notebook(nb_file)
    _set_mtime_days_ago(nb_file, 90)

    today = date(2026, 3, 2)
    stale = scd.find_stale_notebooks(tmp_path, today, max_age_days=60)

    assert len(stale) == 1
    assert stale[0]["rel_dir"] == "old-notebook"
    assert stale[0]["days_stale"] >= 60


def test_fresh_notebook_not_flagged(tmp_path):
    """A recently modified notebook should NOT be flagged."""
    nb_dir = tmp_path / "fresh-notebook"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    nb_file = nb_dir / "guide.ipynb"
    _write_notebook(nb_file)
    # Just created — mtime is now, well within 60 days

    today = date.today()
    stale = scd.find_stale_notebooks(tmp_path, today, max_age_days=60)

    assert len(stale) == 0


def test_custom_max_age_threshold(tmp_path):
    """Custom --max-nb-age should change the threshold."""
    nb_dir = tmp_path / "medium-notebook"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    nb_file = nb_dir / "guide.ipynb"
    _write_notebook(nb_file)
    _set_mtime_days_ago(nb_file, 40)

    today = date(2026, 3, 2)

    # With default 60, should not be stale
    assert len(scd.find_stale_notebooks(tmp_path, today, max_age_days=60)) == 0

    # With custom 30, should be stale
    stale = scd.find_stale_notebooks(tmp_path, today, max_age_days=30)
    assert len(stale) == 1


# ---------------------------------------------------------------------------
# Stale dataset detection
# ---------------------------------------------------------------------------


def test_stale_dataset_flagged(tmp_path):
    """A dataset with old CSV files should be flagged."""
    ds_dir = tmp_path / "datasets" / "old-data"
    ds_dir.mkdir(parents=True)
    _write_dataset_metadata(ds_dir)
    csv_file = ds_dir / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    _set_mtime_days_ago(csv_file, 120)

    today = date(2026, 3, 2)
    stale = scd.find_stale_datasets(tmp_path, today, max_age_days=90)

    assert len(stale) == 1
    assert "old-data" in stale[0]["rel_dir"]
    assert stale[0]["days_stale"] >= 90


def test_fresh_dataset_not_flagged(tmp_path):
    """A recently updated dataset should NOT be flagged."""
    ds_dir = tmp_path / "datasets" / "fresh-data"
    ds_dir.mkdir(parents=True)
    _write_dataset_metadata(ds_dir)
    csv_file = ds_dir / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    today = date.today()
    stale = scd.find_stale_datasets(tmp_path, today, max_age_days=90)

    assert len(stale) == 0


def test_stale_dataset_uses_newest_data_file(tmp_path):
    """If one CSV is old but another is fresh, use the newest mtime."""
    ds_dir = tmp_path / "datasets" / "mixed-data"
    ds_dir.mkdir(parents=True)
    _write_dataset_metadata(ds_dir)

    old_csv = ds_dir / "old.csv"
    old_csv.write_text("x\n1\n")
    _set_mtime_days_ago(old_csv, 120)

    new_csv = ds_dir / "new.csv"
    new_csv.write_text("y\n2\n")
    # new_csv was just created — fresh

    today = date.today()
    stale = scd.find_stale_datasets(tmp_path, today, max_age_days=90)

    assert len(stale) == 0


def test_parquet_files_detected(tmp_path):
    """Parquet data files should also be checked."""
    ds_dir = tmp_path / "datasets" / "parquet-data"
    ds_dir.mkdir(parents=True)
    _write_dataset_metadata(ds_dir)
    pq_file = ds_dir / "data.parquet"
    pq_file.write_bytes(b"\x00")  # minimal placeholder
    _set_mtime_days_ago(pq_file, 100)

    today = date(2026, 3, 2)
    stale = scd.find_stale_datasets(tmp_path, today, max_age_days=90)

    assert len(stale) == 1


# ---------------------------------------------------------------------------
# Outdated library version detection
# ---------------------------------------------------------------------------


def test_outdated_version_detected(tmp_path):
    """A notebook with a very old torch version should be flagged."""
    nb_dir = tmp_path / "old-torch"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    cells = [
        {
            "cell_type": "code",
            "metadata": {},
            "source": "!pip install torch==0.4.1",
            "outputs": [],
            "execution_count": None,
        },
    ]
    _write_notebook(nb_dir / "guide.ipynb", cells)

    outdated = scd.find_outdated_libraries(tmp_path)

    assert len(outdated) == 1
    assert outdated[0]["library"] == "torch"
    assert outdated[0]["pinned_version"] == "0.4.1"


def test_recent_version_not_flagged(tmp_path):
    """A notebook with a recent torch version should NOT be flagged."""
    nb_dir = tmp_path / "new-torch"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    cells = [
        {
            "cell_type": "code",
            "metadata": {},
            "source": "!pip install torch==2.5.0",
            "outputs": [],
            "execution_count": None,
        },
    ]
    _write_notebook(nb_dir / "guide.ipynb", cells)

    outdated = scd.find_outdated_libraries(tmp_path)

    assert len(outdated) == 0


def test_source_as_list_handled(tmp_path):
    """Notebook cells with source as a list of strings should work."""
    nb_dir = tmp_path / "list-source"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    cells = [
        {
            "cell_type": "code",
            "metadata": {},
            "source": ["!pip install tensorflow==0.12.0\n", "import tensorflow"],
            "outputs": [],
            "execution_count": None,
        },
    ]
    _write_notebook(nb_dir / "guide.ipynb", cells)

    outdated = scd.find_outdated_libraries(tmp_path)

    assert len(outdated) == 1
    assert outdated[0]["library"] == "tensorflow"


def test_extract_pinned_versions_from_cell():
    """Unit test for the version extraction regex."""
    source = "!pip install torch==2.0.1 transformers==4.20.0 numpy==1.24.0"
    results = scd.extract_pinned_versions_from_cell(source)

    pkgs = {pkg for pkg, _ in results}
    assert "torch" in pkgs
    assert "transformers" in pkgs
    assert "numpy" in pkgs


def test_parse_version_tuple():
    assert scd.parse_version_tuple("2.0.1") == (2, 0, 1)
    assert scd.parse_version_tuple("1.24") == (1, 24)
    assert scd.parse_version_tuple("bad") is None


def test_is_outdated():
    assert scd.is_outdated((0, 4), (2, 5)) is True       # 2 major behind
    assert scd.is_outdated((1, 0), (2, 5)) is False      # only 1 major behind
    assert scd.is_outdated((2, 0), (2, 5)) is False      # same major


# ---------------------------------------------------------------------------
# CLI main() with --today
# ---------------------------------------------------------------------------


def test_main_with_today_flag(tmp_path, capsys):
    """CLI main should accept --today and produce output."""
    nb_dir = tmp_path / "sample-nb"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    nb_file = nb_dir / "guide.ipynb"
    _write_notebook(nb_file)
    _set_mtime_days_ago(nb_file, 100)

    ret = scd.main([
        "--root", str(tmp_path),
        "--today", "2026-03-02",
        "--max-nb-age", "60",
    ])

    assert ret == 0
    captured = capsys.readouterr()
    assert "Stale Content" in captured.out
    assert "sample-nb" in captured.out


def test_main_json_output(tmp_path, capsys):
    """--json flag should produce valid JSON output."""
    nb_dir = tmp_path / "json-nb"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir)
    nb_file = nb_dir / "guide.ipynb"
    _write_notebook(nb_file)
    _set_mtime_days_ago(nb_file, 70)

    ret = scd.main([
        "--root", str(tmp_path),
        "--today", "2026-03-02",
        "--max-nb-age", "60",
        "--json",
    ])

    assert ret == 0
    captured = capsys.readouterr()
    # The JSON output is mixed with the coloured header; parse only the JSON part
    # Find the first '{' to last '}'
    out = captured.out
    json_start = out.index("{")
    json_end = out.rindex("}") + 1
    data = json.loads(out[json_start:json_end])
    assert data["summary"]["stale_notebooks"] >= 1


# ---------------------------------------------------------------------------
# Report generation format
# ---------------------------------------------------------------------------


def test_markdown_report_format():
    """The markdown report should contain expected sections and tables."""
    stale_nbs = [
        {"rel_dir": "old-nb", "nb_path": "/tmp/old-nb/guide.ipynb",
         "last_modified": "2025-11-01", "days_stale": 122},
    ]
    stale_ds = [
        {"rel_dir": "datasets/old-ds", "oldest_file": "/tmp/data.csv",
         "last_modified": "2025-10-15", "days_stale": 139},
    ]
    outdated = [
        {"rel_dir": "old-nb", "nb_path": "/tmp/old-nb/guide.ipynb",
         "library": "torch", "pinned_version": "0.4.1", "recent_version": "2.5"},
    ]

    report = scd.build_markdown_report(
        stale_nbs, stale_ds, outdated,
        today=date(2026, 3, 2), max_nb_age=60, max_ds_age=90,
    )

    assert "# Stale Content Report" in report
    assert "## Stale Notebooks" in report
    assert "old-nb" in report
    assert "122" in report
    assert "## Stale Datasets" in report
    assert "datasets/old-ds" in report
    assert "## Outdated Library Versions" in report
    assert "torch" in report
    assert "0.4.1" in report
    assert "## Summary" in report
    assert "Total stale items:** 3" in report
    assert "### Most Urgent" in report


def test_markdown_report_empty():
    """Empty report should say no stale items found."""
    report = scd.build_markdown_report(
        [], [], [],
        today=date(2026, 3, 2), max_nb_age=60, max_ds_age=90,
    )

    assert "No stale notebooks found" in report
    assert "No stale datasets found" in report
    assert "No outdated pinned library versions found" in report
    assert "Total stale items:** 0" in report


def test_json_report_structure():
    """JSON report should have expected keys."""
    report = scd.build_json_report(
        [{"rel_dir": "x", "nb_path": "/x", "last_modified": "2025-01-01", "days_stale": 100}],
        [], [],
        today=date(2026, 3, 2), max_nb_age=60, max_ds_age=90,
    )

    assert report["generated"] == "2026-03-02"
    assert report["thresholds"]["notebook_days"] == 60
    assert report["summary"]["stale_notebooks"] == 1
    assert report["summary"]["total"] == 1


# ---------------------------------------------------------------------------
# discover_notebooks / discover_datasets
# ---------------------------------------------------------------------------


def test_discover_notebooks(tmp_path):
    nb_dir = tmp_path / "my-nb"
    nb_dir.mkdir()
    _write_kernel_metadata(nb_dir, "notebook.ipynb")
    _write_notebook(nb_dir / "notebook.ipynb")

    results = scd.discover_notebooks(tmp_path)

    assert len(results) == 1
    assert results[0]["rel_dir"] == "my-nb"


def test_discover_datasets(tmp_path):
    ds_dir = tmp_path / "datasets" / "sample"
    ds_dir.mkdir(parents=True)
    _write_dataset_metadata(ds_dir)

    results = scd.discover_datasets(tmp_path)

    assert len(results) == 1
    assert "sample" in results[0]["rel_dir"]
