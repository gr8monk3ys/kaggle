import csv
import re
import sys
from pathlib import Path

from kaggle_portfolio.datasets import dataset_optimizer
from kaggle_portfolio.shared import proc
from kaggle_portfolio.shared.kaggle_client import FakeKaggleClient


def test_optimize_dataset_fails_on_invalid_metadata_json(tmp_path):
    ds_dir = tmp_path / "broken-meta"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset-metadata.json").write_text("{bad json", encoding="utf-8")

    ok = dataset_optimizer.optimize_dataset(FakeKaggleClient(), ds_dir, push=False)

    assert ok is False


def test_summarize_output_prefers_last_meaningful_line():
    message = proc.summarize_output(
        "warning one\nwarning two\n",
        "401 Client Error: Unauthorized for url: https://www.kaggle.com/api/v1/blobs/upload\n"
        "  warnings.warn(\n",
    )
    assert message.startswith("401 Client Error: Unauthorized")


def test_analyze_csv_error_contains_file_name(tmp_path):
    bad_csv = tmp_path / "broken.csv"
    bad_csv.write_text("", encoding="utf-8")

    analysis = dataset_optimizer.analyze_csv(bad_csv)

    assert analysis["file"] == "broken.csv"
    assert "error" in analysis


def test_generate_readme_handles_error_entries_without_file_key(tmp_path):
    content = dataset_optimizer.generate_readme(
        ds_dir=tmp_path / "dataset",
        meta={"title": "Sample Dataset"},
        file_analyses=[{"error": "boom", "columns": [], "rows": 0, "size_kb": 0.0}],
    )

    assert "unknown-file" in content
    assert "Error reading file: boom" in content


def test_optimize_dataset_includes_parquet_analysis(tmp_path, monkeypatch):
    ds_dir = tmp_path / "sample-dataset"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset-metadata.json").write_text(
        '{"title":"Sample","description":"d","licenses":[{"name":"CC0"}]}',
        encoding="utf-8",
    )
    (ds_dir / "events.parquet").write_bytes(b"PAR1")

    calls: list[str] = []

    def fake_analyze_parquet(path: Path, max_rows: int = 5000) -> dict:
        calls.append(path.name)
        return {
            "file": path.name,
            "rows": 10,
            "size_kb": 1.0,
            "columns": [
                {
                    "name": "event_id",
                    "dtype": "integer",
                    "null_pct": 0.0,
                    "n_unique": 10,
                    "samples": ["1", "2", "3"],
                    "total": 10,
                }
            ],
        }

    monkeypatch.setattr(dataset_optimizer, "analyze_parquet", fake_analyze_parquet)

    ok = dataset_optimizer.optimize_dataset(FakeKaggleClient(), ds_dir, push=False)
    readme = (ds_dir / "README.md").read_text(encoding="utf-8")

    assert ok
    assert calls == ["events.parquet"]
    assert "events.parquet" in readme


def test_apply_metadata_defaults_sets_update_frequency_when_missing():
    meta = {"id": "owner/sample", "title": "Sample", "description": "desc"}

    normalized, changed = dataset_optimizer.apply_metadata_defaults(meta)

    assert changed is True
    assert normalized["updateFrequency"] == "Monthly"


# ── Row-count guardrails ──────────────────────────────────────────────────────
#
# A README stated "Rows: 5,000" for a 50,000-row dataset because analyze_csv
# read a 5,000-row sample and reported it as the total. These figures are
# published on Kaggle, so they are checked against the real files.

README_ROWS_RE = re.compile(
    r"^## (?P<file>\S+\.csv)\s*$\n\s*$\n\*\*Rows:\*\* (?P<rows>[\d,]+)",
    re.MULTILINE,
)


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV, honouring quoted fields with embedded newlines."""
    previous_limit = csv.field_size_limit(sys.maxsize)
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                return 0
            return sum(1 for _ in reader)
    finally:
        csv.field_size_limit(previous_limit)


def readme_row_claims(readme: Path) -> list[tuple[str, int]]:
    """Extract the (csv file name, stated row count) pairs from a README."""
    text = readme.read_text(encoding="utf-8")
    return [
        (match.group("file"), int(match.group("rows").replace(",", "")))
        for match in README_ROWS_RE.finditer(text)
    ]


def test_generated_readmes_state_the_real_csv_row_count(repo_root):
    mismatches = []
    checked = 0
    for readme in sorted((repo_root / "datasets").glob("*/README.md")):
        for file_name, claimed in readme_row_claims(readme):
            csv_path = readme.parent / file_name
            if not csv_path.exists():
                continue
            actual = count_csv_rows(csv_path)
            checked += 1
            if actual != claimed:
                mismatches.append(
                    f"{readme.parent.name}/{file_name}: README says "
                    f"{claimed:,} rows, file has {actual:,}"
                )

    assert checked, "no dataset README stated a row count — the regex went stale"
    assert not mismatches, "README row counts disagree with the CSVs:\n" + "\n".join(
        mismatches
    )


def test_analyze_csv_counts_every_row_not_a_sample(tmp_path):
    """The row count must describe the whole file, however it is sampled."""
    rows = dataset_optimizer.SAMPLE_SCAN_ROWS * 3 + 7
    csv_path = tmp_path / "wide.csv"
    lines = ["id,bucket"]
    lines += [f"{i},{i % 4}" for i in range(rows)]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    analysis = dataset_optimizer.analyze_csv(csv_path)

    assert analysis["rows"] == rows
    by_name = {col["name"]: col for col in analysis["columns"]}
    assert by_name["id"]["n_unique"] == rows
    assert by_name["id"]["total"] == rows
    assert by_name["bucket"]["n_unique"] == 4

    readme = dataset_optimizer.generate_readme(
        ds_dir=tmp_path, meta={"title": "Wide"}, file_analyses=[analysis]
    )
    assert f"**Rows:** {rows:,}" in readme


def test_analyze_csv_counts_nulls_over_the_whole_file(tmp_path):
    rows = dataset_optimizer.SAMPLE_SCAN_ROWS * 2
    csv_path = tmp_path / "sparse.csv"
    # The first half is fully populated, so a sampled null% would read 0.0.
    lines = ["id,note"]
    lines += [f"{i},filled" for i in range(rows // 2)]
    lines += [f"{i}," for i in range(rows // 2, rows)]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    analysis = dataset_optimizer.analyze_csv(csv_path)

    note = next(col for col in analysis["columns"] if col["name"] == "note")
    assert analysis["rows"] == rows
    assert note["null_pct"] == 50.0


def test_analyze_csv_marks_distinct_counts_that_hit_the_cap(tmp_path):
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text(
        "id\n" + "\n".join(str(i) for i in range(50)) + "\n", encoding="utf-8"
    )

    analysis = dataset_optimizer.analyze_csv(csv_path, distinct_cap=10)

    col = analysis["columns"][0]
    assert col["n_unique"] == 10
    assert col["n_unique_capped"] is True
    readme = dataset_optimizer.generate_readme(
        ds_dir=tmp_path, meta={"title": "Ids"}, file_analyses=[analysis]
    )
    assert "10+" in readme
