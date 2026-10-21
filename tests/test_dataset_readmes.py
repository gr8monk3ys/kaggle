"""Published dataset READMEs must agree with the data they describe."""

import csv
import re
from pathlib import Path

import pytest

README_ROWS_RE = re.compile(
    r"^## (?P<file>\S+\.csv)\s*$\n\s*$\n\*\*Rows:\*\* (?P<rows>[\d,]+)",
    re.MULTILINE,
)


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV, honouring quoted fields with embedded newlines."""
    previous_limit = csv.field_size_limit(2**31 - 1)
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
    if not any((repo_root / "datasets").glob("*/*.csv")):
        pytest.skip("dataset CSVs are not built; run ./manage.sh build-datasets")
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
