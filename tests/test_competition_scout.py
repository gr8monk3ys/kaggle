from __future__ import annotations

from datetime import UTC

from kaggle_portfolio.notebooks import competition_scout
from kaggle_portfolio.shared import kaggle_utils


def test_kaggle_command_falls_back_to_module_cli(monkeypatch):
    monkeypatch.setattr(kaggle_utils, "kaggle_cli_path", lambda: None)
    monkeypatch.setattr(
        kaggle_utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "kaggle.cli" else None,
    )
    cmd = kaggle_utils.kaggle_command()
    assert cmd[1:] == ["-m", "kaggle.cli"]


def test_parse_csv_handles_standard_output():
    raw = "ref,title,teamCount,deadline\ncomp/sample,Sample,42,2026-03-01T00:00:00Z\n"
    rows = competition_scout._parse_csv(raw)
    assert len(rows) == 1
    assert rows[0]["ref"] == "comp/sample"
    assert rows[0]["teamCount"] == "42"


def test_parse_deadline_datetime_normalizes_naive_values_to_utc():
    parsed = competition_scout.parse_deadline_datetime("2026-03-01T00:00:00")
    assert parsed.tzinfo == UTC
