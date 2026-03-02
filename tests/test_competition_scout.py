from __future__ import annotations

import kaggle_utils
import competition_scout


def test_kaggle_command_falls_back_to_module_cli(monkeypatch):
    monkeypatch.setattr(kaggle_utils.shutil, "which", lambda _: None)
    cmd = kaggle_utils.kaggle_command()
    assert cmd[1:] == ["-m", "kaggle.cli"]


def test_parse_csv_handles_standard_output():
    raw = "ref,title,teamCount,deadline\ncomp/sample,Sample,42,2026-03-01T00:00:00Z\n"
    rows = competition_scout._parse_csv(raw)
    assert len(rows) == 1
    assert rows[0]["ref"] == "comp/sample"
    assert rows[0]["teamCount"] == "42"
