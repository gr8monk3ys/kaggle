from __future__ import annotations

from pathlib import Path

from kaggle_portfolio.shared import kaggle_utils


def test_kaggle_command_discovers_sibling_binary(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_path = bin_dir / "python"
    python_path.write_text("", encoding="utf-8")
    kaggle_path = bin_dir / "kaggle"
    kaggle_path.write_text("#!/bin/sh\n", encoding="utf-8")
    kaggle_path.chmod(0o755)

    monkeypatch.setattr(kaggle_utils.shutil, "which", lambda _: None)
    monkeypatch.setattr(kaggle_utils.sys, "executable", str(python_path))
    monkeypatch.setattr(kaggle_utils.importlib.util, "find_spec", lambda _: None)

    assert kaggle_utils.kaggle_command() == [str(kaggle_path)]


def test_kaggle_command_falls_back_to_legacy_module(monkeypatch):
    monkeypatch.setattr(kaggle_utils, "kaggle_cli_path", lambda: None)
    monkeypatch.setattr(
        kaggle_utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "kaggle.cli" else None,
    )

    cmd = kaggle_utils.kaggle_command()

    assert cmd[1:] == ["-m", "kaggle.cli"]
