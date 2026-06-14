from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_portfolio import manage_commands


def test_resolve_target_returns_direct_path_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    direct = tmp_path / "feature-engineering"
    direct.mkdir()

    monkeypatch.setattr(manage_commands, "ROOT", tmp_path)
    monkeypatch.setattr(manage_commands, "NOTEBOOK_DIRS", ["projects/tutorials/feature-engineering"])
    monkeypatch.setattr(manage_commands, "DATASET_DIRS", [])

    assert manage_commands.resolve_target("feature-engineering") == direct.resolve()


def test_resolve_target_falls_back_to_unique_basename_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    nested = tmp_path / "projects" / "tutorials" / "feature-engineering"
    nested.mkdir(parents=True)

    monkeypatch.setattr(manage_commands, "ROOT", tmp_path)
    monkeypatch.setattr(manage_commands, "NOTEBOOK_DIRS", ["projects/tutorials/feature-engineering"])
    monkeypatch.setattr(manage_commands, "DATASET_DIRS", [])

    assert manage_commands.resolve_target("feature-engineering") == nested.resolve()


def test_resolve_target_raises_for_ambiguous_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    a = tmp_path / "projects" / "tutorials" / "feature-engineering"
    b = tmp_path / "projects" / "legacy" / "feature-engineering"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    monkeypatch.setattr(manage_commands, "ROOT", tmp_path)
    monkeypatch.setattr(
        manage_commands,
        "NOTEBOOK_DIRS",
        ["projects/tutorials/feature-engineering", "projects/legacy/feature-engineering"],
    )
    monkeypatch.setattr(manage_commands, "DATASET_DIRS", [])

    with pytest.raises(SystemExit, match="Ambiguous target 'feature-engineering'"):
        manage_commands.resolve_target("feature-engineering")


def test_has_kaggle_credentials_accepts_api_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token-123")

    ok, sources = manage_commands.has_kaggle_credentials()

    assert ok is True
    assert "environment-token" in sources


def test_digest_command_is_registered():
    names = [c.name for c in manage_commands.COMMANDS]
    assert "digest" in names
