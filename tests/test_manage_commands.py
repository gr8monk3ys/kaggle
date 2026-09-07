from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_portfolio import manage_commands
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import (
    CredentialState,
    Credentials,
    FakeKaggleClient,
)


@pytest.fixture(autouse=True)
def _reset_deps():
    """Every test gets its own dependencies; none leak into the next."""
    yield
    manage_commands.set_deps(None)


def _notebook(root: Path, rel: str) -> Path:
    d = root / rel
    d.mkdir(parents=True)
    (d / "kernel-metadata.json").write_text(
        json.dumps({"id": f"u/{d.name}"}), encoding="utf-8"
    )
    return d


def _use(root: Path, **kwargs) -> Deps:
    """Point manage_commands at a temporary repo. One call, not three patched globals."""
    deps = Deps.for_test(root, client=FakeKaggleClient(**kwargs))
    manage_commands.set_deps(deps)
    return deps


class TestResolveTarget:
    def test_returns_direct_path_when_present(self, tmp_path: Path):
        direct = _notebook(tmp_path, "feature-engineering")
        _notebook(tmp_path, "projects/tutorials/feature-engineering")
        _use(tmp_path)
        assert manage_commands.resolve_target("feature-engineering") == direct.resolve()

    def test_falls_back_to_unique_basename_match(self, tmp_path: Path):
        nested = _notebook(tmp_path, "projects/tutorials/feature-engineering")
        _use(tmp_path)
        assert manage_commands.resolve_target("feature-engineering") == nested.resolve()

    def test_raises_for_ambiguous_basename(self, tmp_path: Path):
        _notebook(tmp_path, "projects/tutorials/feature-engineering")
        _notebook(tmp_path, "projects/legacy/feature-engineering")
        _use(tmp_path)
        with pytest.raises(SystemExit):
            manage_commands.resolve_target("feature-engineering")


class TestDiscovery:
    def test_discovery_is_lazy_and_reflects_the_current_repo(self, tmp_path: Path):
        _use(tmp_path)
        assert manage_commands.discover_notebook_dirs() == []
        # A directory created after the first scan is visible to a fresh Deps:
        # discovery caches per instance, not per process.
        _notebook(tmp_path, "late-arrival")
        _use(tmp_path)
        assert manage_commands.discover_notebook_dirs() == ["late-arrival"]


class TestCredentials:
    def test_reports_the_clients_answer(self, tmp_path: Path):
        _use(
            tmp_path,
            credentials_state=CredentialState(
                Credentials("u", "k", "environment-token"), ["environment-token"]
            ),
        )
        ok, sources = manage_commands.has_kaggle_credentials()
        assert ok is True
        assert "environment-token" in sources

    def test_reports_absence(self, tmp_path: Path):
        _use(tmp_path, credentials_state=CredentialState(None, [], "nope"))
        ok, sources = manage_commands.has_kaggle_credentials()
        assert ok is False
        assert sources == []


class TestCommandRegistry:
    @pytest.mark.parametrize(
        "name", ["digest", "leaderboard", "validate", "push", "scorecard"]
    )
    def test_command_is_registered(self, name):
        assert name in [c.name for c in manage_commands.COMMANDS]


class TestKaggleCommands:
    def test_status_survives_an_unreachable_kaggle(self, tmp_path, capsys):
        from kaggle_portfolio.shared.kaggle_client import KaggleError

        _use(tmp_path, fail_with=KaggleError("403 Forbidden"))
        assert manage_commands.cmd_status([]) == 0
        assert "unavailable" in capsys.readouterr().out

    def test_competitions_now_reports_failure_instead_of_always_succeeding(
        self, tmp_path, capsys
    ):
        from kaggle_portfolio.shared.kaggle_client import KaggleError

        _use(tmp_path, fail_with=KaggleError("403 Forbidden"))
        # This command previously discarded the return code and always returned 0.
        assert manage_commands.cmd_competitions([]) == 1
        assert "unavailable" in capsys.readouterr().out
