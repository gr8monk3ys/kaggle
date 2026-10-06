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
from kaggle_portfolio.shared.errors import CommandError


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
        with pytest.raises(CommandError):
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
        "name", ["digest", "sync", "doctor", "leaderboard", "validate", "push"]
    )
    def test_command_is_registered(self, name):
        assert name in [c.name for c in manage_commands.COMMANDS]

    @pytest.mark.parametrize(
        "name", ["scorecard", "badge-plan", "weekly-plan", "pace", "sync-template"]
    )
    def test_removed_planning_command_is_gone(self, name):
        assert name not in [c.name for c in manage_commands.COMMANDS]


class TestKaggleCommands:
    def test_status_survives_an_unreachable_kaggle(self, tmp_path, capsys):
        from kaggle_portfolio.shared.kaggle_client import KaggleError

        _use(tmp_path, fail_with=KaggleError("403 Forbidden"))
        assert manage_commands.cmd_status([]) == 0
        assert "unavailable" in capsys.readouterr().out


class TestDryRunReachesLocalHandlers:
    """--dry-run must gate handler commands, not just delegated modules.

    `Command.run` accepted the effects-gated Deps and dropped it for local
    handlers, which reach for the process-wide deps instead. So --dry-run worked
    for `module` commands and silently did nothing for every handler — including
    `push`. A `push <dataset> --dry-run` published a live dataset because of it.
    """

    def _dataset(self, root: Path) -> Path:
        d = root / "datasets" / "demo"
        d.mkdir(parents=True)
        (d / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "Demo Dataset For Tests",
                    "id": "u/demo",
                    "subtitle": "A minimal fixture used to pin the --dry-run gate",
                    "description": "A minimal fixture used to pin the --dry-run gate.",
                    "licenses": [{"name": "CC0-1.0"}],
                    "keywords": ["tabular"],
                    "resources": [
                        {
                            "path": "data.csv",
                            "description": "Two rows.",
                            "schema": {
                                "fields": [
                                    {
                                        "name": "a",
                                        "title": "A",
                                        "description": "First column.",
                                        "type": "integer",
                                    },
                                    {
                                        "name": "b",
                                        "title": "B",
                                        "description": "Second column.",
                                        "type": "integer",
                                    },
                                ]
                            },
                        }
                    ],
                    "authors": [{"name": "Test", "role": "author"}],
                    "coverage": {
                        "temporal_start_date": "2024-01-01",
                        "temporal_end_date": "2024-12-31",
                        "geospatial_coverage": "Global",
                    },
                    "provenance": {
                        "sources": ["generated"],
                        "collection_methodology": "Synthetic, generated for tests.",
                    },
                }
            ),
            encoding="utf-8",
        )
        (d / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        return d

    def test_push_with_dry_run_records_instead_of_publishing(self, tmp_path: Path):
        self._dataset(tmp_path)
        client = FakeKaggleClient()
        manage_commands.set_deps(Deps.for_test(tmp_path, client=client))

        rc = manage_commands.main(["push", "datasets/demo", "--dry-run"])

        assert rc == 0
        assert client.skipped_effects == ["publish_dataset"], (
            "--dry-run must reach the client and skip the publish; a real run of "
            f"this published a live dataset. skipped={client.skipped_effects}"
        )

    def test_push_without_dry_run_still_publishes(self, tmp_path: Path):
        self._dataset(tmp_path)
        client = FakeKaggleClient()
        manage_commands.set_deps(Deps.for_test(tmp_path, client=client))

        manage_commands.main(["push", "datasets/demo"])

        assert [name for name, _ in client.calls if name == "publish_dataset"], (
            "a normal push must still publish"
        )

    def test_the_global_deps_are_restored_afterwards(self, tmp_path: Path):
        self._dataset(tmp_path)
        original = Deps.for_test(tmp_path, client=FakeKaggleClient())
        manage_commands.set_deps(original)

        manage_commands.main(["push", "datasets/demo", "--dry-run"])

        assert manage_commands.deps() is original, (
            "the effects-gated Deps must not leak past the command that used it"
        )


def _dataset_script(root: Path, name: str, body: str) -> Path:
    d = root / "datasets" / name
    d.mkdir(parents=True)
    (d / "create_dataset.py").write_text(body, encoding="utf-8")
    return d


class TestBuildDatasets:
    def test_runs_each_generator_in_its_own_folder(self, tmp_path: Path):
        writes = "open('data.csv', 'w').write('a,b')\n"
        one = _dataset_script(tmp_path, "one", writes)
        two = _dataset_script(tmp_path, "two", writes)
        _use(tmp_path)
        assert manage_commands.cmd_build_datasets([]) == 0
        assert (one / "data.csv").exists()
        assert (two / "data.csv").exists()

    def test_builds_only_the_named_datasets(self, tmp_path: Path):
        writes = "open('data.csv', 'w').write('a,b')\n"
        one = _dataset_script(tmp_path, "one", writes)
        two = _dataset_script(tmp_path, "two", writes)
        _use(tmp_path)
        assert manage_commands.cmd_build_datasets(["two"]) == 0
        assert not (one / "data.csv").exists()
        assert (two / "data.csv").exists()

    def test_a_failing_generator_fails_the_command(self, tmp_path: Path):
        _dataset_script(tmp_path, "broken", "raise SystemExit(3)\n")
        _use(tmp_path)
        assert manage_commands.cmd_build_datasets([]) == 1

    def test_unknown_names_are_rejected_before_anything_runs(self, tmp_path: Path):
        one = _dataset_script(tmp_path, "one", "open('data.csv', 'w').write('a,b')\n")
        _use(tmp_path)
        with pytest.raises(CommandError, match="nope"):
            manage_commands.cmd_build_datasets(["one", "nope"])
        assert not (one / "data.csv").exists()

    def test_validate_points_at_build_datasets_when_data_is_missing(
        self, tmp_path: Path
    ):
        d = _dataset_script(tmp_path, "one", "")
        meta = d / "dataset-metadata.json"
        payload = {
            "id": "u/one",
            "resources": [{"path": "data.csv", "description": "rows"}],
        }
        raw = json.dumps(payload)
        meta.write_text(raw, encoding="utf-8")
        _use(tmp_path)
        errors = manage_commands.validate_dataset(meta, payload, raw)
        assert any("build-datasets" in e for e in errors)
