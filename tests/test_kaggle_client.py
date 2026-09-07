"""Tests for the Kaggle seam.

These replace the per-module tests that used to monkeypatch ``kaggle_command``
and re-implement CSV parsing: that behaviour now lives in exactly one place, so
it is tested in exactly one place.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kaggle_portfolio.shared import kaggle_client as kc
from kaggle_portfolio.shared.kaggle_client import (
    CliKaggleClient,
    Competition,
    Dataset,
    FakeKaggleClient,
    KaggleClient,
    KaggleCommandFailed,
    KaggleFieldMissing,
    Kernel,
    parse_csv,
    strip_noise,
)


class TestOutputParsing:
    def test_strips_the_outdated_api_banner(self):
        raw = "Warning: Looks like you're using an outdated API Version\nref,title\na/b,Hi\n"
        assert parse_csv(raw) == [{"ref": "a/b", "title": "Hi"}]

    def test_strips_the_leaderboard_page_token(self):
        raw = "Next Page Token = ABC\nteamName,score\nalpha,0.9\n"
        assert parse_csv(raw) == [{"teamName": "alpha", "score": "0.9"}]

    def test_strips_absolute_path_noise_lines(self):
        assert strip_noise(
            "/usr/lib/python/site-packages\nref,title\na/b,Hi\n"
        ).startswith("ref,")

    def test_empty_output(self):
        assert parse_csv("") == []
        assert parse_csv("Warning: only noise\n") == []


class TestFieldResolution:
    """The banner-as-header bug is what these make loud."""

    def test_vote_column_spellings_all_resolve(self):
        for column in ("totalVotes", "voteCount", "votes", "vote"):
            row = {"ref": "me/a", "title": "A", column: "7"}
            assert Kernel.from_row(row).total_votes == 7

    def test_missing_required_column_raises_rather_than_yielding_none(self):
        # This is the shape a Warning-banner-as-header produces.
        with pytest.raises(KaggleFieldMissing) as excinfo:
            Kernel.from_row(
                {
                    "Warning: Looks like you're using an outdated API Version": "ref,title"
                }
            )
        assert "kernel ref" in str(excinfo.value)

    def test_competition_deadline_falls_back_to_evaluation_date(self):
        row = {"ref": "c/x", "title": "X", "evaluationDate": "2026-03-01T00:00:00Z"}
        assert Competition.from_row(row).deadline == "2026-03-01T00:00:00Z"

    def test_entered_flag_spellings(self):
        for column in ("userHasEntered", "hasEntered", "entered"):
            assert (
                Competition.from_row({"ref": "c/x", column: "True"}).user_has_entered
                is True
            )

    def test_slug_strips_owner_and_url(self):
        assert Dataset.from_row({"ref": "owner/name"}).slug == "name"
        assert (
            Competition.from_row(
                {"ref": "https://www.kaggle.com/competitions/mm-2026"}
            ).slug
            == "mm-2026"
        )

    def test_private_notebook_placeholder_is_detectable(self):
        assert Kernel.from_row(
            {"ref": "me/a", "title": "[private notebook]", "votes": "0"}
        ).is_private_placeholder


class TestBothAdaptersSatisfyTheInterface:
    @pytest.mark.parametrize("adapter", [CliKaggleClient, FakeKaggleClient])
    def test_every_interface_method_is_implemented(self, adapter):
        missing = [
            name
            for name in dir(KaggleClient)
            if not name.startswith("_") and not hasattr(adapter, name)
        ]
        assert missing == []


class TestPageSizeCapability:
    """One policy, probed once, replacing three incompatible ones."""

    def _client_rejecting_page_size(self, monkeypatch, rows_by_call):
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            if "--page-size" in argv:
                return subprocess.CompletedProcess(
                    argv, 2, "", "error: unrecognized arguments: --page-size"
                )
            return subprocess.CompletedProcess(argv, 0, rows_by_call, "")

        monkeypatch.setattr(kc.subprocess, "run", fake_run)
        client = CliKaggleClient()
        monkeypatch.setattr(client, "_prefix", lambda: ["kaggle"])
        return client, calls

    def test_falls_back_once_then_stops_sending_the_flag(self, monkeypatch):
        rows = "ref,title,votes\n" + "".join(f"me/{i},T{i},1\n" for i in range(3))
        client, calls = self._client_rejecting_page_size(monkeypatch, rows)
        assert len(client.my_kernels()) == 3
        assert client._page_size_ok is False
        # Second call must not retry the rejected flag.
        before = len(calls)
        client.my_kernels()
        assert all("--page-size" not in argv for argv in calls[before:])

    def test_a_real_failure_still_raises(self, monkeypatch):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, "", "403 Forbidden")

        monkeypatch.setattr(kc.subprocess, "run", fake_run)
        client = CliKaggleClient(retries=1)
        monkeypatch.setattr(client, "_prefix", lambda: ["kaggle"])
        with pytest.raises(KaggleCommandFailed) as excinfo:
            client.my_kernels()
        assert "403 Forbidden" in str(excinfo.value)


class TestEffectsGate:
    """--dry-run is enforced here so no caller can forget it."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.push_kernel(Path("d")),
            lambda c: c.publish_dataset(Path("d"), "msg"),
            lambda c: c.submit("comp", Path("f.csv"), "msg"),
        ],
    )
    def test_mutating_calls_are_recorded_not_performed(self, monkeypatch, call):
        def explode(*a, **k):
            raise AssertionError("a subprocess ran with effects disabled")

        monkeypatch.setattr(kc.subprocess, "run", explode)
        client = CliKaggleClient(effects=False)
        outcome = call(client)
        assert outcome.ok and outcome.skipped
        assert len(client.skipped_effects) == 1

    def test_download_is_gated_too(self, monkeypatch):
        monkeypatch.setattr(
            kc.subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")),
        )
        client = CliKaggleClient(effects=False)
        assert client.download_competition("titanic", Path("/tmp/x")) == Path("/tmp/x")
        assert client.skipped_effects == ["download competition titanic"]


class TestCredentials:
    def test_api_token_is_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KAGGLE_API_TOKEN", '{"username": "u", "key": "k"}')
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        state = CliKaggleClient().credentials()
        assert state.credentials.username == "u"
        assert "environment-token" in state.sources

    def test_env_username_and_key(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setenv("KAGGLE_USERNAME", "u2")
        monkeypatch.setenv("KAGGLE_KEY", "k2")
        state = CliKaggleClient().credentials()
        assert (state.credentials.username, state.credentials.source) == (
            "u2",
            "environment",
        )

    def test_config_file(self, monkeypatch, tmp_path):
        for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "kaggle.json").write_text('{"username": "u3", "key": "k3"}')
        assert CliKaggleClient().credentials().credentials.username == "u3"

    def test_missing_credentials_reports_an_error(self, monkeypatch, tmp_path):
        for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "nope"))
        state = CliKaggleClient().credentials()
        assert not state
        assert "Missing Kaggle credentials file" in state.error

    def test_masking_never_reveals_the_key(self):
        assert kc.Credentials("u", "abcdefghijkl", "x").masked() == "abcd…ijkl"


class TestCliDiscovery:
    """CLI discovery lives here now; these are ported from the deleted kaggle_utils suite."""

    def test_discovers_a_binary_sitting_next_to_the_interpreter(
        self, tmp_path, monkeypatch
    ):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        python_path = bin_dir / "python"
        python_path.write_text("", encoding="utf-8")
        kaggle_path = bin_dir / "kaggle"
        kaggle_path.write_text("#!/bin/sh\n", encoding="utf-8")
        kaggle_path.chmod(0o755)

        monkeypatch.delenv("KAGGLE_CLI_BIN", raising=False)
        monkeypatch.setattr(kc.shutil, "which", lambda _: None)
        monkeypatch.setattr(kc.sys, "executable", str(python_path))

        assert CliKaggleClient._cli_path() == str(kaggle_path)

    def test_honours_the_binary_override(self, tmp_path, monkeypatch):
        override = tmp_path / "my-kaggle"
        override.write_text("#!/bin/sh\n", encoding="utf-8")
        override.chmod(0o755)
        monkeypatch.setenv("KAGGLE_CLI_BIN", str(override))
        assert CliKaggleClient._cli_path() == str(override)

    @pytest.mark.parametrize(
        "boom",
        [
            ModuleNotFoundError("No module named 'kaggle'"),
            # kaggle 1.x touches credentials at import time and can raise OSError.
            OSError("Could not find kaggle.json. Make sure it's located in ..."),
        ],
    )
    def test_a_missing_or_broken_kaggle_package_is_not_fatal(self, monkeypatch, boom):
        import importlib.util

        def raise_it(_name):
            raise boom

        monkeypatch.setattr(CliKaggleClient, "_cli_path", staticmethod(lambda: None))
        monkeypatch.setattr(importlib.util, "find_spec", raise_it)

        assert CliKaggleClient()._prefix() == ["kaggle"]
        assert CliKaggleClient().available() is False

    def test_falls_back_to_the_python_module(self, monkeypatch):
        monkeypatch.setattr(CliKaggleClient, "_cli_path", staticmethod(lambda: None))
        monkeypatch.setattr(
            CliKaggleClient, "_cli_module_available", staticmethod(lambda: True)
        )
        assert CliKaggleClient()._prefix()[1:] == ["-m", "kaggle.cli"]

    def test_falls_back_to_the_bare_name(self, monkeypatch):
        monkeypatch.setattr(CliKaggleClient, "_cli_path", staticmethod(lambda: None))
        monkeypatch.setattr(
            CliKaggleClient, "_cli_module_available", staticmethod(lambda: False)
        )
        assert CliKaggleClient()._prefix() == ["kaggle"]
        assert CliKaggleClient().available() is False


class TestFake:
    def test_seeding_from_real_shaped_csv(self):
        client = FakeKaggleClient.from_csv(
            kernels="Warning: outdated\nref,title,totalVotes\nme/a,A,3\n"
        )
        assert [(k.slug, k.total_votes) for k in client.my_kernels()] == [("a", 3)]

    def test_owner_filter_is_case_insensitive(self):
        client = FakeKaggleClient(datasets=[Dataset.from_row({"ref": "Me/a"})])
        assert len(client.datasets_owned_by("me")) == 1

    def test_records_mutating_calls(self):
        client = FakeKaggleClient()
        client.push_kernel(Path("d"))
        client.submit("c", Path("f"), "m")
        assert [name for name, _ in client.calls] == ["push_kernel", "submit"]

    def test_an_explicitly_empty_credential_state_is_preserved(self):
        state = kc.CredentialState(None, [], "nope")
        assert FakeKaggleClient(credentials_state=state).credentials() is state
