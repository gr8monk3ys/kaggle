from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from kaggle_portfolio.ops import kaggle_auth_doctor
from kaggle_portfolio.shared import kaggle_client as kc
from kaggle_portfolio.shared.kaggle_client import (
    AuthProbe,
    CliKaggleClient,
    Dataset,
    FakeKaggleClient,
    KaggleError,
)


def test_dataset_id_owners_counts_and_malformed(tmp_path):
    ds = tmp_path / "datasets"
    for name, ident in (
        ("a", "owner-one/a"),
        ("b", "owner-one/b"),
        ("c", "missing-slash"),
    ):
        (ds / name).mkdir(parents=True)
        (ds / name / "dataset-metadata.json").write_text(
            json.dumps({"id": ident}), encoding="utf-8"
        )

    owners, malformed = kaggle_auth_doctor.dataset_id_owners(tmp_path)

    assert owners == {"owner-one": 2}
    assert malformed == ["missing-slash"]


class TestPublicListingProbe:
    def test_reports_the_row_count(self):
        client = FakeKaggleClient(datasets=[Dataset.from_row({"ref": "owner/x"})])
        ok, msg = kaggle_auth_doctor.probe_public_listing(client, "owner")
        assert ok is True
        assert "retrieved 1 public dataset rows" in msg

    def test_reports_a_kaggle_failure(self):
        client = FakeKaggleClient(fail_with=KaggleError("403 Forbidden"))
        ok, msg = kaggle_auth_doctor.probe_public_listing(client, "owner")
        assert ok is False
        assert "403 Forbidden" in msg

    def test_a_timeout_is_surfaced_and_never_retried(self, monkeypatch):
        attempts = []

        def fake_run(argv, **kwargs):
            attempts.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

        monkeypatch.setattr(kc.subprocess, "run", fake_run)
        client = CliKaggleClient(timeout=13, retries=3)
        monkeypatch.setattr(client, "_prefix", lambda: ["kaggle"])

        ok, msg = kaggle_auth_doctor.probe_public_listing(client, "owner")

        assert ok is False
        assert "timed out" in msg
        # A retried timeout would multiply the caller's deadline by the attempt count.
        assert attempts == [13]


class TestUploadAuthProbe:
    """The SDK path, now reached through the client rather than a top-level import."""

    def _sdk(self, monkeypatch, should_fail: bool):
        class FakeBlobApiClient:
            def start_blob_upload(self, _request):
                if should_fail:
                    raise RuntimeError(
                        "401 Client Error: Unauthorized for url: "
                        "https://www.kaggle.com/api/v1/blobs/upload"
                    )
                return SimpleNamespace(create_url="https://upload.example", token="tok")

        class FakeClientContext:
            def __enter__(self):
                return SimpleNamespace(
                    blobs=SimpleNamespace(blob_api_client=FakeBlobApiClient())
                )

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeKaggleApi:
            def authenticate(self):
                return None

            def build_kaggle_client(self):
                return FakeClientContext()

        import sys
        import types

        api_mod = types.ModuleType("kaggle.api.kaggle_api_extended")
        api_mod.KaggleApi = FakeKaggleApi
        blob_mod = types.ModuleType("kagglesdk.blobs.types.blob_api_service")
        blob_mod.ApiBlobType = SimpleNamespace(DATASET="dataset")
        blob_mod.ApiStartBlobUploadRequest = lambda: SimpleNamespace()
        for name, mod in (
            ("kaggle", types.ModuleType("kaggle")),
            ("kaggle.api", types.ModuleType("kaggle.api")),
            ("kaggle.api.kaggle_api_extended", api_mod),
            ("kagglesdk", types.ModuleType("kagglesdk")),
            ("kagglesdk.blobs", types.ModuleType("kagglesdk.blobs")),
            ("kagglesdk.blobs.types", types.ModuleType("kagglesdk.blobs.types")),
            ("kagglesdk.blobs.types.blob_api_service", blob_mod),
        ):
            monkeypatch.setitem(sys.modules, name, mod)

    def test_unauthorized_is_classified(self, monkeypatch):
        self._sdk(monkeypatch, should_fail=True)
        ok, msg = kaggle_auth_doctor.probe_blob_upload_auth(
            CliKaggleClient(), timeout=1
        )
        assert ok is False
        assert "401" in msg and "rejected credentials" in msg

    def test_success(self, monkeypatch):
        self._sdk(monkeypatch, should_fail=False)
        ok, msg = kaggle_auth_doctor.probe_blob_upload_auth(
            CliKaggleClient(), timeout=1
        )
        assert ok is True
        assert "succeeded" in msg

    def test_a_missing_sdk_is_reported_not_raised(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", None)
        ok, msg = kaggle_auth_doctor.probe_blob_upload_auth(
            CliKaggleClient(), timeout=1
        )
        assert ok is False
        assert "kaggle package not installed" in msg


class TestMain:
    def test_reports_missing_credentials(self, tmp_path, capsys):
        from kaggle_portfolio.shared.deps import Deps

        client = FakeKaggleClient(
            credentials_state=kc.CredentialState(None, [], "no credentials")
        )
        deps = Deps.for_test(tmp_path, client=client)
        assert kaggle_auth_doctor.main(["--root", str(tmp_path)], deps=deps) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_passes_with_good_credentials_and_probes(self, tmp_path, capsys):
        from kaggle_portfolio.shared.deps import Deps

        client = FakeKaggleClient(
            datasets=[Dataset.from_row({"ref": "tester/x"})],
            auth_probe=AuthProbe(True, "official upload-start probe succeeded"),
        )
        deps = Deps.for_test(tmp_path, client=client)
        rc = kaggle_auth_doctor.main(["--root", str(tmp_path)], deps=deps)
        out = capsys.readouterr().out
        assert "AUTH DOCTOR: PASS" in out
        assert rc == 0
