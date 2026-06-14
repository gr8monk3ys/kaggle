from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from kaggle_portfolio.ops import kaggle_auth_doctor


def test_resolve_credentials_prefers_environment(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key-123")

    creds, err = kaggle_auth_doctor.resolve_credentials()

    assert err is None
    assert creds is not None
    assert creds.username == "env-user"
    assert creds.key == "env-key-123"
    assert creds.source == "environment"


def test_resolve_credentials_reads_file_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "file-user", "key": "file-key"}), encoding="utf-8")
    monkeypatch.setattr(kaggle_auth_doctor, "kaggle_config_path", lambda: cfg)

    creds, err = kaggle_auth_doctor.resolve_credentials()

    assert err is None
    assert creds is not None
    assert creds.username == "file-user"
    assert creds.key == "file-key"
    assert creds.source.startswith("file:")


def test_dataset_id_owners_counts_and_malformed(tmp_path):
    root = tmp_path
    ds = root / "datasets"
    (ds / "a").mkdir(parents=True)
    (ds / "b").mkdir(parents=True)
    (ds / "c").mkdir(parents=True)

    (ds / "a" / "dataset-metadata.json").write_text(
        json.dumps({"id": "owner-one/a"}),
        encoding="utf-8",
    )
    (ds / "b" / "dataset-metadata.json").write_text(
        json.dumps({"id": "owner-one/b"}),
        encoding="utf-8",
    )
    (ds / "c" / "dataset-metadata.json").write_text(
        json.dumps({"id": "missing-slash"}),
        encoding="utf-8",
    )

    owners, malformed = kaggle_auth_doctor.dataset_id_owners(root)

    assert owners == {"owner-one": 2}
    assert malformed == ["missing-slash"]


def test_probe_blob_upload_auth_status_classification(monkeypatch):
    class FakeBlobApiClient:
        def __init__(self, should_fail: bool):
            self.should_fail = should_fail

        def start_blob_upload(self, _request):
            if self.should_fail:
                raise RuntimeError("401 Client Error: Unauthorized for url: https://www.kaggle.com/api/v1/blobs/upload")
            return SimpleNamespace(create_url="https://upload.example", token="tok")

    class FakeClientContext:
        def __init__(self, should_fail: bool):
            self.should_fail = should_fail

        def __enter__(self):
            return SimpleNamespace(blobs=SimpleNamespace(blob_api_client=FakeBlobApiClient(self.should_fail)))

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeKaggleApi:
        should_fail = False

        def authenticate(self):
            return None

        def build_kaggle_client(self):
            return FakeClientContext(self.should_fail)

    monkeypatch.setattr(kaggle_auth_doctor, "KaggleApi", FakeKaggleApi)

    FakeKaggleApi.should_fail = True
    ok, msg = kaggle_auth_doctor.probe_blob_upload_auth(timeout=1)
    assert ok is False
    assert "401" in msg

    FakeKaggleApi.should_fail = False
    ok, msg = kaggle_auth_doctor.probe_blob_upload_auth(timeout=1)
    assert ok is True
    assert "succeeded" in msg


def test_probe_public_listing_passes_timeout_to_subprocess(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ref\nuser/x\n", stderr="")

    monkeypatch.setattr(kaggle_auth_doctor, "kaggle_command", lambda: ["kaggle"])
    monkeypatch.setattr(kaggle_auth_doctor.subprocess, "run", fake_run)

    ok, _ = kaggle_auth_doctor.probe_public_listing("owner", timeout=13)
    assert ok is True
    assert captured.get("timeout") == 13


def test_probe_public_listing_reports_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(kaggle_auth_doctor, "kaggle_command", lambda: ["kaggle"])
    monkeypatch.setattr(kaggle_auth_doctor.subprocess, "run", fake_run)

    ok, msg = kaggle_auth_doctor.probe_public_listing("owner", timeout=2)
    assert ok is False
    assert "timed out" in msg
