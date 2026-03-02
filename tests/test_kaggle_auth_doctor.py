from __future__ import annotations

import json
from pathlib import Path

import kaggle_auth_doctor


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
    class FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    monkeypatch.setattr(
        kaggle_auth_doctor.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(401),
    )
    ok, msg = kaggle_auth_doctor.probe_blob_upload_auth("u", "k", timeout=1)
    assert ok is False
    assert "401" in msg

    monkeypatch.setattr(
        kaggle_auth_doctor.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(400),
    )
    ok, msg = kaggle_auth_doctor.probe_blob_upload_auth("u", "k", timeout=1)
    assert ok is True
    assert "400" in msg
