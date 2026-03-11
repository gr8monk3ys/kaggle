#!/usr/bin/env python3
"""Validate Kaggle credentials and upload authorization before push operations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kaggle_portfolio.shared.kaggle_utils import kaggle_command, summarize_subprocess_error


BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
RESET = "\033[0m"

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Credentials:
    username: str
    key: str
    source: str




def kaggle_config_path() -> Path:
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / "kaggle.json"

    home_default = Path.home() / ".kaggle" / "kaggle.json"
    if home_default.exists():
        return home_default

    if sys.platform.startswith("linux"):
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        return xdg / "kaggle" / "kaggle.json"
    return home_default


def resolve_credentials() -> tuple[Credentials | None, str | None]:
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key = os.environ.get("KAGGLE_KEY", "").strip()
    if env_user and env_key:
        return Credentials(username=env_user, key=env_key, source="environment"), None

    cfg = kaggle_config_path()
    if not cfg.exists():
        return None, f"Missing Kaggle credentials file: {cfg}"

    try:
        payload = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Invalid kaggle.json: {exc}"
    if not isinstance(payload, dict):
        return None, "kaggle.json must be a JSON object"

    user = str(payload.get("username", "")).strip()
    key = str(payload.get("key", "")).strip()
    if not user or not key:
        return None, "kaggle.json must include non-empty username and key"

    return Credentials(username=user, key=key, source=f"file:{cfg}"), None


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def dataset_id_owners(root: Path) -> tuple[dict[str, int], list[str]]:
    owners: dict[str, int] = {}
    malformed: list[str] = []
    for meta_path in sorted((root / "datasets").glob("*/dataset-metadata.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        ds_id = str(payload.get("id", "")).strip().lower()
        if not ds_id:
            continue
        if "/" not in ds_id:
            malformed.append(ds_id)
            continue
        owner = ds_id.split("/", 1)[0]
        owners[owner] = owners.get(owner, 0) + 1
    return owners, malformed


def probe_public_listing(owner: str) -> tuple[bool, str]:
    cmd = [*kaggle_command(), "datasets", "list", "-s", owner, "--csv"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, summarize_subprocess_error(result.stdout, result.stderr)
    reader = csv.DictReader(io.StringIO(result.stdout))
    count = sum(1 for _ in reader)
    return True, f"retrieved {count} public dataset rows"


def probe_blob_upload_auth(timeout: int) -> tuple[bool, str]:
    """Check whether Kaggle's official upload-start flow accepts the local credentials."""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", prefix="kaggle-auth-doctor-", suffix=".txt", delete=False) as handle:
            handle.write(b"auth-doctor upload probe\n")
            temp_path = handle.name

        api = KaggleApi()
        api.authenticate()

        request = ApiStartBlobUploadRequest()
        request.type = ApiBlobType.DATASET
        request.name = Path(temp_path).name
        request.content_length = os.path.getsize(temp_path)
        request.last_modified_epoch_seconds = int(os.path.getmtime(temp_path))

        with api.build_kaggle_client() as kaggle:
            response = kaggle.blobs.blob_api_client.start_blob_upload(request)

        create_url = str(getattr(response, "create_url", "") or "")
        token = str(getattr(response, "token", "") or "")
        if create_url and token:
            return True, "official upload-start probe succeeded"
        return False, "official upload-start probe returned no create_url/token"
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        if any(marker in lowered for marker in ("401", "403", "unauthenticated", "unauthorized")):
            return False, f"official upload-start probe rejected credentials ({message})"
        return False, f"official upload-start probe failed: {message}"
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kaggle credential and upload preflight checks.")
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--expected-owner",
        default=None,
        help="Expected dataset owner slug (defaults to credential username).",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds for upload probe.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings in addition to hard failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    print(f"{BLUE}=== Kaggle Auth Doctor ==={RESET}")

    failures: list[str] = []
    warnings: list[str] = []

    creds, cred_err = resolve_credentials()
    if cred_err or creds is None:
        failures.append(cred_err or "Unable to resolve credentials.")
    else:
        print(f"Credentials: {GREEN}OK{RESET} ({creds.source})")
        print(f"Username: {creds.username}")
        print(f"Key: {mask_key(creds.key)}")

    if creds is None:
        print(f"{RED}FAIL{RESET}: {failures[0]}")
        return 1

    expected_owner = (args.expected_owner or creds.username).strip().lower()
    owners, malformed_ids = dataset_id_owners(root)
    if malformed_ids:
        warnings.append(f"{len(malformed_ids)} dataset IDs are missing owner/slug format")
    if owners:
        mismatch_owners = sorted(owner for owner in owners if owner != expected_owner)
        if mismatch_owners:
            failures.append(
                "dataset owner mismatch for local metadata: "
                + ", ".join(f"{owner} ({owners[owner]})" for owner in mismatch_owners)
            )
        else:
            print(f"Local metadata owners: {GREEN}OK{RESET} ({expected_owner})")
    else:
        warnings.append("no dataset metadata IDs found for owner consistency check")

    listing_ok, listing_msg = probe_public_listing(expected_owner)
    if listing_ok:
        print(f"Public listing probe: {GREEN}OK{RESET} ({listing_msg})")
    else:
        warnings.append(f"public listing probe failed: {listing_msg}")

    upload_ok, upload_msg = probe_blob_upload_auth(timeout=args.timeout)
    if upload_ok:
        print(f"Upload auth probe: {GREEN}OK{RESET} ({upload_msg})")
    else:
        failures.append(f"upload auth probe failed: {upload_msg}")
        failures.append(
            "download a fresh API token from https://www.kaggle.com/settings/account "
            f"and replace {kaggle_config_path()}"
        )

    if warnings:
        print(f"{YELLOW}Warnings:{RESET}")
        for item in warnings:
            print(f"- {item}")
    if failures:
        print(f"{RED}Failures:{RESET}")
        for item in failures:
            print(f"- {item}")

    if failures or (args.strict and warnings):
        print(f"{RED}AUTH DOCTOR: FAIL{RESET}")
        return 1

    print(f"{GREEN}AUTH DOCTOR: PASS{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
