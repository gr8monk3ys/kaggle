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
from dataclasses import dataclass
from pathlib import Path

import requests
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


def probe_blob_upload_auth(username: str, key: str, timeout: int) -> tuple[bool, str]:
    """Check whether credentials are authorized for upload endpoint access.

    Status 401/403 indicates credentials are not authorized for upload actions.
    Status 400 can still indicate auth success with invalid request payload.
    """
    url = "https://www.kaggle.com/api/v1/blobs/upload"
    try:
        response = requests.post(url, auth=(username, key), timeout=timeout)
    except requests.RequestException as exc:
        return False, f"request failed: {exc}"

    if response.status_code in {401, 403}:
        return False, f"{response.status_code} unauthorized for upload endpoint"
    if response.status_code in {200, 201, 202, 400, 405, 413, 415}:
        return True, f"{response.status_code} upload endpoint reachable"
    return False, f"unexpected status {response.status_code}"


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

    upload_ok, upload_msg = probe_blob_upload_auth(creds.username, creds.key, timeout=args.timeout)
    if upload_ok:
        print(f"Upload auth probe: {GREEN}OK{RESET} ({upload_msg})")
    else:
        failures.append(f"upload auth probe failed: {upload_msg}")

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
