#!/usr/bin/env python3
"""Validate Kaggle credentials and upload authorization before push operations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The `kaggle` package is only needed for the live upload-auth probe. Import it
# lazily so this module (and everything that imports it, e.g. leaderboard_tracker
# and the manage.sh CLI) loads cleanly in minimal environments where kaggle is
# not installed. The probe degrades gracefully when the names are unavailable,
# and tests monkeypatch `KaggleApi` directly.
# Two separate guards: kaggle 1.x authenticates eagerly in its __init__ and
# calls exit(1) — a SystemExit, not an Exception — when no credentials are
# present, which is exactly the state CI collects tests in. kagglesdk's types
# have no such side effect, so they stay importable and the tests can
# monkeypatch `KaggleApi` alone.
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import KaggleClient, KaggleError


BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
RESET = "\033[0m"

ROOT = Path(__file__).resolve().parents[2]


def kaggle_config_path() -> Path:
    """Where the credentials file is expected to live. Reported in failure hints."""
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


def probe_public_listing(client: KaggleClient, owner: str) -> tuple[bool, str]:
    try:
        rows = client.datasets_by_owner(owner)
    except KaggleError as exc:
        return False, str(exc)
    return True, f"retrieved {len(rows)} public dataset rows"


def probe_blob_upload_auth(client: KaggleClient, timeout: int) -> tuple[bool, str]:
    """Check whether Kaggle's official upload-start flow accepts the credentials."""
    probe = client.probe_upload_auth(timeout)
    return probe.ok, probe.detail


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kaggle credential and upload preflight checks."
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--expected-owner",
        default=None,
        help="Expected dataset owner slug (defaults to credential username).",
    )
    parser.add_argument(
        "--timeout", type=int, default=20, help="HTTP timeout seconds for upload probe."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings in addition to hard failures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parse_args(argv)
    deps = deps or Deps.resolve(timeout=args.timeout)
    root = Path(args.root).resolve()

    print(f"{BLUE}=== Kaggle Auth Doctor ==={RESET}")

    failures: list[str] = []
    warnings: list[str] = []

    state = deps.client.credentials()
    creds = state.credentials
    if creds is None:
        failures.append(state.error or "Unable to resolve credentials.")
    else:
        print(f"Credentials: {GREEN}OK{RESET} ({creds.source})")
        print(f"Username: {creds.username}")
        print(f"Key: {creds.masked()}")

    if creds is None:
        print(f"{RED}FAIL{RESET}: {failures[0]}")
        return 1

    expected_owner = (args.expected_owner or creds.username or "").strip().lower()
    owners, malformed_ids = dataset_id_owners(root)
    if malformed_ids:
        warnings.append(
            f"{len(malformed_ids)} dataset IDs are missing owner/slug format"
        )
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

    listing_ok, listing_msg = probe_public_listing(deps.client, expected_owner)
    if listing_ok:
        print(f"Public listing probe: {GREEN}OK{RESET} ({listing_msg})")
    else:
        warnings.append(f"public listing probe failed: {listing_msg}")

    upload_ok, upload_msg = probe_blob_upload_auth(deps.client, args.timeout)
    if "kaggle package not installed" in upload_msg:
        # The kaggle package is an optional dependency for the live upload probe.
        # Its absence is an environment-setup gap, not a credential failure, so it
        # is a (skippable) warning rather than a hard failure unless --strict.
        warnings.append(
            "kaggle package not installed; upload auth probe skipped "
            "(run `pip install kaggle` to enable it)"
        )
        print(f"Upload auth probe: {YELLOW}SKIP{RESET} (kaggle package not installed)")
    else:
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
