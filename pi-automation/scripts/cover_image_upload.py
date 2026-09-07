#!/usr/bin/env python3
"""Upload cover images to Kaggle datasets via Playwright.

Auto-discovers datasets with cover.png files and uploads them through the
Kaggle dataset settings page. Uses a JSON tracker to avoid re-uploading.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running from repo root or scripts dir
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_browser as kb


REPO_ROOT = kb.REPO_ROOT
DATASETS_ROOT = REPO_ROOT / "datasets"
TRACKER_PATH = REPO_ROOT / "pi-automation" / "data" / "cover_upload_tracker.json"


def discover_cover_datasets(
    datasets_root: Path, *, only: str | None = None,
) -> list[tuple[str, str, Path]]:
    """Return list of (dataset_dir_name, owner/slug, cover_path) for datasets with cover.png."""
    results: list[tuple[str, str, Path]] = []
    for ds_dir in sorted(datasets_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        if only and ds_dir.name != only:
            continue
        cover = ds_dir / "cover.png"
        meta_path = ds_dir / "dataset-metadata.json"
        if not cover.exists() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dataset_ref = str(meta.get("id", "")).strip().lower()
        except (json.JSONDecodeError, OSError):
            continue
        if not dataset_ref or "/" not in dataset_ref:
            continue
        results.append((ds_dir.name, dataset_ref, cover))
    return results


def _wait_for_settings_tab(page, dataset_ref: str, *, timeout_ms: int) -> None:
    """Navigate to dataset settings and ensure the Settings tab content is loaded."""
    import time
    settings_url = f"https://www.kaggle.com/datasets/{dataset_ref}/settings"
    page.goto(settings_url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(2000)

    # Dismiss cookie banner if present
    cookie_ok = kb.first_available(
        page.get_by_text("OK, Got it.", exact=True).first,
    )
    if cookie_ok is not None:
        cookie_ok.click(timeout=5000)
        page.wait_for_timeout(500)

    # The /settings URL should land on Settings tab, but verify and click if needed
    deadline = time.time() + max(timeout_ms, 8000) / 1000.0
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        edit_btn = page.get_by_role("button", name=re.compile(r"edit image", re.IGNORECASE)).first
        if kb.locator_count(edit_btn):
            return
        # Try clicking the Settings tab in case we landed on Data Card
        settings_tab = kb.first_available(
            page.get_by_role("tab", name=re.compile(r"settings", re.IGNORECASE)).first,
        )
        if settings_tab is not None and attempt <= 3:
            settings_tab.click(timeout=5000)
        page.wait_for_timeout(1500)

    # Edit Image button not found after polling


def upload_cover(page, dataset_ref: str, cover_path: Path, *, timeout_ms: int) -> str:
    """Navigate to dataset settings, open image modal, upload, and save."""
    _wait_for_settings_tab(page, dataset_ref, timeout_ms=timeout_ms)

    if not kb.is_authenticated(page):
        raise RuntimeError("Not authenticated on settings page")

    # Step 1: Click "Edit Image" to open the crop/upload modal
    edit_img_btn = kb.first_available(
        page.get_by_role("button", name=re.compile(r"edit image", re.IGNORECASE)).first,
    )
    if edit_img_btn is None:
        raise RuntimeError(f"Edit Image button not found for {dataset_ref}")

    edit_img_btn.click(timeout=timeout_ms)
    page.wait_for_timeout(1000)

    # Step 2: Set file on hidden input inside the modal
    file_input = kb.first_available(
        page.locator('input[type="file"][accept*=".png"]').first,
        page.locator('input[type="file"][accept*="image"]').first,
        page.locator('input[type="file"]').first,
    )
    if file_input is None:
        raise RuntimeError(f"File input not found in image modal for {dataset_ref}")

    file_input.set_input_files(str(cover_path))
    page.wait_for_timeout(2000)

    # Step 3: Click Save in the modal (not the main settings Save Changes)
    save_btn = kb.first_available(
        page.get_by_role("button", name=re.compile(r"^save$", re.IGNORECASE)).first,
    )
    if save_btn is None:
        raise RuntimeError(f"Save button not found in image modal for {dataset_ref}")

    save_btn.click(timeout=timeout_ms)
    page.wait_for_timeout(2000)

    return f"uploaded {cover_path.name} to {dataset_ref}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload cover images to Kaggle datasets.")
    kb.add_common_browser_args(parser)
    parser.add_argument("--dataset", default=None, help="Only upload for this dataset directory name.")
    parser.add_argument("--tracker", type=Path, default=TRACKER_PATH, help="Tracker JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = discover_cover_datasets(DATASETS_ROOT, only=args.dataset)
    if not datasets:
        print("No datasets with cover.png found.")
        return 0

    tracker = kb.TrackerFile(args.tracker)
    pending = [(name, ref, path) for name, ref, path in datasets if not tracker.has(ref)]
    if not pending:
        print(f"All {len(datasets)} cover images already uploaded.")
        return 0

    print(f"Found {len(pending)} cover images to upload (of {len(datasets)} total):")
    for name, ref, path in pending:
        print(f"  {name} -> {ref} ({path})")

    if args.dry_run:
        print("[dry-run] No uploads performed.")
        return 0

    success = 0
    failures = 0
    with kb.open_kaggle_browser(args) as page:
        for idx, (name, ref, cover_path) in enumerate(pending):
            try:
                result = upload_cover(page, ref, cover_path, timeout_ms=args.timeout_ms)
                tracker.mark(ref, result)
                tracker.save()
                success += 1
                print(f"[done] {name}: {result}")
            except Exception as exc:
                failures += 1
                print(f"[failed] {name}: {exc}")

            if idx < len(pending) - 1:
                kb.human_delay(base=2.0, jitter=1.5)

    print(f"Upload summary: success={success} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
