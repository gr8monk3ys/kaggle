#!/usr/bin/env python3
"""Detect stale notebooks, datasets, and outdated library versions.

Scans the repo for:
  - Notebooks whose .ipynb hasn't been modified in N days (default 60)
  - Datasets whose CSV/Parquet files haven't been modified in N days (default 90)
  - Outdated pinned library versions in pip install cells

Generates a markdown report to medal_ops/reports/latest-stale-content.md.

Usage
-----
    python3 stale_content_detector.py                    # print report
    python3 stale_content_detector.py --max-nb-age 30    # custom notebook threshold
    python3 stale_content_detector.py --max-ds-age 60    # custom dataset threshold
    python3 stale_content_detector.py --today 2026-03-02 # override today for testing
    python3 stale_content_detector.py --json             # output JSON instead of markdown

Invoked by: ./manage.sh stale-content [--max-nb-age N] [--max-ds-age N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from kaggle_utils import configure_logging, parse_iso_date, resolve_today

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "medal_ops" / "reports"
REPORT_FILE = REPORTS_DIR / "latest-stale-content.md"

LOG = configure_logging("stale_content")

# ---------------------------------------------------------------------------
# Known recent library versions — used to flag outdated pinned installs
# ---------------------------------------------------------------------------

KNOWN_RECENT_VERSIONS: dict[str, tuple[int, ...]] = {
    "torch": (2, 5),
    "transformers": (4, 46),
    "sklearn": (1, 5),           # scikit-learn
    "scikit-learn": (1, 5),
    "tensorflow": (2, 18),
    "pandas": (2, 2),
    "numpy": (2, 1),
    "plotly": (6, 0),
}

# How many major versions behind before we flag
MAX_MAJOR_BEHIND = 2


# ---------------------------------------------------------------------------
# Notebook scanning
# ---------------------------------------------------------------------------

def discover_notebooks(root: Path) -> list[dict]:
    """Find all kernel-metadata.json files and resolve notebook paths.

    Returns a list of dicts: {meta_path, nb_path, rel_dir}.
    """
    results = []
    for meta_path in sorted(root.rglob("kernel-metadata.json")):
        # Skip anything buried too deep or in node_modules etc.
        try:
            rel = meta_path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) > 4:
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        code_file = meta.get("code_file", "")
        if not code_file:
            continue
        nb_path = meta_path.parent / code_file
        rel_dir = str(meta_path.parent.relative_to(root))
        results.append({
            "meta_path": meta_path,
            "nb_path": nb_path,
            "rel_dir": rel_dir,
        })
    return results


def find_stale_notebooks(
    root: Path,
    today: date,
    max_age_days: int = 60,
) -> list[dict]:
    """Return notebooks whose .ipynb is older than max_age_days.

    Each entry: {rel_dir, nb_path, last_modified, days_stale}.
    """
    stale = []
    for entry in discover_notebooks(root):
        nb_path = entry["nb_path"]
        if not nb_path.exists():
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(nb_path)).date()
        age = (today - mtime).days
        if age >= max_age_days:
            stale.append({
                "rel_dir": entry["rel_dir"],
                "nb_path": str(nb_path),
                "last_modified": mtime.isoformat(),
                "days_stale": age,
            })
    # Sort most stale first
    stale.sort(key=lambda x: -x["days_stale"])
    return stale


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def discover_datasets(root: Path) -> list[dict]:
    """Find all dataset-metadata.json files.

    Returns a list of dicts: {meta_path, dir_path, rel_dir}.
    """
    results = []
    for meta_path in sorted(root.rglob("dataset-metadata.json")):
        try:
            rel = meta_path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) > 4:
            continue
        results.append({
            "meta_path": meta_path,
            "dir_path": meta_path.parent,
            "rel_dir": str(meta_path.parent.relative_to(root)),
        })
    return results


def find_stale_datasets(
    root: Path,
    today: date,
    max_age_days: int = 90,
) -> list[dict]:
    """Return datasets whose data files (CSV/Parquet) are older than max_age_days.

    Each entry: {rel_dir, oldest_file, last_modified, days_stale}.
    """
    stale = []
    for entry in discover_datasets(root):
        dir_path = entry["dir_path"]
        data_files = list(dir_path.glob("*.csv")) + list(dir_path.glob("*.parquet"))
        if not data_files:
            continue
        # Use the most recently modified data file
        newest_mtime = max(
            datetime.fromtimestamp(os.path.getmtime(f)).date()
            for f in data_files
        )
        age = (today - newest_mtime).days
        if age >= max_age_days:
            stale.append({
                "rel_dir": entry["rel_dir"],
                "oldest_file": str(max(data_files, key=os.path.getmtime)),
                "last_modified": newest_mtime.isoformat(),
                "days_stale": age,
            })
    stale.sort(key=lambda x: -x["days_stale"])
    return stale


# ---------------------------------------------------------------------------
# Outdated library version detection
# ---------------------------------------------------------------------------

def parse_version_tuple(version_str: str) -> tuple[int, ...] | None:
    """Parse '2.0.1' into (2, 0, 1).  Returns None on failure."""
    parts = version_str.strip().split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def is_outdated(pinned: tuple[int, ...], known_recent: tuple[int, ...]) -> bool:
    """Return True if the pinned version is more than MAX_MAJOR_BEHIND major versions behind."""
    if len(pinned) == 0 or len(known_recent) == 0:
        return False
    return (known_recent[0] - pinned[0]) >= MAX_MAJOR_BEHIND


# Matches pkg==version tokens (e.g. torch==2.0.1, numpy==1.24)
_VERSION_PIN_RE = re.compile(r"([\w][\w\-]*)==(\d+(?:\.\d+)*)")


def extract_pinned_versions_from_cell(source: str) -> list[tuple[str, str]]:
    """Return list of (package_name, version_string) found in a pip install cell.

    Only scans cells that contain ``!pip install`` (or ``%pip install``).
    """
    if "pip install" not in source:
        return []
    results = []
    for match in _VERSION_PIN_RE.finditer(source):
        pkg = match.group(1).lower()
        ver = match.group(2)
        results.append((pkg, ver))
    return results


def find_outdated_libraries(root: Path) -> list[dict]:
    """Scan all notebooks for outdated pinned library versions.

    Returns list of {rel_dir, nb_path, library, pinned_version, recent_version}.
    """
    outdated = []
    for entry in discover_notebooks(root):
        nb_path = entry["nb_path"]
        if not nb_path.exists():
            continue
        try:
            with open(nb_path, encoding="utf-8") as f:
                nb = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            for pkg, ver_str in extract_pinned_versions_from_cell(source):
                known = KNOWN_RECENT_VERSIONS.get(pkg)
                if known is None:
                    continue
                pinned = parse_version_tuple(ver_str)
                if pinned is None:
                    continue
                if is_outdated(pinned, known):
                    outdated.append({
                        "rel_dir": entry["rel_dir"],
                        "nb_path": str(nb_path),
                        "library": pkg,
                        "pinned_version": ver_str,
                        "recent_version": ".".join(str(v) for v in known),
                    })
    return outdated


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_markdown_report(
    stale_notebooks: list[dict],
    stale_datasets: list[dict],
    outdated_libs: list[dict],
    today: date,
    max_nb_age: int,
    max_ds_age: int,
) -> str:
    """Build the full markdown report."""
    lines = [
        "# Stale Content Report",
        "",
        f"*Generated: {today.isoformat()}*",
        "",
        f"Thresholds: notebooks >{max_nb_age} days, datasets >{max_ds_age} days",
        "",
    ]

    # --- Stale notebooks ---
    lines.append("## Stale Notebooks")
    lines.append("")
    if stale_notebooks:
        lines.append("| Directory | Last Modified | Days Stale |")
        lines.append("|-----------|---------------|------------|")
        for item in stale_notebooks:
            lines.append(
                f"| {item['rel_dir']} | {item['last_modified']} | {item['days_stale']} |"
            )
    else:
        lines.append("No stale notebooks found.")
    lines.append("")

    # --- Stale datasets ---
    lines.append("## Stale Datasets")
    lines.append("")
    if stale_datasets:
        lines.append("| Directory | Last Modified | Days Stale |")
        lines.append("|-----------|---------------|------------|")
        for item in stale_datasets:
            lines.append(
                f"| {item['rel_dir']} | {item['last_modified']} | {item['days_stale']} |"
            )
    else:
        lines.append("No stale datasets found.")
    lines.append("")

    # --- Outdated libraries ---
    lines.append("## Outdated Library Versions")
    lines.append("")
    if outdated_libs:
        lines.append("| Directory | Library | Pinned | Recent |")
        lines.append("|-----------|---------|--------|--------|")
        for item in outdated_libs:
            lines.append(
                f"| {item['rel_dir']} | {item['library']} "
                f"| {item['pinned_version']} | {item['recent_version']} |"
            )
    else:
        lines.append("No outdated pinned library versions found.")
    lines.append("")

    # --- Summary ---
    total_stale = len(stale_notebooks) + len(stale_datasets) + len(outdated_libs)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Stale notebooks:** {len(stale_notebooks)}")
    lines.append(f"- **Stale datasets:** {len(stale_datasets)}")
    lines.append(f"- **Outdated libraries:** {len(outdated_libs)}")
    lines.append(f"- **Total stale items:** {total_stale}")
    lines.append("")

    # Most urgent items
    if stale_notebooks or stale_datasets:
        lines.append("### Most Urgent")
        lines.append("")
        all_stale = [
            (item["days_stale"], "notebook", item["rel_dir"])
            for item in stale_notebooks
        ] + [
            (item["days_stale"], "dataset", item["rel_dir"])
            for item in stale_datasets
        ]
        all_stale.sort(key=lambda x: -x[0])
        for days, kind, rel_dir in all_stale[:5]:
            lines.append(f"- **{rel_dir}** ({kind}) — {days} days stale")
        lines.append("")

    return "\n".join(lines)


def build_json_report(
    stale_notebooks: list[dict],
    stale_datasets: list[dict],
    outdated_libs: list[dict],
    today: date,
    max_nb_age: int,
    max_ds_age: int,
) -> dict[str, Any]:
    """Build the report as a JSON-serialisable dict."""
    total = len(stale_notebooks) + len(stale_datasets) + len(outdated_libs)
    return {
        "generated": today.isoformat(),
        "thresholds": {"notebook_days": max_nb_age, "dataset_days": max_ds_age},
        "stale_notebooks": stale_notebooks,
        "stale_datasets": stale_datasets,
        "outdated_libraries": outdated_libs,
        "summary": {
            "stale_notebooks": len(stale_notebooks),
            "stale_datasets": len(stale_datasets),
            "outdated_libraries": len(outdated_libs),
            "total": total,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect stale notebooks, datasets, and outdated library versions."
    )
    parser.add_argument(
        "--max-nb-age", type=int, default=60,
        help="Flag notebooks not modified in this many days (default: 60)",
    )
    parser.add_argument(
        "--max-ds-age", type=int, default=90,
        help="Flag datasets not modified in this many days (default: 90)",
    )
    parser.add_argument(
        "--today", type=str, default=None,
        help="Override today's date (YYYY-MM-DD) for testing",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output JSON instead of markdown",
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Root directory to scan (default: repo root)",
    )
    args = parser.parse_args(argv)

    today = resolve_today(args.today)
    root = Path(args.root) if args.root else ROOT

    print(f"{BLUE}=== Stale Content Detector ==={RESET}\n")
    print(f"Today: {today.isoformat()}  |  "
          f"Notebook threshold: {args.max_nb_age}d  |  "
          f"Dataset threshold: {args.max_ds_age}d\n")

    stale_notebooks = find_stale_notebooks(root, today, args.max_nb_age)
    stale_datasets = find_stale_datasets(root, today, args.max_ds_age)
    outdated_libs = find_outdated_libraries(root)

    total = len(stale_notebooks) + len(stale_datasets) + len(outdated_libs)

    if args.output_json:
        report = build_json_report(
            stale_notebooks, stale_datasets, outdated_libs,
            today, args.max_nb_age, args.max_ds_age,
        )
        print(json.dumps(report, indent=2))
    else:
        report_md = build_markdown_report(
            stale_notebooks, stale_datasets, outdated_libs,
            today, args.max_nb_age, args.max_ds_age,
        )
        print(report_md)

        # Write report to file
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(report_md, encoding="utf-8")
        print(f"{GREEN}Report written to {REPORT_FILE.relative_to(ROOT)}{RESET}")

    if total == 0:
        print(f"\n{GREEN}All content is fresh.{RESET}")
    else:
        colour = RED if total > 5 else YELLOW
        print(f"\n{colour}{total} stale item(s) found.{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
