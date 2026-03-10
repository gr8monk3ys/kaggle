#!/usr/bin/env python3
"""Medal operations tooling for Kaggle Grandmaster execution."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from kaggle_portfolio.shared.kaggle_utils import (
    has_kaggle_cli,
    kaggle_command,
    parse_iso_date,
    resolve_today,
    summarize_subprocess_error,
)


DEFAULT_TRACKER_PATH = Path("docs/reports/grandmaster-tracker.md")
DEFAULT_OUTPUT_ROOT = Path("medal_ops")
DEFAULT_SYNC_INPUT_DIRNAME = "sync_inputs"
DEFAULT_KAGGLE_PAGE_SIZE = 20


@dataclass(frozen=True)
class ParsedDeadline:
    competition: str
    deadline_raw: str
    deadline_date: date | None
    days_to_deadline: int | None
    teams: str
    difficulty: str
    strategy: str


def extract_first_int(value: str) -> int | None:
    match = re.search(r"-?\d[\d,]*", value)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def extract_all_ints(value: str) -> list[int]:
    return [int(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*", value)]




def parse_deadline_date(text: str) -> date | None:
    text = text.strip()
    if not text or text in {"—", "-", "TBD"}:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def extract_section(content: str, heading: str) -> str:
    pattern = re.compile(rf"{re.escape(heading)}\n(.*?)(?=\n## |\n### |\Z)", re.DOTALL)
    match = pattern.search(content)
    return match.group(1) if match else ""


def split_markdown_tables(section: str) -> list[list[dict[str, str]]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.endswith("|"):
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    tables: list[list[dict[str, str]]] = []
    for block in blocks:
        if len(block) < 2:
            continue
        header = [cell.strip() for cell in block[0].strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for line in block[2:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rows.append(dict(zip(header, cells)))
        tables.append(rows)
    return tables


def parse_last_updated(content: str) -> date | None:
    match = re.search(r"\*\*Last Updated:\*\*\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", content)
    if not match:
        return None
    return parse_iso_date(match.group(1))


def parse_tier_requirements(content: str) -> dict[str, dict[str, int]]:
    section = extract_section(content, "## Tier Requirements")
    tables = split_markdown_tables(section)
    if not tables:
        return {}

    requirements: dict[str, dict[str, int]] = {}
    for row in tables[0]:
        category_raw = row.get("Category", "").strip().lower()
        category = category_raw.replace("*", "").strip()
        if not category:
            continue

        expert_cell = row.get("Expert", "")
        grandmaster_cell = row.get("Grandmaster", "")

        expert_numbers = extract_all_ints(expert_cell)
        grandmaster_numbers = extract_all_ints(grandmaster_cell)

        data: dict[str, int] = {}
        if expert_numbers:
            data["expert_bronze"] = expert_numbers[0]
        if grandmaster_numbers:
            data["grandmaster_gold"] = grandmaster_numbers[0]
        if category == "discussion" and len(grandmaster_numbers) > 1:
            data["grandmaster_total"] = grandmaster_numbers[1]

        requirements[category] = data

    return requirements


def parse_progress_metrics(content: str, category: str) -> dict[str, Any]:
    section = extract_section(content, f"### {category.title()}")
    tables = split_markdown_tables(section)
    if not tables:
        return {}

    metrics: dict[str, Any] = {}
    for row in tables[0]:
        label = row.get("Status", "").strip().lower()
        current = row.get("Current", "").strip()
        if "tier" in label:
            metrics["tier"] = current
            continue

        number = extract_first_int(current)
        if number is None:
            continue

        if "gold" in label:
            metrics["gold"] = number
        elif "silver" in label:
            metrics["silver"] = number
        elif "bronze" in label:
            metrics["bronze"] = number
        elif "entered" in label:
            metrics["entered"] = number
        elif "total notebooks" in label:
            metrics["total_notebooks"] = number
        elif "total datasets" in label:
            metrics["total_datasets"] = number
        elif "total posts" in label:
            metrics["total_posts"] = number
        elif "total votes" in label:
            metrics["total_votes"] = number

    return metrics


def parse_active_competitions(content: str, today: date) -> list[ParsedDeadline]:
    section = extract_section(content, "### Competitions")
    tables = split_markdown_tables(section)
    if len(tables) < 2:
        return []

    rows = tables[1]
    parsed: list[ParsedDeadline] = []
    for row in rows:
        comp_name = row.get("Competition", "").strip()
        deadline_raw = row.get("Deadline", "").strip()
        deadline_date = parse_deadline_date(deadline_raw)
        days_to_deadline = (deadline_date - today).days if deadline_date else None
        parsed.append(
            ParsedDeadline(
                competition=comp_name,
                deadline_raw=deadline_raw,
                deadline_date=deadline_date,
                days_to_deadline=days_to_deadline,
                teams=row.get("Teams", "").strip(),
                difficulty=row.get("Medal Difficulty", "").strip(),
                strategy=row.get("Strategy", "").strip(),
            )
        )
    return parsed


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "").strip().lower())


def update_last_updated_line(content: str, today: date) -> tuple[str, bool]:
    pattern = re.compile(r"(\*\*Last Updated:\*\*\s*)([0-9]{4}-[0-9]{2}-[0-9]{2})")
    match = pattern.search(content)
    if not match:
        return content, False
    previous = match.group(2)
    replacement = f"{match.group(1)}{today.isoformat()}"
    updated = pattern.sub(replacement, content, count=1)
    return updated, previous != today.isoformat()


def update_progress_current_cell(
    content: str, section_heading: str, status_label: str, new_current: str
) -> tuple[str, bool]:
    section_pattern = re.compile(
        rf"(### {re.escape(section_heading)}\n)(.*?)(?=\n### |\n## |\Z)", re.DOTALL
    )
    match = section_pattern.search(content)
    if not match:
        return content, False

    section_body = match.group(2)
    lines = section_body.splitlines()
    target_label = normalize_label(status_label)
    changed = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if normalize_label(cells[0]) != target_label:
            continue
        previous = cells[2]
        cells[2] = new_current
        lines[idx] = "| " + " | ".join(cells) + " |"
        changed = changed or (previous != new_current)
        break

    if not changed:
        return content, False

    new_section = "\n".join(lines)
    updated = content[: match.start(2)] + new_section + content[match.end(2) :]
    return updated, True


def run_checked_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"Command failed: {' '.join(cmd)}\n{exc}") from exc
    if result.returncode != 0:
        stderr = summarize_subprocess_error(result.stdout, result.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return result.stdout


def run_kaggle_csv(args: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    output = run_checked_command([*kaggle_command(), *args, "--csv"])
    reader = csv.DictReader(io.StringIO(output))
    rows = [dict(row) for row in reader]
    fieldnames = [name for name in (reader.fieldnames or []) if name]
    return rows, fieldnames


def run_kaggle_csv_paginated(
    args: list[str], *, page_size: int | None = None
) -> tuple[list[dict[str, str]], list[str]]:
    rows_all: list[dict[str, str]] = []
    fieldnames: list[str] = []
    page = 1
    expected_page_size = page_size or DEFAULT_KAGGLE_PAGE_SIZE

    while True:
        paged_args = [*args]
        if page_size is not None:
            paged_args.extend(["--page-size", str(page_size)])
        paged_args.extend(["--page", str(page)])
        rows, fields = run_kaggle_csv(paged_args)
        if rows and not fieldnames:
            fieldnames = fields
        rows_all.extend(rows)
        if len(rows) < expected_page_size:
            break
        page += 1

    return rows_all, fieldnames


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = [name for name in (reader.fieldnames or []) if name]
    return rows, fieldnames


def find_key(keys: list[str], predicates: list[str]) -> str | None:
    for wanted in predicates:
        for key in keys:
            if wanted in key.lower():
                return key
    return None


def parse_truthy(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "t"}


def format_columns(fieldnames: list[str]) -> str:
    if not fieldnames:
        return "(none)"
    return ", ".join(fieldnames)


def parse_vote_total(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    source_label: str,
    strict: bool,
) -> tuple[int, str | None]:
    keys = fieldnames or (list(rows[0].keys()) if rows else [])
    vote_key = find_key(keys, ["totalvotes", "votecount", "votes"])
    if not vote_key:
        vote_key = find_key(keys, ["vote"])
    if not vote_key:
        if strict:
            raise SystemExit(
                f"{source_label} is missing a vote column. "
                f"Expected one containing totalVotes, voteCount, or votes. "
                f"Found columns: {format_columns(keys)}"
            )
        return 0, None

    total = 0
    for row in rows:
        total += extract_first_int(row.get(vote_key, "")) or 0
    return total, vote_key


def parse_integer_total(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    source_label: str,
    *,
    predicates: list[str],
    metric_name: str,
    strict: bool,
) -> tuple[int | None, str | None]:
    keys = fieldnames or (list(rows[0].keys()) if rows else [])
    metric_key = find_key(keys, predicates)
    if not metric_key:
        if strict:
            raise SystemExit(
                f"{source_label} is missing a {metric_name} column. "
                f"Found columns: {format_columns(keys)}"
            )
        return None, None

    total = 0
    for row in rows:
        total += extract_first_int(row.get(metric_key, "")) or 0
    return total, metric_key


def count_vote_threshold_medals(
    rows: list[dict[str, str]], vote_key: str | None
) -> dict[str, int]:
    counts = {"gold": 0, "silver": 0, "bronze": 0}
    if not vote_key:
        return counts

    for row in rows:
        votes = extract_first_int(row.get(vote_key, "")) or 0
        if votes >= 50:
            counts["gold"] += 1
        elif votes >= 20:
            counts["silver"] += 1
        elif votes >= 5:
            counts["bronze"] += 1

    return counts


def parse_entered_total(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    source_label: str,
    strict: bool,
) -> tuple[int | None, str | None]:
    keys = fieldnames or (list(rows[0].keys()) if rows else [])
    entered_key = find_key(keys, ["userhasentered", "hasentered", "entered"])
    if not entered_key:
        if strict:
            raise SystemExit(
                f"{source_label} is missing an entered column. "
                f"Expected one containing userHasEntered, hasEntered, or entered. "
                f"Found columns: {format_columns(keys)}"
            )
        return None, None
    total = sum(1 for row in rows if parse_truthy(row.get(entered_key, "")))
    return total, entered_key


def fetch_live_kaggle_metrics() -> dict[str, Any]:
    if not has_kaggle_cli():
        raise SystemExit(
            "kaggle CLI not found. Install/authenticate it, or run sync with exported CSV files "
            "(--kernels-csv, --datasets-csv, --competitions-csv)."
        )

    kernels_rows, kernels_columns = run_kaggle_csv_paginated(
        ["kernels", "list", "--mine"], page_size=100
    )
    datasets_rows, datasets_columns = run_kaggle_csv_paginated(["datasets", "list", "-m"])
    entered_rows, _ = run_kaggle_csv_paginated(
        ["competitions", "list", "--group", "entered"], page_size=100
    )

    notebooks_votes, notebooks_vote_key = parse_vote_total(
        kernels_rows, kernels_columns, "kaggle kernels list output", strict=True
    )
    notebook_medals = count_vote_threshold_medals(kernels_rows, notebooks_vote_key)
    datasets_votes, datasets_vote_key = parse_vote_total(
        datasets_rows, datasets_columns, "kaggle datasets list output", strict=True
    )
    dataset_medals = count_vote_threshold_medals(datasets_rows, datasets_vote_key)
    datasets_downloads, datasets_download_key = parse_integer_total(
        datasets_rows,
        datasets_columns,
        "kaggle datasets list output",
        predicates=["downloadcount", "downloads", "download"],
        metric_name="download",
        strict=False,
    )
    return {
        "notebooks_count": len(kernels_rows),
        "notebooks_total_votes": notebooks_votes,
        "notebooks_vote_key": notebooks_vote_key,
        "notebooks_gold": notebook_medals["gold"],
        "notebooks_silver": notebook_medals["silver"],
        "notebooks_bronze": notebook_medals["bronze"],
        "datasets_count": len(datasets_rows),
        "datasets_total_votes": datasets_votes,
        "datasets_vote_key": datasets_vote_key,
        "datasets_total_downloads": datasets_downloads,
        "datasets_download_key": datasets_download_key,
        "datasets_gold": dataset_medals["gold"],
        "datasets_silver": dataset_medals["silver"],
        "datasets_bronze": dataset_medals["bronze"],
        "competitions_entered": len(entered_rows),
        "competitions_entered_key": "group=entered",
    }


def fetch_metrics_from_csv(
    kernels_csv: Path, datasets_csv: Path, competitions_csv: Path | None
) -> dict[str, Any]:
    kernels_rows, kernels_columns = load_csv_rows(kernels_csv)
    datasets_rows, datasets_columns = load_csv_rows(datasets_csv)

    competition_rows: list[dict[str, str]] = []
    competition_columns: list[str] = []
    if competitions_csv:
        competition_rows, competition_columns = load_csv_rows(competitions_csv)

    notebooks_votes, notebooks_vote_key = parse_vote_total(
        kernels_rows, kernels_columns, f"kernels CSV ({kernels_csv})", strict=True
    )
    notebook_medals = count_vote_threshold_medals(kernels_rows, notebooks_vote_key)
    datasets_votes, datasets_vote_key = parse_vote_total(
        datasets_rows, datasets_columns, f"datasets CSV ({datasets_csv})", strict=True
    )
    dataset_medals = count_vote_threshold_medals(datasets_rows, datasets_vote_key)
    datasets_downloads, datasets_download_key = parse_integer_total(
        datasets_rows,
        datasets_columns,
        f"datasets CSV ({datasets_csv})",
        predicates=["downloadcount", "downloads", "download"],
        metric_name="download",
        strict=False,
    )
    if competitions_csv:
        entered_total, entered_key = parse_entered_total(
            competition_rows,
            competition_columns,
            f"competitions CSV ({competitions_csv})",
            strict=True,
        )
    else:
        entered_total, entered_key = None, None

    return {
        "notebooks_count": len(kernels_rows),
        "notebooks_total_votes": notebooks_votes,
        "notebooks_vote_key": notebooks_vote_key,
        "notebooks_gold": notebook_medals["gold"],
        "notebooks_silver": notebook_medals["silver"],
        "notebooks_bronze": notebook_medals["bronze"],
        "datasets_count": len(datasets_rows),
        "datasets_total_votes": datasets_votes,
        "datasets_vote_key": datasets_vote_key,
        "datasets_total_downloads": datasets_downloads,
        "datasets_download_key": datasets_download_key,
        "datasets_gold": dataset_medals["gold"],
        "datasets_silver": dataset_medals["silver"],
        "datasets_bronze": dataset_medals["bronze"],
        "competitions_entered": entered_total,
        "competitions_entered_key": entered_key,
    }


def write_template_asset(path: Path, content: str, force: bool) -> str:
    if path.exists():
        if not force:
            return "skipped"
        path.write_text(content, encoding="utf-8")
        return "overwritten"
    path.write_text(content, encoding="utf-8")
    return "created"


def ensure_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | 0o111)


def generate_sync_template_assets(output_dir: Path, force: bool) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels_path = output_dir / "kernels.csv"
    datasets_path = output_dir / "datasets.csv"
    competitions_path = output_dir / "competitions.csv"
    script_path = output_dir / "export_kaggle_sync_csv.sh"
    readme_path = output_dir / "README.md"

    script_content = """#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p "${OUT_DIR}"

kaggle kernels list --mine --page-size 100 --csv > "${OUT_DIR}/kernels.csv"
kaggle datasets list -m --csv > "${OUT_DIR}/datasets.csv"
kaggle competitions list --group entered --page-size 100 --csv > "${OUT_DIR}/competitions.csv"

echo "CSV exports written to ${OUT_DIR}"
"""

    readme_content = """# Sync Input Bundle

This folder was generated by `python3 -m kaggle_portfolio.ops.medal_ops sync-template`.

## Files

- `kernels.csv`: expected to include a vote column such as `totalVotes`.
- `datasets.csv`: expected to include a vote column such as `voteCount`.
- `competitions.csv`: optional for sync; `kaggle competitions list --group entered` output is preferred.
- `export_kaggle_sync_csv.sh`: helper that exports CSV files from Kaggle CLI.

## Quick Start

```bash
# Requires authenticated kaggle CLI:
./export_kaggle_sync_csv.sh

# Dry-run sync:
python3 -m kaggle_portfolio.ops.medal_ops sync --dry-run \\
  --kernels-csv kernels.csv \\
  --datasets-csv datasets.csv \\
  --competitions-csv competitions.csv
```
"""

    statuses = {
        str(kernels_path): write_template_asset(
            kernels_path, "title,totalVotes\nsample-notebook,0\n", force
        ),
        str(datasets_path): write_template_asset(
            datasets_path, "title,voteCount\nsample-dataset,0\n", force
        ),
        str(competitions_path): write_template_asset(
            competitions_path, "competition,userHasEntered\nsample-competition,false\n", force
        ),
        str(script_path): write_template_asset(script_path, script_content, force),
        str(readme_path): write_template_asset(readme_path, readme_content, force),
    }

    ensure_executable(script_path)
    return statuses


def apply_tracker_sync(content: str, today: date, live: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    before = {
        "competitions": parse_progress_metrics(content, "Competitions"),
        "notebooks": parse_progress_metrics(content, "Notebooks"),
        "datasets": parse_progress_metrics(content, "Datasets"),
    }
    updated = content
    changed_fields: list[str] = []

    updated, changed = update_last_updated_line(updated, today)
    if changed:
        changed_fields.append("Last Updated")

    updated, changed = update_progress_current_cell(
        updated, "Notebooks", "Total notebooks", f"{live['notebooks_count']} (on Kaggle)"
    )
    if changed:
        changed_fields.append("Notebooks.Total notebooks")

    updated, changed = update_progress_current_cell(
        updated, "Notebooks", "Total votes", str(live["notebooks_total_votes"])
    )
    if changed:
        changed_fields.append("Notebooks.Total votes")

    if isinstance(live.get("notebooks_gold"), int):
        updated, changed = update_progress_current_cell(
            updated, "Notebooks", "Gold medals (50+ votes)", str(live["notebooks_gold"])
        )
        if changed:
            changed_fields.append("Notebooks.Gold medals")

    if isinstance(live.get("notebooks_silver"), int):
        updated, changed = update_progress_current_cell(
            updated, "Notebooks", "Silver medals (20+ votes)", str(live["notebooks_silver"])
        )
        if changed:
            changed_fields.append("Notebooks.Silver medals")

    if isinstance(live.get("notebooks_bronze"), int):
        updated, changed = update_progress_current_cell(
            updated, "Notebooks", "Bronze medals (5+ votes)", str(live["notebooks_bronze"])
        )
        if changed:
            changed_fields.append("Notebooks.Bronze medals")

    updated, changed = update_progress_current_cell(
        updated, "Datasets", "Total datasets", f"{live['datasets_count']} (on Kaggle)"
    )
    if changed:
        changed_fields.append("Datasets.Total datasets")

    updated, changed = update_progress_current_cell(
        updated, "Datasets", "Total votes", str(live["datasets_total_votes"])
    )
    if changed:
        changed_fields.append("Datasets.Total votes")

    if isinstance(live.get("datasets_total_downloads"), int):
        updated, changed = update_progress_current_cell(
            updated, "Datasets", "Total downloads", str(live["datasets_total_downloads"])
        )
        if changed:
            changed_fields.append("Datasets.Total downloads")

    if isinstance(live.get("datasets_gold"), int):
        updated, changed = update_progress_current_cell(
            updated, "Datasets", "Gold medals (50+ votes)", str(live["datasets_gold"])
        )
        if changed:
            changed_fields.append("Datasets.Gold medals")

    if isinstance(live.get("datasets_silver"), int):
        updated, changed = update_progress_current_cell(
            updated, "Datasets", "Silver medals (20+ votes)", str(live["datasets_silver"])
        )
        if changed:
            changed_fields.append("Datasets.Silver medals")

    if isinstance(live.get("datasets_bronze"), int):
        updated, changed = update_progress_current_cell(
            updated, "Datasets", "Bronze medals (5+ votes)", str(live["datasets_bronze"])
        )
        if changed:
            changed_fields.append("Datasets.Bronze medals")

    if isinstance(live.get("competitions_entered"), int):
        updated, changed = update_progress_current_cell(
            updated, "Competitions", "Entered", str(live["competitions_entered"])
        )
        if changed:
            changed_fields.append("Competitions.Entered")

    after = {
        "competitions": parse_progress_metrics(updated, "Competitions"),
        "notebooks": parse_progress_metrics(updated, "Notebooks"),
        "datasets": parse_progress_metrics(updated, "Datasets"),
    }

    return updated, {"before": before, "after": after, "changed_fields": changed_fields}


def build_snapshot(content: str, today: date) -> dict[str, Any]:
    tracker_last_updated = parse_last_updated(content)
    stale_days = (today - tracker_last_updated).days if tracker_last_updated else None

    requirements = parse_tier_requirements(content)
    competitions = parse_progress_metrics(content, "Competitions")
    notebooks = parse_progress_metrics(content, "Notebooks")
    datasets = parse_progress_metrics(content, "Datasets")
    discussion = parse_progress_metrics(content, "Discussion")
    active_competitions = parse_active_competitions(content, today)

    def gap(current: dict[str, Any], req: dict[str, int], key: str, req_key: str) -> int | None:
        if key not in current or req_key not in req:
            return None
        return max(0, req[req_key] - int(current[key]))

    category_summary = {
        "competitions": {
            **competitions,
            "gold_goal": requirements.get("competitions", {}).get("grandmaster_gold"),
            "gold_gap": gap(competitions, requirements.get("competitions", {}), "gold", "grandmaster_gold"),
            "expert_bronze_goal": requirements.get("competitions", {}).get("expert_bronze"),
            "expert_bronze_gap": gap(competitions, requirements.get("competitions", {}), "bronze", "expert_bronze"),
        },
        "notebooks": {
            **notebooks,
            "gold_goal": requirements.get("notebooks", {}).get("grandmaster_gold"),
            "gold_gap": gap(notebooks, requirements.get("notebooks", {}), "gold", "grandmaster_gold"),
            "expert_bronze_goal": requirements.get("notebooks", {}).get("expert_bronze"),
            "expert_bronze_gap": gap(notebooks, requirements.get("notebooks", {}), "bronze", "expert_bronze"),
        },
        "datasets": {
            **datasets,
            "gold_goal": requirements.get("datasets", {}).get("grandmaster_gold"),
            "gold_gap": gap(datasets, requirements.get("datasets", {}), "gold", "grandmaster_gold"),
            "expert_bronze_goal": requirements.get("datasets", {}).get("expert_bronze"),
            "expert_bronze_gap": gap(datasets, requirements.get("datasets", {}), "bronze", "expert_bronze"),
        },
        "discussion": {
            **discussion,
            "gold_goal": requirements.get("discussion", {}).get("grandmaster_gold"),
            "gold_gap": gap(discussion, requirements.get("discussion", {}), "gold", "grandmaster_gold"),
            "total_goal": requirements.get("discussion", {}).get("grandmaster_total"),
            "total_gap": gap(discussion, requirements.get("discussion", {}), "total_posts", "grandmaster_total"),
            "expert_bronze_goal": requirements.get("discussion", {}).get("expert_bronze"),
            "expert_bronze_gap": gap(discussion, requirements.get("discussion", {}), "bronze", "expert_bronze"),
        },
    }

    active_competitions_json = [
        {
            "competition": deadline.competition,
            "deadline_raw": deadline.deadline_raw,
            "deadline_date": deadline.deadline_date.isoformat() if deadline.deadline_date else None,
            "days_to_deadline": deadline.days_to_deadline,
            "teams": deadline.teams,
            "difficulty": deadline.difficulty,
            "strategy": deadline.strategy,
        }
        for deadline in active_competitions
    ]

    return {
        "generated_on": today.isoformat(),
        "tracker_last_updated": tracker_last_updated.isoformat() if tracker_last_updated else None,
        "tracker_stale_days": stale_days,
        "categories": category_summary,
        "active_competitions": active_competitions_json,
    }


def load_latest_snapshot_path(history_dir: Path) -> Path | None:
    if not history_dir.exists():
        return None
    snapshots = sorted(history_dir.glob("snapshot-*.json"))
    if not snapshots:
        return None
    return snapshots[-1]


def load_latest_snapshot(history_dir: Path) -> dict[str, Any] | None:
    latest_path = load_latest_snapshot_path(history_dir)
    if not latest_path:
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))


def load_all_snapshots(history_dir: Path) -> list[dict[str, Any]]:
    if not history_dir.exists():
        return []
    snapshots = sorted(history_dir.glob("snapshot-*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in snapshots]


def write_snapshot(history_dir: Path, snapshot: dict[str, Any]) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    latest_path = load_latest_snapshot_path(history_dir)
    if latest_path:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        same_day = latest.get("generated_on") == snapshot.get("generated_on")
        same_tracker_state = (
            latest.get("tracker_last_updated") == snapshot.get("tracker_last_updated")
            and latest.get("categories") == snapshot.get("categories")
            and latest.get("active_competitions") == snapshot.get("active_competitions")
        )
        if same_day and same_tracker_state:
            return latest_path

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    path = history_dir / f"snapshot-{stamp}.json"
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return path


def delta(current: dict[str, Any], previous: dict[str, Any] | None, path: tuple[str, ...]) -> int | None:
    if not previous:
        return None
    curr: Any = current
    prev: Any = previous
    for key in path:
        if key not in curr or key not in prev:
            return None
        curr = curr[key]
        prev = prev[key]
    if not isinstance(curr, int) or not isinstance(prev, int):
        return None
    return curr - prev


def format_delta(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value > 0:
        return f"+{value}"
    return str(value)


def nested_int(snapshot: dict[str, Any], path: tuple[str, ...]) -> int | None:
    node: Any = snapshot
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if not isinstance(node, int):
        return None
    return node


def snapshot_generated_date(snapshot: dict[str, Any]) -> date | None:
    generated = snapshot.get("generated_on")
    if not isinstance(generated, str):
        return None
    return parse_iso_date(generated)


def weekly_velocity(snapshots: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    if len(snapshots) < 2:
        return None

    first = snapshots[0]
    last = snapshots[-1]
    first_date = snapshot_generated_date(first)
    last_date = snapshot_generated_date(last)
    if not first_date or not last_date:
        return None

    days = (last_date - first_date).days
    if days <= 0:
        return None

    first_val = nested_int(first, path)
    last_val = nested_int(last, path)
    if first_val is None or last_val is None:
        return None

    return (last_val - first_val) * 7.0 / days


def format_velocity(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}/wk"


def estimate_eta_weeks(gap: int | None, velocity_per_week: float | None) -> str:
    if gap is None:
        return "n/a"
    if gap <= 0:
        return "0.0"
    if velocity_per_week is None or velocity_per_week <= 0:
        return "n/a"
    return f"{gap / velocity_per_week:.1f}"


def top_actions(snapshot: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    stale_days = snapshot.get("tracker_stale_days")
    categories = snapshot["categories"]

    if isinstance(stale_days, int) and stale_days > 7:
        actions.append(f"Refresh `grandmaster-tracker.md` (currently {stale_days} days stale).")

    discussion = categories["discussion"]
    if isinstance(discussion.get("expert_bronze_gap"), int) and discussion["expert_bronze_gap"] > 0:
        actions.append(
            f"Prioritize discussion medals: need {discussion['expert_bronze_gap']} more bronze-equivalent medals for Discussion Expert."
        )

    competitions = categories["competitions"]
    if int(competitions.get("entered", 0)) < 3:
        actions.append("Increase active competition footprint to at least 3 simultaneous entries.")

    notebooks = categories["notebooks"]
    if int(notebooks.get("total_votes", 0)) < 20:
        actions.append("Run a 7-day notebook promotion sprint on the top 5 notebooks to accelerate first notebook medals.")

    datasets = categories["datasets"]
    if int(datasets.get("total_votes", 0)) == 0:
        actions.append("Improve dataset discoverability: add full data dictionaries and publish one baseline notebook per dataset.")

    return actions[:5]


def generate_scorecard_markdown(snapshot: dict[str, Any], previous: dict[str, Any] | None) -> str:
    categories = snapshot["categories"]
    stale_days = snapshot.get("tracker_stale_days")
    generated_on = snapshot["generated_on"]
    tracker_last_updated = snapshot.get("tracker_last_updated") or "unknown"

    lines = [
        "# Kaggle Medal Ops Scorecard",
        "",
        f"- Generated: {generated_on}",
        f"- Tracker last updated: {tracker_last_updated}",
        "",
    ]

    if isinstance(stale_days, int):
        freshness = "stale" if stale_days > 7 else "fresh"
        lines.append(f"- Tracker freshness: {stale_days} day(s) ({freshness})")
        lines.append("")

    lines.extend(
        [
            "## Progress Snapshot",
            "",
            "| Category | Tier | Gold | Gold Goal | Gold Gap | Bronze | Votes/Posts |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| Competitions | {categories['competitions'].get('tier', 'n/a')} | {categories['competitions'].get('gold', 0)} | {categories['competitions'].get('gold_goal', 'n/a')} | {categories['competitions'].get('gold_gap', 'n/a')} | {categories['competitions'].get('bronze', 0)} | Entered: {categories['competitions'].get('entered', 0)} |",
            f"| Notebooks | {categories['notebooks'].get('tier', 'n/a')} | {categories['notebooks'].get('gold', 0)} | {categories['notebooks'].get('gold_goal', 'n/a')} | {categories['notebooks'].get('gold_gap', 'n/a')} | {categories['notebooks'].get('bronze', 0)} | Votes: {categories['notebooks'].get('total_votes', 0)} |",
            f"| Datasets | {categories['datasets'].get('tier', 'n/a')} | {categories['datasets'].get('gold', 0)} | {categories['datasets'].get('gold_goal', 'n/a')} | {categories['datasets'].get('gold_gap', 'n/a')} | {categories['datasets'].get('bronze', 0)} | Votes: {categories['datasets'].get('total_votes', 0)} |",
            f"| Discussion | {categories['discussion'].get('tier', 'n/a')} | {categories['discussion'].get('gold', 0)} | {categories['discussion'].get('gold_goal', 'n/a')} | {categories['discussion'].get('gold_gap', 'n/a')} | {categories['discussion'].get('bronze', 0)} | Posts: {categories['discussion'].get('total_posts', 0)} |",
            "",
        ]
    )

    lines.extend(
        [
            "## Weekly Delta (vs previous snapshot)",
            "",
            f"- Notebook votes: {categories['notebooks'].get('total_votes', 0)} ({format_delta(delta(snapshot, previous, ('categories', 'notebooks', 'total_votes')))})",
            f"- Dataset votes: {categories['datasets'].get('total_votes', 0)} ({format_delta(delta(snapshot, previous, ('categories', 'datasets', 'total_votes')))})",
            f"- Discussion posts: {categories['discussion'].get('total_posts', 0)} ({format_delta(delta(snapshot, previous, ('categories', 'discussion', 'total_posts')))})",
            f"- Competition entries: {categories['competitions'].get('entered', 0)} ({format_delta(delta(snapshot, previous, ('categories', 'competitions', 'entered')))})",
            "",
        ]
    )

    deadlines = []
    for raw in snapshot.get("active_competitions", []):
        days = raw.get("days_to_deadline")
        if isinstance(days, int):
            deadlines.append(raw)
    deadlines.sort(key=lambda item: item["days_to_deadline"])

    lines.append("## Deadline Radar")
    lines.append("")
    if not deadlines:
        lines.append("- No parseable deadlines found in tracker.")
    else:
        for item in deadlines[:8]:
            days = item["days_to_deadline"]
            if days < 0:
                status = f"overdue by {-days}d"
            elif days == 0:
                status = "due today"
            elif days <= 7:
                status = f"due in {days}d (urgent)"
            else:
                status = f"due in {days}d"
            lines.append(f"- {item['competition']} ({item['deadline_raw']}): {status}")
    lines.append("")

    lines.append("## Top Actions")
    lines.append("")
    for action in top_actions(snapshot):
        lines.append(f"- {action}")
    lines.append("")

    return "\n".join(lines)


def generate_weekly_plan_markdown(snapshot: dict[str, Any]) -> str:
    categories = snapshot["categories"]
    discussion = categories["discussion"]
    competitions = categories["competitions"]
    notebooks = categories["notebooks"]
    datasets = categories["datasets"]

    discussion_gap = int(discussion.get("expert_bronze_gap") or 0)
    discussion_target = min(14, max(7, discussion_gap // 5 if discussion_gap else 7))
    competition_entries = int(competitions.get("entered", 0))
    competition_target = 3 if competition_entries < 3 else competition_entries

    upcoming = [
        item
        for item in snapshot.get("active_competitions", [])
        if isinstance(item.get("days_to_deadline"), int) and item["days_to_deadline"] >= 0
    ]
    upcoming.sort(key=lambda item: item["days_to_deadline"])
    urgent = [item for item in upcoming if item["days_to_deadline"] <= 10][:3]

    lines = [
        "# Kaggle Weekly Plan",
        "",
        f"- Plan generated: {snapshot['generated_on']}",
        "",
        "## Primary Objectives (7 days)",
        "",
        f"- Earn `{discussion_target}` discussion medals (comments + posts), with at least 1 medal/day.",
        f"- Maintain `{competition_target}` active competitions and submit improvements on each active board.",
        "- Run one notebook optimization sprint on top 5 notebooks (title/intro/results/update notes).",
        "- Publish data dictionaries for all active datasets and one baseline notebook link per dataset.",
        "",
        "## Daily Cadence",
        "",
        "- 2 competition experiments and at least 1 submission when leaderboard gain is positive.",
        "- 5 high-signal discussion comments + 1 focused discussion post.",
        "- 1 notebook refresh block (20-30 minutes) with changelog note.",
        "",
        "## Competition Priority Queue",
        "",
    ]

    if urgent:
        for item in urgent:
            lines.append(
                f"- {item['competition']} ({item['deadline_raw']}, {item['days_to_deadline']}d left): {item.get('strategy', 'No strategy recorded')}"
            )
    elif upcoming:
        for item in upcoming[:3]:
            lines.append(
                f"- {item['competition']} ({item['deadline_raw']}, {item['days_to_deadline']}d left): {item.get('strategy', 'No strategy recorded')}"
            )
    else:
        lines.append("- Add current active competitions in `grandmaster-tracker.md` to generate a prioritized queue.")

    lines.extend(
        [
            "",
            "## KPI Targets",
            "",
            f"- Notebook votes: {notebooks.get('total_votes', 0)} -> {notebooks.get('total_votes', 0) + 10}",
            f"- Dataset votes: {datasets.get('total_votes', 0)} -> {datasets.get('total_votes', 0) + 5}",
            f"- Discussion posts: {discussion.get('total_posts', 0)} -> {discussion.get('total_posts', 0) + 14}",
            f"- Competition entries: {competition_entries} -> {competition_target}",
            "",
        ]
    )

    return "\n".join(lines)


def generate_badge_plan_markdown(snapshot: dict[str, Any]) -> str:
    categories = snapshot["categories"]
    generated_on = snapshot["generated_on"]
    notebook_votes = int(categories["notebooks"].get("total_votes", 0) or 0)
    dataset_votes = int(categories["datasets"].get("total_votes", 0) or 0)
    competition_entries = int(categories["competitions"].get("entered", 0) or 0)
    discussion_posts = int(categories["discussion"].get("total_posts", 0) or 0)

    submission_action = (
        "Start a 7-day submission streak across your already-entered competitions."
        if competition_entries > 0
        else "Enter an easy competition first, then start a 7-day submission streak."
    )

    lines = [
        "# Kaggle Badge Roadmap",
        "",
        f"- Generated: {generated_on}",
        f"- Current live tracker basis: notebooks={categories['notebooks'].get('total_notebooks', 0)}, notebook_votes={notebook_votes}, datasets={categories['datasets'].get('total_datasets', 0)}, dataset_votes={dataset_votes}, competition_entries={competition_entries}, discussion_posts={discussion_posts}",
        "",
        "## Phase 1: Same-Day Wins",
        "",
        "| Badge(s) | Why now | Action |",
        "|---|---|---|",
        "| `Collector`, `Agent of Discord` | Pure UI/account actions with no content dependency. | Create one collection and link your Kaggle account to Discord. |",
        "| `Github Coder`, `Colab Coder`, `Code Forker` | Fast notebook workflow badges. | Import one notebook from GitHub, open one Kaggle notebook in Colab, and fork one public notebook with a meaningful edit. |",
        "",
        "## Phase 2: Fast Publish Badges",
        "",
        "| Badge(s) | Why now | Action |",
        "|---|---|---|",
        "| `R Coder`, `R Markdown Coder` | Single artifact each; no traction required. | Publish one simple R notebook and one R Markdown script. |",
        "| `Utility Scripter`, `Notebook Modeler` | Lightweight publishing tasks that build reusable assets. | Publish one utility script and one notebook that uses a Kaggle model. |",
        "| `Learner` | Reliable progress with no vote dependency. | Complete one Kaggle Learn course. |",
        "",
        "## Phase 3: This Week",
        "",
        "| Badge(s) | Why now | Action |",
        "|---|---|---|",
        f"| `Submission Streak`, `7 Day Login Streak` | Pure consistency; easiest streak layer. | {submission_action} Also log in every day for 7 straight days. |",
        "| `Dataset Pipeline Creator`, `Linked Dataset Creator` | Fits your existing repo workflow and content volume. | Create one dataset from notebook output and one dataset from a URL or GitHub link. |",
        "| `Student` | Straight extension of `Learner`. | Push from 1 completed Kaggle Learn course to 5 total. |",
        "",
        "## Phase 4: This Month",
        "",
        "| Badge(s) | Why now | Action |",
        "|---|---|---|",
        "| `Competitor` | Requires a valid medal-eligible competition submission. | Submit to a current Featured, Community, Research, or Playground competition. |",
        "| `Model Creator`, `Model Variation Creator`, `Model Tagger`, `Linked Model Creator`, `Model Pipeline Creator`, `Competition Modeler` | These chain well once you create the first model artifact. | Create one model, add a second variation, tag it, publish one from notebook output or a link, then use it in a competition notebook. |",
        "| `Graduate`, `30 Day Login Streak`, `Super Submission Streak` | Medium-effort compounding badges. | Extend Learn progress to 10 courses and keep streaks alive for 30 days. |",
        "",
        "## Phase 5: Harder Quality Badges",
        "",
        "| Badge(s) | Why later | Action |",
        "|---|---|---|",
        "| `Dataset Documenter`, `Model Documenter` | These require a perfect usability score, not just publication. | Pick one flagship dataset/model and fully complete metadata, provenance, tags, descriptions, and usage docs until Kaggle shows a perfect rating. |",
        "| `API Model Creator` | Depends on a working upload-capable API path. | Use a full Kaggle API credential set or the Kaggle UI to publish a model through the API workflow. |",
        "",
        "## Phase 6: Seasonal Or Availability-Dependent",
        "",
        "| Badge(s) | Why blocked | Action |",
        "|---|---|---|",
        "| `Research Competitor`, `Playground Competitor`, `Community Competitor` | Depends on live competition inventory. | Watch current competition listings and submit as soon as a qualifying competition is available. |",
        "| `Simulation Competitor`, `Santa Competitor`, `March Mania Competitor`, `Code Submitter` | Product- or season-specific competition formats. | Join the relevant event when it is live and make one qualifying submission. |",
        "",
        "## Phase 7: Pure Time Gates Or Likely Retired",
        "",
        "- `10 Years on Kaggle`, `15 Years on Kaggle`",
        "- `30 Day Login Streak`, `100 Day Login Streak`, `Year Long Login Streak`, `Mega Submission Streak`",
        "- `Completed 5-Day Gen AI Intensive`, `5-Day AI Agents Intensive Course with Google`",
        "- `Stack Overflow Road Safety Challenge Badge`, `Founding Benchmark Task Author`",
        "",
        "## Suggested Next 7 Actions",
        "",
        "- Create a collection.",
        "- Link Discord.",
        "- Import one notebook from GitHub.",
        "- Open one Kaggle notebook in Colab.",
        "- Fork one public notebook and save an edited version.",
        "- Publish one R or R Markdown artifact.",
        f"- {submission_action}",
        "",
    ]

    return "\n".join(lines)


def generate_pace_markdown(snapshots: list[dict[str, Any]]) -> str:
    if not snapshots:
        return "# Kaggle Medal Ops Pace Analysis\n\n- No snapshots found.\n"

    current = snapshots[-1]
    categories = current["categories"]
    first_date = snapshot_generated_date(snapshots[0])
    last_date = snapshot_generated_date(current)
    sample_days = (last_date - first_date).days if first_date and last_date else 0
    has_time_window = sample_days > 0

    lines = [
        "# Kaggle Medal Ops Pace Analysis",
        "",
        f"- Latest snapshot: {current.get('generated_on', 'unknown')}",
        f"- Samples: {len(snapshots)} snapshot(s)",
        f"- Analysis window: {sample_days} day(s)",
        "",
        "## Outcome Pace",
        "",
        "| Metric | Current | Goal | Gap | Velocity | ETA (weeks) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    metric_specs = [
        ("Competitions Gold", ("categories", "competitions", "gold"), ("categories", "competitions", "gold_goal"), ("categories", "competitions", "gold_gap")),
        ("Notebooks Gold", ("categories", "notebooks", "gold"), ("categories", "notebooks", "gold_goal"), ("categories", "notebooks", "gold_gap")),
        ("Datasets Gold", ("categories", "datasets", "gold"), ("categories", "datasets", "gold_goal"), ("categories", "datasets", "gold_gap")),
        ("Discussion Gold", ("categories", "discussion", "gold"), ("categories", "discussion", "gold_goal"), ("categories", "discussion", "gold_gap")),
        ("Discussion Bronze (Expert)", ("categories", "discussion", "bronze"), ("categories", "discussion", "expert_bronze_goal"), ("categories", "discussion", "expert_bronze_gap")),
    ]

    for label, current_path, goal_path, gap_path in metric_specs:
        current_value = nested_int(current, current_path) or 0
        goal_value = nested_int(current, goal_path)
        gap_value = nested_int(current, gap_path)
        velocity = weekly_velocity(snapshots, current_path)
        eta = estimate_eta_weeks(gap_value, velocity)
        lines.append(
            f"| {label} | {current_value} | {goal_value if goal_value is not None else 'n/a'} | "
            f"{gap_value if gap_value is not None else 'n/a'} | {format_velocity(velocity)} | {eta} |"
        )

    lines.extend(
        [
            "",
            "## Leading Indicators",
            "",
            f"- Competition entries velocity: {format_velocity(weekly_velocity(snapshots, ('categories', 'competitions', 'entered')))}",
            f"- Notebook votes velocity: {format_velocity(weekly_velocity(snapshots, ('categories', 'notebooks', 'total_votes')))}",
            f"- Dataset votes velocity: {format_velocity(weekly_velocity(snapshots, ('categories', 'datasets', 'total_votes')))}",
            f"- Discussion posts velocity: {format_velocity(weekly_velocity(snapshots, ('categories', 'discussion', 'total_posts')))}",
            "",
            "## Pace Flags",
            "",
        ]
    )

    if len(snapshots) < 2:
        lines.append("- Need at least 2 snapshots for reliable pace estimates.")
    elif not has_time_window:
        lines.append("- Need snapshots across at least 1 full day for velocity and ETA estimates.")
    else:
        critical_flags: list[str] = []
        for label, current_path, _, gap_path in metric_specs:
            gap_value = nested_int(current, gap_path)
            velocity = weekly_velocity(snapshots, current_path)
            if isinstance(gap_value, int) and gap_value > 0 and (velocity is None or velocity <= 0):
                critical_flags.append(f"- {label} is off pace (gap {gap_value}, velocity {format_velocity(velocity)}).")
        if not critical_flags:
            lines.append("- No negative pace flags detected in tracked outcomes.")
        else:
            lines.extend(critical_flags)

    lines.append("")
    return "\n".join(lines)


def generate_sync_markdown(
    tracker_path: Path,
    today: date,
    live: dict[str, Any],
    changes: dict[str, Any],
    dry_run: bool,
) -> str:
    before = changes["before"]
    after = changes["after"]
    changed_fields = changes["changed_fields"]

    lines = [
        "# Kaggle Medal Ops Sync Report",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Tracker: `{tracker_path}`",
        f"- Mode: {'dry-run' if dry_run else 'write'}",
        "",
        "## Live Pull Summary",
        "",
        f"- Notebooks pulled: {live['notebooks_count']} (vote key: {live.get('notebooks_vote_key') or 'n/a'})",
        f"- Notebook votes pulled: {live['notebooks_total_votes']}",
        f"- Datasets pulled: {live['datasets_count']} (vote key: {live.get('datasets_vote_key') or 'n/a'})",
        f"- Dataset votes pulled: {live['datasets_total_votes']}",
    ]

    if live.get("competitions_entered") is not None:
        lines.append(
            f"- Competition entries pulled: {live['competitions_entered']} (entered key: {live.get('competitions_entered_key')})"
        )
    else:
        lines.append("- Competition entries pulled: n/a (no entered column found in competitions list output)")

    lines.extend(
        [
            "",
            "## Tracker Diff Summary",
            "",
            f"- Changed fields: {', '.join(changed_fields) if changed_fields else 'none'}",
            "",
            "## Key Metric Changes",
            "",
            f"- Notebooks total: {before['notebooks'].get('total_notebooks', 'n/a')} -> {after['notebooks'].get('total_notebooks', 'n/a')}",
            f"- Notebooks votes: {before['notebooks'].get('total_votes', 'n/a')} -> {after['notebooks'].get('total_votes', 'n/a')}",
            f"- Datasets total: {before['datasets'].get('total_datasets', 'n/a')} -> {after['datasets'].get('total_datasets', 'n/a')}",
            f"- Datasets votes: {before['datasets'].get('total_votes', 'n/a')} -> {after['datasets'].get('total_votes', 'n/a')}",
        ]
    )

    if (
        before["competitions"].get("entered") is not None
        or after["competitions"].get("entered") is not None
    ):
        lines.append(
            f"- Competition entries: {before['competitions'].get('entered', 'n/a')} -> {after['competitions'].get('entered', 'n/a')}"
        )

    lines.append("")
    return "\n".join(lines)


def has_kaggle_credentials() -> tuple[bool, list[str]]:
    sources: list[str] = []

    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key = os.environ.get("KAGGLE_KEY", "").strip()
    if env_token:
        sources.append("environment-token")
    if env_user and env_key:
        sources.append("environment")

    candidates = [Path.home() / ".kaggle" / "kaggle.json", Path("kaggle.json")]
    sources.extend(str(path) for path in candidates if path.exists())
    return bool(sources), sources


def run_preflight_checks(
    tracker_path: Path,
    output_root: Path,
    today: date,
    kernels_csv: Path | None,
    datasets_csv: Path | None,
    competitions_csv: Path | None,
    require_kaggle: bool,
    max_stale_days: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []
    snapshot: dict[str, Any] | None = None
    csv_metrics: dict[str, Any] | None = None

    if not tracker_path.exists():
        errors.append(f"Tracker file not found: {tracker_path}")
    else:
        try:
            tracker_content = tracker_path.read_text(encoding="utf-8")
            snapshot = build_snapshot(tracker_content, today)
            infos.append(f"Tracker readable: {tracker_path}")

            requirements = parse_tier_requirements(tracker_content)
            if not requirements:
                errors.append("Tier requirements table could not be parsed from tracker.")

            for heading in ("Competitions", "Notebooks", "Datasets", "Discussion"):
                metrics = parse_progress_metrics(tracker_content, heading)
                if not metrics:
                    errors.append(f"Progress table missing or invalid for `{heading}`.")

            stale_days = snapshot.get("tracker_stale_days")
            if isinstance(stale_days, int):
                if stale_days > max_stale_days:
                    warnings.append(
                        f"Tracker is stale by {stale_days} day(s) (threshold: {max_stale_days})."
                    )
                else:
                    infos.append(f"Tracker freshness OK ({stale_days} day(s)).")
        except OSError as exc:
            errors.append(f"Failed to read tracker: {exc}")
        except Exception as exc:  # defensive parsing guard
            errors.append(f"Failed to parse tracker: {exc}")

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe_path = output_root / ".doctor-write-probe"
        probe_path.write_text("ok\n", encoding="utf-8")
        probe_path.unlink()
        infos.append(f"Output root writable: {output_root}")
    except OSError as exc:
        errors.append(f"Output root is not writable (`{output_root}`): {exc}")

    offline_mode = bool(kernels_csv or datasets_csv or competitions_csv)
    kaggle_cli_available = has_kaggle_cli()
    if kaggle_cli_available:
        infos.append("kaggle CLI available.")
    else:
        if require_kaggle:
            errors.append("kaggle CLI not found.")
        elif offline_mode:
            infos.append("kaggle CLI not found; offline CSV sync mode selected.")
        else:
            warnings.append("kaggle CLI not found; live sync is unavailable.")

    creds_ok, creds_paths = has_kaggle_credentials()
    if creds_ok:
        joined = ", ".join(str(path) for path in creds_paths)
        infos.append(f"Kaggle credentials found: {joined}")
    else:
        if require_kaggle:
            errors.append("Kaggle credentials not found (`~/.kaggle/kaggle.json` or local `kaggle.json`).")
        elif offline_mode:
            infos.append("Kaggle credentials not found; offline CSV sync mode selected.")
        else:
            warnings.append("Kaggle credentials not found for live sync.")

    if offline_mode:
        if not kernels_csv or not datasets_csv:
            errors.append("CSV preflight requires both --kernels-csv and --datasets-csv when any CSV is provided.")
        else:
            try:
                csv_metrics = fetch_metrics_from_csv(kernels_csv, datasets_csv, competitions_csv)
                infos.append(
                    "CSV sync inputs validated: "
                    f"notebooks={csv_metrics['notebooks_count']}, "
                    f"datasets={csv_metrics['datasets_count']}, "
                    f"competitions_entered={csv_metrics.get('competitions_entered', 'n/a')}"
                )
                if not competitions_csv:
                    warnings.append(
                        "No competitions CSV provided; `Competitions.Entered` will not be updated during sync."
                    )
            except SystemExit as exc:
                errors.append(str(exc))
    elif not kaggle_cli_available or not creds_ok:
        warnings.append(
            "No fully configured live sync path detected. Run `./manage.sh sync-template` and use CSV sync."
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "snapshot": snapshot,
        "csv_metrics": csv_metrics,
        "kaggle_cli_available": kaggle_cli_available,
        "kaggle_credentials_available": creds_ok,
    }


def generate_doctor_markdown(
    tracker_path: Path,
    output_root: Path,
    today: date,
    checks: dict[str, Any],
    strict: bool,
    max_stale_days: int,
) -> str:
    errors = checks["errors"]
    warnings = checks["warnings"]
    infos = checks["infos"]

    if errors:
        status = "BLOCKED"
    elif warnings:
        status = "ATTENTION"
    else:
        status = "READY"

    strict_line = "enabled" if strict else "disabled"

    lines = [
        "# Kaggle Medal Ops Preflight Report",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Tracker: `{tracker_path}`",
        f"- Output root: `{output_root}`",
        f"- Status: {status}",
        f"- Strict mode: {strict_line}",
        f"- Max stale days: {max_stale_days}",
        "",
        "## Summary",
        "",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        f"- Info checks: {len(infos)}",
        "",
    ]

    lines.append("## Blocking Issues")
    lines.append("")
    if errors:
        lines.extend(f"- {item}" for item in errors)
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Passing Checks")
    lines.append("")
    if infos:
        lines.extend(f"- {item}" for item in infos)
    else:
        lines.append("- none")
    lines.append("")

    recommended: list[str] = []
    if any("Tracker file not found" in item for item in errors):
        recommended.append(f"Verify tracker path: `--tracker {tracker_path}`")
    if any("missing a vote column" in item or "missing an entered column" in item for item in errors):
        recommended.append("Regenerate CSVs using `./manage.sh sync-template` and the export script.")
    if any("kaggle CLI not found" in item for item in warnings + errors):
        recommended.append("Install/authenticate Kaggle CLI, or continue with CSV sync.")
    if any("credentials not found" in item.lower() for item in warnings + errors):
        recommended.append("Add Kaggle credentials to `~/.kaggle/kaggle.json` (chmod 600).")

    if not recommended and status == "READY":
        recommended.append("Preflight passed. Run `./manage.sh sync --dry-run`.")
    elif not recommended:
        recommended.append("Resolve listed issues, then rerun `./manage.sh doctor --strict`.")

    lines.append("## Recommended Next Step")
    lines.append("")
    lines.extend(f"- {item}" for item in recommended)
    lines.append("")

    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def add_shared_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER_PATH), help="Path to grandmaster tracker markdown file.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for history and reports.")
    parser.add_argument("--today", default=None, help="Override date in YYYY-MM-DD format.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle medal operations CLI.")
    add_shared_cli_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    scorecard_parser = subparsers.add_parser("scorecard", help="Generate scorecard report and save snapshot.")
    add_shared_cli_args(scorecard_parser)
    badge_plan_parser = subparsers.add_parser("badge-plan", help="Generate ordered Kaggle badge roadmap.")
    add_shared_cli_args(badge_plan_parser)
    weekly_parser = subparsers.add_parser("weekly-plan", help="Generate weekly execution plan.")
    add_shared_cli_args(weekly_parser)
    pace_parser = subparsers.add_parser("pace", help="Generate progress velocity and ETA analysis.")
    add_shared_cli_args(pace_parser)
    sync_parser = subparsers.add_parser("sync", help="Sync tracker metrics from live Kaggle CLI data.")
    add_shared_cli_args(sync_parser)
    sync_parser.add_argument("--dry-run", action="store_true", help="Generate sync report without writing tracker changes.")
    sync_parser.add_argument("--kernels-csv", default=None, help="Path to exported kernels CSV.")
    sync_parser.add_argument("--datasets-csv", default=None, help="Path to exported datasets CSV.")
    sync_parser.add_argument(
        "--competitions-csv",
        default=None,
        help="Path to exported competitions CSV (optional, used for 'Entered' metric).",
    )
    template_parser = subparsers.add_parser(
        "sync-template", help="Generate CSV templates and helper script for offline sync."
    )
    add_shared_cli_args(template_parser)
    template_parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Directory for generated template files (default: <output-root>/{DEFAULT_SYNC_INPUT_DIRNAME}).",
    )
    template_parser.add_argument("--force", action="store_true", help="Overwrite existing template files.")
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run preflight checks for tracker health, environment readiness, and sync inputs.",
    )
    add_shared_cli_args(doctor_parser)
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when warnings are present.",
    )
    doctor_parser.add_argument(
        "--require-kaggle",
        action="store_true",
        help="Require kaggle CLI and credentials for a passing check.",
    )
    doctor_parser.add_argument(
        "--max-stale-days",
        type=int,
        default=7,
        help="Warn when tracker staleness exceeds this threshold (default: 7).",
    )
    doctor_parser.add_argument("--kernels-csv", default=None, help="Path to exported kernels CSV.")
    doctor_parser.add_argument("--datasets-csv", default=None, help="Path to exported datasets CSV.")
    doctor_parser.add_argument(
        "--competitions-csv",
        default=None,
        help="Path to exported competitions CSV (optional).",
    )
    return parser.parse_args()




def main() -> int:
    args = parse_args()
    today = resolve_today(args.today)
    output_root = Path(args.output_root)
    tracker_path = Path(args.tracker)
    history_dir = output_root / "history"
    reports_dir = output_root / "reports"

    if args.command == "sync-template":
        out_dir = Path(args.out_dir) if args.out_dir else output_root / DEFAULT_SYNC_INPUT_DIRNAME
        statuses = generate_sync_template_assets(out_dir, force=args.force)
        print(f"Sync templates directory: {out_dir}")
        for path, status in statuses.items():
            print(f"- {status}: {path}")
        print("Next step:")
        print(f"  {out_dir / 'export_kaggle_sync_csv.sh'}")
        return 0

    if args.command == "doctor":
        if int(args.max_stale_days) < 0:
            raise SystemExit("--max-stale-days must be >= 0")
        checks = run_preflight_checks(
            tracker_path=tracker_path,
            output_root=output_root,
            today=today,
            kernels_csv=Path(args.kernels_csv) if args.kernels_csv else None,
            datasets_csv=Path(args.datasets_csv) if args.datasets_csv else None,
            competitions_csv=Path(args.competitions_csv) if args.competitions_csv else None,
            require_kaggle=bool(args.require_kaggle),
            max_stale_days=int(args.max_stale_days),
        )

        report = generate_doctor_markdown(
            tracker_path=tracker_path,
            output_root=output_root,
            today=today,
            checks=checks,
            strict=bool(args.strict),
            max_stale_days=int(args.max_stale_days),
        )
        dated_report_path = reports_dir / f"doctor-{today.isoformat()}.md"
        latest_report_path = reports_dir / "latest-doctor.md"
        write_report(dated_report_path, report)
        write_report(latest_report_path, report)

        errors = checks["errors"]
        warnings = checks["warnings"]
        print(f"Doctor report written: {dated_report_path}")
        print(f"Latest doctor report: {latest_report_path}")
        print(f"Summary: {len(errors)} error(s), {len(warnings)} warning(s)")

        if errors:
            print("Preflight status: BLOCKED")
            return 1
        if args.strict and warnings:
            print("Preflight status: ATTENTION (strict mode failure)")
            return 1

        if warnings:
            print("Preflight status: ATTENTION")
        else:
            print("Preflight status: READY")
        return 0

    if not tracker_path.exists():
        raise SystemExit(f"Tracker file not found: {tracker_path}")

    content = tracker_path.read_text(encoding="utf-8")
    snapshot = build_snapshot(content, today)

    if args.command == "scorecard":
        previous = load_latest_snapshot(history_dir)
        snapshot_path = write_snapshot(history_dir, snapshot)
        report = generate_scorecard_markdown(snapshot, previous)
        dated_report_path = reports_dir / f"scorecard-{today.isoformat()}.md"
        latest_report_path = reports_dir / "latest-scorecard.md"
        write_report(dated_report_path, report)
        write_report(latest_report_path, report)
        print(f"Snapshot written: {snapshot_path}")
        print(f"Scorecard written: {dated_report_path}")
        print(f"Latest scorecard: {latest_report_path}")
        return 0

    if args.command == "badge-plan":
        report = generate_badge_plan_markdown(snapshot)
        dated_report_path = reports_dir / f"badge-plan-{today.isoformat()}.md"
        latest_report_path = reports_dir / "latest-badge-plan.md"
        write_report(dated_report_path, report)
        write_report(latest_report_path, report)
        print(f"Badge roadmap written: {dated_report_path}")
        print(f"Latest badge roadmap: {latest_report_path}")
        return 0

    if args.command == "weekly-plan":
        latest_snapshot = load_latest_snapshot(history_dir) or snapshot
        report = generate_weekly_plan_markdown(latest_snapshot)
        dated_report_path = reports_dir / f"weekly-plan-{today.isoformat()}.md"
        latest_report_path = reports_dir / "latest-weekly-plan.md"
        write_report(dated_report_path, report)
        write_report(latest_report_path, report)
        print(f"Weekly plan written: {dated_report_path}")
        print(f"Latest weekly plan: {latest_report_path}")
        return 0

    if args.command == "pace":
        snapshot_path = write_snapshot(history_dir, snapshot)
        snapshots = load_all_snapshots(history_dir)
        report = generate_pace_markdown(snapshots)
        dated_report_path = reports_dir / f"pace-{today.isoformat()}.md"
        latest_report_path = reports_dir / "latest-pace.md"
        write_report(dated_report_path, report)
        write_report(latest_report_path, report)
        print(f"Snapshot written: {snapshot_path}")
        print(f"Pace report written: {dated_report_path}")
        print(f"Latest pace report: {latest_report_path}")
        return 0

    if args.command == "sync":
        if args.kernels_csv or args.datasets_csv or args.competitions_csv:
            if not args.kernels_csv or not args.datasets_csv:
                raise SystemExit("CSV sync requires both --kernels-csv and --datasets-csv.")
            live = fetch_metrics_from_csv(
                kernels_csv=Path(args.kernels_csv),
                datasets_csv=Path(args.datasets_csv),
                competitions_csv=Path(args.competitions_csv) if args.competitions_csv else None,
            )
        else:
            live = fetch_live_kaggle_metrics()
        original_content = tracker_path.read_text(encoding="utf-8")
        updated_content, changes = apply_tracker_sync(original_content, today, live)

        if not args.dry_run and updated_content != original_content:
            tracker_path.write_text(updated_content, encoding="utf-8")

        report = generate_sync_markdown(tracker_path, today, live, changes, args.dry_run)
        dated_report_path = reports_dir / f"sync-{today.isoformat()}.md"
        latest_report_path = reports_dir / "latest-sync.md"
        write_report(dated_report_path, report)
        write_report(latest_report_path, report)

        if args.dry_run:
            print("Dry-run mode: tracker file was not modified.")
        elif updated_content != original_content:
            print(f"Tracker updated: {tracker_path}")
        else:
            print("Tracker already up to date with pulled metrics.")
        print(f"Sync report written: {dated_report_path}")
        print(f"Latest sync report: {latest_report_path}")
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
