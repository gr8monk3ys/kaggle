#!/usr/bin/env python3
"""The Tracker: its markdown format, in one module.

``docs/reports/grandmaster-tracker.md`` is the hand-maintained record of where
the portfolio stands against the Grandmaster goal, and the source of truth when
it and live Kaggle counts disagree. Reading and writing that format used to live
inside ``ops/medal_ops.py`` — 15 of its 53 functions — alongside report
generation, snapshot building, doctor checks and an eight-subcommand CLI.

Parsing and mutation live together on purpose. Splitting read from write is what
lets a format drift: a module that can parse but not write the same markdown is
half a concept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from kaggle_portfolio.shared.clock import parse_iso_date


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


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "").strip().lower())


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


def apply_tracker_sync(
    content: str, today: date, live: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
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
        updated,
        "Notebooks",
        "Total notebooks",
        f"{live['notebooks_count']} (on Kaggle)",
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
            updated,
            "Notebooks",
            "Silver medals (20+ votes)",
            str(live["notebooks_silver"]),
        )
        if changed:
            changed_fields.append("Notebooks.Silver medals")

    if isinstance(live.get("notebooks_bronze"), int):
        updated, changed = update_progress_current_cell(
            updated,
            "Notebooks",
            "Bronze medals (5+ votes)",
            str(live["notebooks_bronze"]),
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
            updated,
            "Datasets",
            "Total downloads",
            str(live["datasets_total_downloads"]),
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
            updated,
            "Datasets",
            "Silver medals (20+ votes)",
            str(live["datasets_silver"]),
        )
        if changed:
            changed_fields.append("Datasets.Silver medals")

    if isinstance(live.get("datasets_bronze"), int):
        updated, changed = update_progress_current_cell(
            updated,
            "Datasets",
            "Bronze medals (5+ votes)",
            str(live["datasets_bronze"]),
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
