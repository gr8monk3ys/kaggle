#!/usr/bin/env python3
"""Generate a multi-channel promotion campaign pack for dataset growth."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from kaggle_portfolio.shared.kaggle_utils import parse_iso_date, resolve_today

DEFAULT_OUTPUT_ROOT = Path("medal_ops")
DEFAULT_DATASET_REPORT = DEFAULT_OUTPUT_ROOT / "reports" / "latest-dataset-usability.json"
DEFAULT_QUEUE_PATH = Path("pi-automation") / "data" / "promotion_campaign_queue.json"
DEFAULT_CHANNELS = ["kaggle-discussion", "kaggle-changelog", "x", "linkedin"]


def normalize_ref(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_ref_filter(raw: str | None) -> set[str]:
    if not raw:
        return set()
    refs = {
        normalize_ref(item)
        for part in raw.split(",")
        for item in [part.strip()]
        if item.strip()
    }
    return {item for item in refs if item}


def filter_rows_by_refs(rows: list[dict[str, Any]], refs: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    if not refs:
        return rows, []

    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        ref = normalize_ref(row.get("dataset_ref"))
        if ref in refs:
            filtered.append(row)
            seen.add(ref)

    missing = sorted(ref for ref in refs if ref not in seen)
    warnings = [f"Requested dataset ref not found in report: {ref}" for ref in missing]
    return filtered, warnings




def resolve_start_date(today: date, start_override: str | None) -> date:
    if not start_override:
        return today
    parsed = parse_iso_date(start_override)
    if not parsed:
        raise SystemExit(f"Invalid --start-date value: {start_override}")
    return parsed


def load_dataset_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Dataset report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("datasets")
    if not isinstance(rows, list):
        raise SystemExit(f"Dataset report missing 'datasets' array: {path}")
    return [row for row in rows if isinstance(row, dict)]


def rating_status(rating: float | None, alert_under: float, target_rating: float) -> str:
    if rating is None:
        return "unknown"
    if rating < alert_under:
        return "critical"
    if rating < target_rating:
        return "watch"
    return "strong"


def dataset_url(dataset_ref: str | None) -> str | None:
    if not dataset_ref:
        return None
    if "/" not in dataset_ref:
        return None
    return f"https://www.kaggle.com/datasets/{dataset_ref}"


def prioritize_datasets(
    rows: list[dict[str, Any]],
    *,
    alert_under: float,
    target_rating: float,
    max_datasets: int,
) -> list[dict[str, Any]]:
    prioritized: list[dict[str, Any]] = []
    for row in rows:
        rating_raw = row.get("kaggle_usability_rating")
        rating = float(rating_raw) if isinstance(rating_raw, (int, float)) else None
        status = rating_status(rating, alert_under=alert_under, target_rating=target_rating)
        ref = str(row.get("dataset_ref") or row.get("path") or "").strip()
        title = str(row.get("title") or ref or "Dataset").strip()
        gap_to_target = round(max(0.0, target_rating - rating), 4) if rating is not None else None
        gap_to_one = round(max(0.0, 1.0 - rating), 4) if rating is not None else None

        if status == "critical":
            objective = "Recover from critical usability and push above 0.8."
        elif status == "watch":
            objective = "Cross 0.8 usability with promotion + quick dataset refresh."
        elif status == "strong":
            objective = "Keep momentum and optimize toward 1.0 usability."
        else:
            objective = "Resolve live coverage: verify listing/public state and rerun tracker."

        prioritized.append(
            {
                "dataset_ref": ref,
                "title": title,
                "path": str(row.get("path", "")),
                "rating": rating,
                "status": status,
                "gap_to_target": gap_to_target,
                "gap_to_one": gap_to_one,
                "dataset_url": dataset_url(ref),
                "objective": objective,
            }
        )

    rank = {"critical": 0, "watch": 1, "strong": 2, "unknown": 3}
    prioritized.sort(
        key=lambda item: (
            rank.get(str(item.get("status")), 9),
            float(item["rating"]) if isinstance(item.get("rating"), float) else 9.0,
            str(item.get("dataset_ref")),
        )
    )
    return prioritized[:max_datasets]


def build_channel_copy(dataset: dict[str, Any], target_rating: float) -> dict[str, str]:
    ref = str(dataset.get("dataset_ref", "dataset"))
    title = str(dataset.get("title", ref))
    url = dataset.get("dataset_url") or "(add dataset URL)"
    rating = dataset.get("rating")
    rating_text = f"{rating:.3f}" if isinstance(rating, float) else "n/a"
    objective = str(dataset.get("objective") or "Improve docs, examples, and discoverability.")

    kaggle_discussion = (
        f"I am planning the next refresh for {title} and want concrete feedback before I publish it.\n"
        f"Dataset: {url}\n"
        f"Current usability rating: {rating_text} (target {target_rating:.1f}).\n"
        f"Current focus: {objective}\n"
        "If you used this dataset, what is the one improvement that would make it more useful?"
    )
    kaggle_changelog = (
        f"Refresh plan for {title}\n"
        f"- Dataset: {url}\n"
        f"- Current usability rating: {rating_text}\n"
        f"- Short-term target: {target_rating:.1f}\n"
        f"- Next change set: {objective}\n"
        "- Planned additions: richer data dictionary, example workflows, and clearer file notes."
    )
    x_copy = (
        f"Dataset refresh: {title} ({ref}) now in active usability sprint. "
        f"Current {rating_text}, pushing to {target_rating:.1f}+ and then 1.0. {url}"
    )
    linkedin_copy = (
        f"I am running a usability improvement campaign for {title} ({ref}). "
        f"Current rating: {rating_text}. Goal: {target_rating:.1f}+ in the short term, then 1.0. "
        f"Dataset link: {url}"
    )

    return {
        "kaggle-discussion": kaggle_discussion,
        "kaggle-changelog": kaggle_changelog,
        "x": x_copy,
        "linkedin": linkedin_copy,
    }


def build_campaign_actions(
    datasets: list[dict[str, Any]],
    *,
    channels: list[str],
    target_rating: float,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for dataset in datasets:
        copy_map = build_channel_copy(dataset, target_rating=target_rating)
        for channel in channels:
            text = copy_map.get(channel)
            if not text:
                continue
            actions.append(
                {
                    "dataset_ref": dataset.get("dataset_ref"),
                    "dataset_title": dataset.get("title"),
                    "dataset_status": dataset.get("status"),
                    "dataset_rating": dataset.get("rating"),
                    "channel": channel,
                    "objective": dataset.get("objective"),
                    "copy": text,
                }
            )
    return actions


def schedule_campaign(
    actions: list[dict[str, Any]],
    *,
    start_date: date,
    days: int,
    posts_per_day: int,
) -> list[dict[str, Any]]:
    if not actions:
        return []
    total_slots = days * posts_per_day
    queue: list[dict[str, Any]] = []
    for idx in range(total_slots):
        action = dict(actions[idx % len(actions)])
        day_offset = idx // posts_per_day
        slot_index = idx % posts_per_day
        post_date = start_date + timedelta(days=day_offset)
        post_time = time(hour=14 + min(slot_index, 3), minute=0, tzinfo=timezone.utc)
        scheduled = datetime.combine(post_date, post_time)
        action["id"] = f"campaign_{idx + 1:03d}"
        action["scheduled_for"] = scheduled.isoformat().replace("+00:00", "Z")
        action["status"] = "planned"
        queue.append(action)
    return queue


def generate_markdown(
    *,
    today: date,
    report_path: Path,
    datasets: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    alert_under: float,
    target_rating: float,
) -> str:
    ratings = [item["rating"] for item in datasets if isinstance(item.get("rating"), float)]
    lines = [
        "# Dataset Promotion Campaign Pack",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Dataset report source: `{report_path}`",
        f"- Alert threshold: < {alert_under:.3f}",
        f"- Target threshold: >= {target_rating:.3f}",
        f"- Prioritized datasets: {len(datasets)}",
        f"- Scheduled actions: {len(queue)}",
    ]
    if ratings:
        lines.append(f"- Average live rating (priority set): {statistics.mean(ratings):.3f}")
    else:
        lines.append("- Average live rating (priority set): n/a")

    lines.extend(
        [
            "",
            "## Priority Datasets",
            "",
            f"| Rank | Dataset | Live rating | Status | Gap to {target_rating:.1f} | Gap to 1.0 | Objective |",
            "|---:|---|---:|---|---:|---:|---|",
        ]
    )
    for idx, item in enumerate(datasets, start=1):
        rating = item.get("rating")
        rating_text = f"{rating:.3f}" if isinstance(rating, float) else "n/a"
        gap_to_target = item.get("gap_to_target")
        gap_to_one = item.get("gap_to_one")
        gap_to_target_text = f"{gap_to_target:.3f}" if isinstance(gap_to_target, float) else "n/a"
        gap_to_one_text = f"{gap_to_one:.3f}" if isinstance(gap_to_one, float) else "n/a"
        lines.append(
            f"| {idx} | `{item.get('dataset_ref')}` | {rating_text} | {item.get('status')} | "
            f"{gap_to_target_text} | {gap_to_one_text} | {item.get('objective')} |"
        )

    lines.extend(
        [
            "",
            "## Cadence Queue",
            "",
            "| Slot | Scheduled (UTC) | Channel | Dataset | Objective |",
            "|---:|---|---|---|---|",
        ]
    )
    for idx, item in enumerate(queue, start=1):
        lines.append(
            f"| {idx} | {item.get('scheduled_for')} | {item.get('channel')} | "
            f"`{item.get('dataset_ref')}` | {item.get('objective')} |"
        )

    lines.extend(["", "## Copy Library", ""])
    seen: set[tuple[str, str]] = set()
    for item in queue:
        key = (str(item.get("dataset_ref")), str(item.get("channel")))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"### {item.get('dataset_ref')} - {item.get('channel')}")
        lines.append("")
        lines.append(item.get("copy", ""))
        lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset promotion campaign pack + queue.")
    parser.add_argument("--dataset-report", default=str(DEFAULT_DATASET_REPORT), help="Path to dataset usability JSON report.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for generated reports.")
    parser.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH), help="Where to write JSON campaign queue.")
    parser.add_argument("--today", default=None, help="Override today date (YYYY-MM-DD).")
    parser.add_argument("--start-date", default=None, help="Campaign start date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--days", type=int, default=14, help="Campaign duration in days (default 14).")
    parser.add_argument("--posts-per-day", type=int, default=2, help="Number of scheduled actions per day (default 2).")
    parser.add_argument("--alert-under", type=float, default=0.7, help="Critical threshold (default 0.7).")
    parser.add_argument("--target-rating", type=float, default=0.8, help="Target threshold (default 0.8).")
    parser.add_argument("--max-datasets", type=int, default=12, help="Maximum datasets to include in priority set.")
    parser.add_argument(
        "--refs",
        default=None,
        help="Optional comma-separated dataset refs to include exactly (owner/slug).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.days < 1:
        raise SystemExit("--days must be >= 1")
    if args.posts_per_day < 1:
        raise SystemExit("--posts-per-day must be >= 1")
    if args.max_datasets < 1:
        raise SystemExit("--max-datasets must be >= 1")
    if args.alert_under < 0.0 or args.alert_under > 1.0:
        raise SystemExit("--alert-under must be between 0.0 and 1.0")
    if args.target_rating < 0.0 or args.target_rating > 1.0:
        raise SystemExit("--target-rating must be between 0.0 and 1.0")
    if args.alert_under > args.target_rating:
        raise SystemExit("--alert-under cannot be greater than --target-rating")

    today = resolve_today(args.today)
    start_date = resolve_start_date(today, args.start_date)
    output_root = Path(args.output_root)
    report_path = Path(args.dataset_report)
    queue_path = Path(args.queue_path)

    dataset_rows = load_dataset_report(report_path)
    requested_refs = parse_ref_filter(args.refs)
    dataset_rows, ref_warnings = filter_rows_by_refs(dataset_rows, requested_refs)
    prioritized = prioritize_datasets(
        dataset_rows,
        alert_under=args.alert_under,
        target_rating=args.target_rating,
        max_datasets=args.max_datasets,
    )
    actions = build_campaign_actions(
        prioritized,
        channels=DEFAULT_CHANNELS,
        target_rating=args.target_rating,
    )
    queue = schedule_campaign(
        actions,
        start_date=start_date,
        days=args.days,
        posts_per_day=args.posts_per_day,
    )

    markdown = generate_markdown(
        today=today,
        report_path=report_path,
        datasets=prioritized,
        queue=queue,
        alert_under=args.alert_under,
        target_rating=args.target_rating,
    )
    payload = {
        "generated_on": today.isoformat(),
        "source_report": str(report_path),
        "thresholds": {
            "alert_under": args.alert_under,
            "target_rating": args.target_rating,
        },
        "ref_filter": sorted(requested_refs),
        "warnings": ref_warnings,
        "schedule": {
            "start_date": start_date.isoformat(),
            "days": args.days,
            "posts_per_day": args.posts_per_day,
            "channels": DEFAULT_CHANNELS,
        },
        "prioritized_datasets": prioritized,
        "queue": queue,
    }

    reports_dir = output_root / "reports"
    dated_md = reports_dir / f"promotion-campaign-{today.isoformat()}.md"
    latest_md = reports_dir / "latest-promotion-campaign.md"
    dated_json = reports_dir / f"promotion-campaign-{today.isoformat()}.json"
    latest_json = reports_dir / "latest-promotion-campaign.json"

    write_text(dated_md, markdown)
    write_text(latest_md, markdown)
    write_json(dated_json, payload)
    write_json(latest_json, payload)
    write_json(queue_path, {"generated_on": today.isoformat(), "queue": queue})

    print(f"Campaign pack written: {dated_md}")
    print(f"Latest campaign pack: {latest_md}")
    print(f"Campaign queue written: {queue_path}")
    for warning in ref_warnings:
        print(f"Warning: {warning}")
    print(f"Summary: {len(prioritized)} datasets, {len(queue)} scheduled actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
