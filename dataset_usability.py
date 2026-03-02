#!/usr/bin/env python3
"""Score dataset usability and generate actionable reports."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from kaggle_utils import kaggle_command, parse_iso_date, resolve_today, summarize_subprocess_error


DEFAULT_OUTPUT_ROOT = Path("medal_ops")


@dataclass(frozen=True)
class DatasetScore:
    path: str
    title: str
    score: int
    score_10: int
    tier: str
    criteria: dict[str, int]
    issues: list[str]
    data_files: int
    dataset_ref: str | None = None
    kaggle_usability_rating: float | None = None
    kaggle_score_10: float | None = None




def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def tier_for_score(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Needs Work"
    return "At Risk"


def score_out_of_10(score: int) -> int:
    # Ceil to a 10-point ops score: 91-100 => 10, 81-90 => 9, etc.
    return max(0, min(10, (score + 9) // 10))


def kaggle_rating_to_10(rating: float) -> float:
    return max(0.0, min(10.0, round(rating * 10.0, 1)))




def parse_kaggle_datasets_csv(raw_csv: str) -> dict[str, float]:
    lines = raw_csv.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("ref,"):
            start_idx = idx
            break
    if start_idx is None:
        return {}

    ratings: dict[str, float] = {}
    csv_payload = "\n".join(lines[start_idx:]) + "\n"
    reader = csv.DictReader(io.StringIO(csv_payload))
    for row in reader:
        ref = str(row.get("ref", "")).strip().lower()
        if not ref:
            continue
        try:
            rating = float(str(row.get("usabilityRating", "")).strip())
        except ValueError:
            continue
        ratings[ref] = rating
    return ratings


def load_live_ratings_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        raise SystemExit(f"Live ratings CSV not found: {path}")
    return parse_kaggle_datasets_csv(path.read_text(encoding="utf-8"))


def write_live_ratings_csv(path: Path, ratings: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ref", "usabilityRating"])
        writer.writeheader()
        for ref in sorted(ratings):
            writer.writerow(
                {
                    "ref": ref,
                    "usabilityRating": f"{ratings[ref]:.6f}".rstrip("0").rstrip("."),
                }
            )


def infer_owner_from_scores(scores: list[DatasetScore]) -> str | None:
    owners: dict[str, int] = {}
    for item in scores:
        ref = (item.dataset_ref or "").strip()
        if "/" not in ref:
            continue
        owner = ref.split("/", 1)[0].strip().lower()
        if owner:
            owners[owner] = owners.get(owner, 0) + 1
    if not owners:
        return None
    return max(owners.items(), key=lambda row: row[1])[0]


def _run_kaggle_dataset_list(args: list[str]) -> tuple[dict[str, float] | None, str | None]:
    result = subprocess.run([*kaggle_command(), "datasets", "list", *args], capture_output=True, text=True)
    if result.returncode != 0:
        return None, summarize_subprocess_error(result.stdout, result.stderr)
    return parse_kaggle_datasets_csv(result.stdout), None


def fetch_kaggle_live_ratings(owner: str) -> tuple[dict[str, float], str | None]:
    owner = owner.strip().lower()
    if not owner:
        return {}, "owner is required"

    ratings: dict[str, float] = {}
    errors: list[str] = []

    mine_ratings, mine_err = _run_kaggle_dataset_list(["--mine", "--csv"])
    if mine_ratings is not None:
        ratings.update({ref: rating for ref, rating in mine_ratings.items() if ref.startswith(f"{owner}/")})
    elif mine_err:
        errors.append(f"--mine: {mine_err}")

    search_ratings, search_err = _run_kaggle_dataset_list(["-s", owner, "--csv"])
    if search_ratings is not None:
        ratings.update({ref: rating for ref, rating in search_ratings.items() if ref.startswith(f"{owner}/")})
    elif search_err:
        errors.append(f"-s {owner}: {search_err}")

    if ratings:
        return ratings, None
    if errors:
        return {}, "; ".join(errors)
    return {}, "no live ratings returned"


def attach_kaggle_live_ratings(
    scores: list[DatasetScore],
    ratings_by_ref: dict[str, float],
) -> list[DatasetScore]:
    updated: list[DatasetScore] = []
    for item in scores:
        ref = (item.dataset_ref or "").strip().lower()
        rating = ratings_by_ref.get(ref)
        updated.append(
            replace(
                item,
                kaggle_usability_rating=rating,
                kaggle_score_10=(kaggle_rating_to_10(rating) if rating is not None else None),
            )
        )
    return updated


def live_status(rating: float | None, alert_under: float, target_rating: float) -> str:
    if rating is None:
        return "unknown"
    if rating < alert_under:
        return "critical"
    if rating < target_rating:
        return "watch"
    return "strong"


def build_live_priority_queue(
    scores: list[DatasetScore],
    *,
    alert_under: float,
    target_rating: float,
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for item in scores:
        rating = item.kaggle_usability_rating
        status = live_status(rating, alert_under=alert_under, target_rating=target_rating)
        if status == "critical":
            action = (
                "Critical sprint: refresh metadata and README, push one dataset explorer update, "
                "and run promotion within 48h."
            )
        elif status == "watch":
            action = (
                "Boost sprint: publish one changelog update and one cross-channel promotion to break 0.8."
            )
        elif status == "strong":
            action = "Scale momentum: keep promotion cadence and optimize toward 1.0."
        else:
            action = "Missing live rating: verify dataset is public and listed, then rerun with --live."

        queue.append(
            {
                "path": item.path,
                "title": item.title,
                "dataset_ref": item.dataset_ref,
                "rating": rating,
                "status": status,
                "gap_to_target": (
                    round(max(0.0, target_rating - rating), 4) if rating is not None else None
                ),
                "gap_to_one": (round(max(0.0, 1.0 - rating), 4) if rating is not None else None),
                "action": action,
            }
        )

    status_rank = {"critical": 0, "watch": 1, "strong": 2, "unknown": 3}
    queue.sort(
        key=lambda row: (
            status_rank.get(str(row["status"]), 9),
            float(row["rating"]) if isinstance(row.get("rating"), float) else 9.0,
            str(row["path"]),
        )
    )
    return queue


def summarize_live_queue(queue: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("status", "unknown")) for item in queue)
    return {
        "critical": int(counts.get("critical", 0)),
        "watch": int(counts.get("watch", 0)),
        "strong": int(counts.get("strong", 0)),
        "unknown": int(counts.get("unknown", 0)),
    }


def generate_live_tracker_markdown(
    scores: list[DatasetScore],
    queue: list[dict[str, Any]],
    *,
    today: date,
    alert_under: float,
    target_rating: float,
) -> str:
    live_values = [item.kaggle_usability_rating for item in scores if item.kaggle_usability_rating is not None]
    summary = summarize_live_queue(queue)
    lines = [
        "# Dataset Usability Daily Tracker",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Alert threshold: < {alert_under:.3f}",
        f"- Target threshold: >= {target_rating:.3f}",
        f"- Datasets tracked: {len(scores)}",
        f"- Live ratings matched: {len(live_values)}",
        (
            f"- Average live rating: {statistics.mean(live_values):.3f}"
            if live_values
            else "- Average live rating: n/a"
        ),
        f"- Critical (<{alert_under:.3f}): {summary['critical']}",
        f"- Watch ({alert_under:.3f}-{target_rating:.3f}): {summary['watch']}",
        f"- Strong (>={target_rating:.3f}): {summary['strong']}",
        "",
        "## Priority Queue",
        "",
        f"| Rank | Dataset | Live rating | Gap to {target_rating:.1f} | Gap to 1.0 | Status | Recommended action |",
        "|---:|---|---:|---:|---:|---|---|",
    ]

    for idx, item in enumerate(queue, start=1):
        rating = item.get("rating")
        rating_text = f"{rating:.3f}" if isinstance(rating, float) else "n/a"
        gap_target = item.get("gap_to_target")
        gap_one = item.get("gap_to_one")
        gap_target_text = f"{gap_target:.3f}" if isinstance(gap_target, float) else "n/a"
        gap_one_text = f"{gap_one:.3f}" if isinstance(gap_one, float) else "n/a"
        dataset_label = f"`{item.get('path')}`"
        if item.get("dataset_ref"):
            dataset_label = f"`{item.get('dataset_ref')}`"
        lines.append(
            f"| {idx} | {dataset_label} | {rating_text} | {gap_target_text} | {gap_one_text} | "
            f"{item.get('status')} | {item.get('action')} |"
        )
    lines.append("")

    missing = [item for item in queue if item.get("status") == "unknown"]
    lines.append("## Coverage Gaps")
    lines.append("")
    if not missing:
        lines.append("- All tracked datasets have live Kaggle ratings.")
    else:
        for item in missing:
            lines.append(f"- `{item.get('dataset_ref') or item.get('path')}` has no live rating match.")
    lines.append("")

    return "\n".join(lines)


def build_live_tracker_json(
    scores: list[DatasetScore],
    queue: list[dict[str, Any]],
    *,
    today: date,
    alert_under: float,
    target_rating: float,
) -> dict[str, Any]:
    live_values = [item.kaggle_usability_rating for item in scores if item.kaggle_usability_rating is not None]
    summary = summarize_live_queue(queue)
    return {
        "generated_on": today.isoformat(),
        "thresholds": {
            "alert_under": alert_under,
            "target_rating": target_rating,
            "max_rating": 1.0,
        },
        "summary": {
            "dataset_count": len(scores),
            "live_matched": len(live_values),
            "average_live_rating": round(statistics.mean(live_values), 4) if live_values else None,
            **summary,
        },
        "priority_queue": queue,
    }


def score_dataset(ds_dir: Path, root: Path) -> DatasetScore:
    rel = str(ds_dir.relative_to(root))
    metadata_path = ds_dir / "dataset-metadata.json"
    readme_path = ds_dir / "README.md"
    create_script = ds_dir / "create_dataset.py"
    explore_nb = ds_dir / "explore.ipynb"
    kernel_meta = ds_dir / "kernel-metadata.json"
    csv_files = sorted(ds_dir.glob("*.csv"))
    parquet_files = sorted(ds_dir.glob("*.parquet"))
    data_files = len(csv_files) + len(parquet_files)

    criteria = {
        "metadata_core": 0,   # max 25
        "documentation": 0,   # max 35
        "data_assets": 0,     # max 20
        "notebook_assets": 0, # max 10
        "discovery": 0,       # max 10
    }
    issues: list[str] = []

    ds_ref: str | None = None
    meta = load_json(metadata_path) if metadata_path.exists() else None
    if meta is None:
        issues.append("Missing or invalid dataset-metadata.json.")
    else:
        criteria["metadata_core"] += 5
        title = str(meta.get("title", "")).strip()
        ds_id = str(meta.get("id", "")).strip()
        ds_ref = ds_id or None
        description = str(meta.get("description", "")).strip()
        licenses = meta.get("licenses") if isinstance(meta.get("licenses"), list) else []
        keywords = meta.get("keywords") if isinstance(meta.get("keywords"), list) else []
        subtitle = str(meta.get("subtitle", "")).strip()

        if title:
            criteria["metadata_core"] += 4
        else:
            issues.append("Metadata missing title.")

        if ds_id:
            criteria["metadata_core"] += 4
        else:
            issues.append("Metadata missing id.")

        desc_len = len(description)
        if desc_len >= 600:
            criteria["metadata_core"] += 6
        elif desc_len >= 250:
            criteria["metadata_core"] += 4
        elif desc_len >= 80:
            criteria["metadata_core"] += 2
        else:
            issues.append("Metadata description is too short (<80 chars).")

        if licenses:
            criteria["metadata_core"] += 3
        else:
            issues.append("Metadata missing licenses.")

        if len(keywords) >= 5:
            criteria["metadata_core"] += 3
        elif keywords:
            criteria["metadata_core"] += 2
            issues.append("Metadata has fewer than 5 keywords.")
        else:
            issues.append("Metadata missing keywords.")

        if subtitle:
            criteria["discovery"] += 2
        else:
            issues.append("Metadata missing subtitle.")

    readme_text = ""
    if not readme_path.exists():
        issues.append("README.md missing.")
    else:
        criteria["documentation"] += 10
        readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
        if "| Column | Type | Null% | Unique | Sample values |" in readme_text:
            criteria["documentation"] += 10
        else:
            issues.append("README missing column dictionary table.")
        if "## Suggested Use Cases" in readme_text:
            criteria["documentation"] += 6
        else:
            issues.append("README missing suggested use cases section.")
        if "## Tags" in readme_text:
            criteria["documentation"] += 4
        else:
            issues.append("README missing tags section.")
        if "https://www.kaggle.com/datasets/" in readme_text:
            criteria["documentation"] += 5
        else:
            issues.append("README missing Kaggle dataset link.")

        if "## Description" in readme_text:
            criteria["discovery"] += 2
        else:
            issues.append("README missing Description section.")
        if len(readme_text) >= 2000:
            criteria["discovery"] += 2
        else:
            issues.append("README could be more detailed (<2000 chars).")

    if data_files > 0:
        criteria["data_assets"] += 10
    else:
        issues.append("No CSV/Parquet data files present.")
    if data_files > 1:
        criteria["data_assets"] += 4
    if csv_files:
        criteria["data_assets"] += 3
    else:
        issues.append("No CSV export available.")
    if parquet_files:
        criteria["data_assets"] += 3

    if explore_nb.exists():
        criteria["notebook_assets"] += 6
    else:
        issues.append("Missing explore.ipynb.")
    if kernel_meta.exists():
        criteria["notebook_assets"] += 4
    else:
        issues.append("Missing kernel-metadata.json for dataset explorer.")

    if create_script.exists():
        criteria["discovery"] += 4
    else:
        issues.append("Missing create_dataset.py generator.")

    score = max(0, min(100, sum(criteria.values())))
    title = ds_dir.name if not meta else str(meta.get("title", ds_dir.name))
    return DatasetScore(
        path=rel,
        title=title,
        score=score,
        score_10=score_out_of_10(score),
        tier=tier_for_score(score),
        criteria=criteria,
        issues=issues,
        data_files=data_files,
        dataset_ref=ds_ref,
    )


def discover_dataset_dirs(root: Path) -> list[Path]:
    ds_root = root / "datasets"
    if not ds_root.exists():
        return []
    return sorted([path for path in ds_root.iterdir() if path.is_dir()])


def criteria_averages(scores: list[DatasetScore]) -> dict[str, float]:
    if not scores:
        return {}

    totals: dict[str, int] = {}
    for item in scores:
        for key, value in item.criteria.items():
            totals[key] = totals.get(key, 0) + value

    return {key: round(total / len(scores), 2) for key, total in sorted(totals.items())}


def summarize_common_gaps(scores: list[DatasetScore], limit: int = 8) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in scores:
        for issue in item.issues:
            counts[issue] = counts.get(issue, 0) + 1
    return sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:limit]


def generate_markdown(scores: list[DatasetScore], today: date, fail_under: int) -> str:
    if not scores:
        return "# Dataset Usability Report\n\n- No dataset directories found.\n"

    values = [item.score for item in scores]
    avg = statistics.mean(values)
    med = statistics.median(values)
    avg10 = statistics.mean([item.score_10 for item in scores])
    live_values = [item.kaggle_usability_rating for item in scores if item.kaggle_usability_rating is not None]
    below = [item for item in scores if item.score < fail_under]
    avg_by_criteria = criteria_averages(scores)
    common_gaps = summarize_common_gaps(scores)

    lines = [
        "# Dataset Usability Report",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Dataset count: {len(scores)}",
        f"- Average score: {avg:.1f}",
        f"- Average score (10-point): {avg10:.1f}",
        f"- Median score: {med:.1f}",
        f"- Gate (fail-under): {fail_under}",
        f"- Below gate: {len(below)}",
        "- Note: `Score (10)` is an internal ops rubric. Kaggle `usabilityRating` is a separate 0.0-1.0 platform metric.",
    ]
    if live_values:
        lines.extend(
            [
                f"- Live Kaggle ratings matched: {len(live_values)}/{len(scores)}",
                f"- Average Kaggle `usabilityRating`: {statistics.mean(live_values):.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Scoreboard",
            "",
            "| Dataset | Score | Score (10) | Kaggle | Kaggle (10) | Tier | Key Gaps |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )

    for item in sorted(scores, key=lambda s: (s.score, s.path)):
        gaps = "; ".join(item.issues[:2]) if item.issues else "Strong baseline"
        kaggle_rating = f"{item.kaggle_usability_rating:.3f}" if item.kaggle_usability_rating is not None else "n/a"
        kaggle_10 = f"{item.kaggle_score_10:.1f}" if item.kaggle_score_10 is not None else "n/a"
        lines.append(
            f"| `{item.path}` | {item.score} | {item.score_10} | {kaggle_rating} | {kaggle_10} | {item.tier} | {gaps} |"
        )
    lines.append("")

    lines.append("## Focus Queue")
    lines.append("")
    if not below:
        lines.append("- No datasets below fail-under threshold.")
    else:
        for item in sorted(below, key=lambda s: s.score):
            lines.append(f"- `{item.path}` ({item.score}): {'; '.join(item.issues[:3])}")
    lines.append("")

    lines.append("## Criteria Averages")
    lines.append("")
    if not avg_by_criteria:
        lines.append("- No criteria to summarize.")
    else:
        for key, value in avg_by_criteria.items():
            lines.append(f"- `{key}`: {value:.1f}")
    lines.append("")

    lines.append("## Common Gaps")
    lines.append("")
    if not common_gaps:
        lines.append("- No repeated gaps detected.")
    else:
        for issue, count in common_gaps:
            lines.append(f"- {count} dataset(s): {issue}")
    lines.append("")

    if live_values:
        missing_live = [item for item in scores if item.kaggle_usability_rating is None]
        lines.append("## Live Coverage")
        lines.append("")
        if not missing_live:
            lines.append("- All local datasets matched a live Kaggle listing.")
        else:
            lines.append("- Missing from live Kaggle listing:")
            for item in missing_live:
                ref = item.dataset_ref or item.path
                lines.append(f"- `{ref}`")
        lines.append("")

    return "\n".join(lines)


def build_json_report(scores: list[DatasetScore], today: date, fail_under: int) -> dict[str, Any]:
    live_values = [item.kaggle_usability_rating for item in scores if item.kaggle_usability_rating is not None]
    return {
        "generated_on": today.isoformat(),
        "fail_under": fail_under,
        "summary": {
            "count": len(scores),
            "average_score": round(statistics.mean([item.score for item in scores]), 2) if scores else 0.0,
            "average_score_10": round(statistics.mean([item.score_10 for item in scores]), 2) if scores else 0.0,
            "median_score": round(statistics.median([item.score for item in scores]), 2) if scores else 0.0,
            "below_gate": sum(1 for item in scores if item.score < fail_under),
            "criteria_average": criteria_averages(scores),
            "live_kaggle": {
                "matched_count": len(live_values),
                "missing_count": len(scores) - len(live_values),
                "average_usability_rating": round(statistics.mean(live_values), 4) if live_values else None,
            },
        },
        "common_gaps": [
            {"issue": issue, "count": count} for issue, count in summarize_common_gaps(scores)
        ],
        "datasets": [
            {
                "path": item.path,
                "title": item.title,
                "score": item.score,
                "score_10": item.score_10,
                "tier": item.tier,
                "criteria": item.criteria,
                "issues": item.issues,
                "data_files": item.data_files,
                "dataset_ref": item.dataset_ref,
                "kaggle_usability_rating": item.kaggle_usability_rating,
                "kaggle_score_10": item.kaggle_score_10,
            }
            for item in sorted(scores, key=lambda s: (s.score, s.path))
        ],
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score dataset usability and emit reports.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for reports.")
    parser.add_argument("--today", default=None, help="Override date in YYYY-MM-DD format.")
    parser.add_argument("--fail-under", type=int, default=70, help="Exit non-zero if any dataset score is below this value.")
    parser.add_argument("--strict", action="store_true", help="Enable fail-under gate.")
    parser.add_argument("--live", action="store_true", help="Join report with live Kaggle usability ratings.")
    parser.add_argument(
        "--live-ratings-csv",
        default=None,
        help="Optional CSV path with ref/usabilityRating columns for live joins without Kaggle API.",
    )
    parser.add_argument("--owner", default=None, help="Kaggle owner for --live lookups.")
    parser.add_argument(
        "--daily-tracker",
        action="store_true",
        help="Emit daily live usability tracker with threshold alerts and ranking.",
    )
    parser.add_argument(
        "--alert-under",
        type=float,
        default=0.7,
        help="Critical alert threshold for live Kaggle usability ratings (default 0.7).",
    )
    parser.add_argument(
        "--target-rating",
        type=float,
        default=0.8,
        help="Target threshold used for watch/strong segmentation (default 0.8).",
    )
    parser.add_argument(
        "--fail-on-live-alert",
        action="store_true",
        help="Exit non-zero when any dataset live rating is below --alert-under.",
    )
    parser.add_argument(
        "--write-live-ratings-csv",
        default=None,
        help="Optional path to persist fetched live ratings CSV (`ref,usabilityRating`).",
    )
    parser.add_argument(
        "--fallback-live-ratings-csv",
        default=None,
        help="Optional CSV used when --live fetch fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fail_under < 0 or args.fail_under > 100:
        raise SystemExit("--fail-under must be between 0 and 100")
    if args.alert_under < 0.0 or args.alert_under > 1.0:
        raise SystemExit("--alert-under must be between 0.0 and 1.0")
    if args.target_rating < 0.0 or args.target_rating > 1.0:
        raise SystemExit("--target-rating must be between 0.0 and 1.0")
    if args.alert_under > args.target_rating:
        raise SystemExit("--alert-under cannot be greater than --target-rating")

    root = Path(args.root).resolve()
    output_root = Path(args.output_root)
    today = resolve_today(args.today)

    dataset_dirs = discover_dataset_dirs(root)
    if not dataset_dirs:
        raise SystemExit("No datasets/ directories found to score.")

    scores = [score_dataset(ds_dir, root=root) for ds_dir in dataset_dirs]
    live_loaded = False
    if args.live_ratings_csv:
        ratings = load_live_ratings_csv(Path(args.live_ratings_csv))
        scores = attach_kaggle_live_ratings(scores, ratings)
        live_loaded = True
        print(f"Live ratings loaded from CSV: {len(ratings)} refs")
    elif args.live:
        owner = (args.owner or infer_owner_from_scores(scores) or "").strip().lower()
        if owner:
            live_ratings, live_error = fetch_kaggle_live_ratings(owner)
            if live_error:
                print(f"Warning: live Kaggle rating lookup failed: {live_error}")
                fallback_path = Path(args.fallback_live_ratings_csv).resolve() if args.fallback_live_ratings_csv else None
                if fallback_path is not None and fallback_path.exists():
                    fallback_ratings = load_live_ratings_csv(fallback_path)
                    scores = attach_kaggle_live_ratings(scores, fallback_ratings)
                    live_loaded = True
                    print(f"Fallback live ratings loaded from CSV: {len(fallback_ratings)} refs ({fallback_path})")
            else:
                scores = attach_kaggle_live_ratings(scores, live_ratings)
                live_loaded = True
                print(f"Live Kaggle ratings loaded for owner '{owner}': {len(live_ratings)} refs")
                if args.write_live_ratings_csv:
                    local_refs = {
                        (item.dataset_ref or "").strip().lower()
                        for item in scores
                        if item.dataset_ref
                    }
                    filtered = {
                        ref: rating
                        for ref, rating in live_ratings.items()
                        if ref in local_refs
                    }
                    write_path = Path(args.write_live_ratings_csv).resolve()
                    write_live_ratings_csv(write_path, filtered)
                    print(f"Live ratings CSV written: {write_path}")
        else:
            print("Warning: unable to infer Kaggle owner for --live lookup; skipping live join.")

    markdown = generate_markdown(scores, today=today, fail_under=args.fail_under)
    json_report = build_json_report(scores, today=today, fail_under=args.fail_under)

    reports_dir = output_root / "reports"
    dated_md = reports_dir / f"dataset-usability-{today.isoformat()}.md"
    latest_md = reports_dir / "latest-dataset-usability.md"
    dated_json = reports_dir / f"dataset-usability-{today.isoformat()}.json"
    latest_json = reports_dir / "latest-dataset-usability.json"

    write_text(dated_md, markdown)
    write_text(latest_md, markdown)
    write_json(dated_json, json_report)
    write_json(latest_json, json_report)

    live_alert_fail = False
    if args.daily_tracker:
        if not live_loaded:
            print("Warning: --daily-tracker enabled without live ratings. Report will show unknown live status.")
        live_queue = build_live_priority_queue(
            scores,
            alert_under=args.alert_under,
            target_rating=args.target_rating,
        )
        live_markdown = generate_live_tracker_markdown(
            scores,
            live_queue,
            today=today,
            alert_under=args.alert_under,
            target_rating=args.target_rating,
        )
        live_json = build_live_tracker_json(
            scores,
            live_queue,
            today=today,
            alert_under=args.alert_under,
            target_rating=args.target_rating,
        )
        tracker_dated_md = reports_dir / f"dataset-usability-tracker-{today.isoformat()}.md"
        tracker_latest_md = reports_dir / "latest-dataset-usability-tracker.md"
        tracker_dated_json = reports_dir / f"dataset-usability-tracker-{today.isoformat()}.json"
        tracker_latest_json = reports_dir / "latest-dataset-usability-tracker.json"

        write_text(tracker_dated_md, live_markdown)
        write_text(tracker_latest_md, live_markdown)
        write_json(tracker_dated_json, live_json)
        write_json(tracker_latest_json, live_json)

        live_summary = live_json["summary"]
        print(f"Dataset usability tracker written: {tracker_dated_md}")
        print(
            "Live alerts: "
            f"critical={live_summary['critical']} "
            f"watch={live_summary['watch']} "
            f"strong={live_summary['strong']} "
            f"unknown={live_summary['unknown']}"
        )
        live_alert_fail = args.fail_on_live_alert and int(live_summary["critical"]) > 0

    below = [item for item in scores if item.score < args.fail_under]
    print(f"Dataset usability report written: {dated_md}")
    print(f"Latest dataset usability report: {latest_md}")
    print(
        "Summary: "
        f"{len(scores)} datasets, "
        f"average {statistics.mean([item.score for item in scores]):.1f}, "
        f"{len(below)} below gate {args.fail_under}"
    )

    strict_fail = args.strict and bool(below)
    if strict_fail:
        print(f"Dataset usability gate failed: {len(below)} dataset(s) below {args.fail_under}.")
    if live_alert_fail:
        print(
            f"Live usability alert gate failed: one or more datasets are below {args.alert_under:.3f}."
        )
    if strict_fail or live_alert_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
