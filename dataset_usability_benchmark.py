#!/usr/bin/env python3
"""Benchmark local datasets against public Kaggle datasets with high usability ratings."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from kaggle_utils import (
    kaggle_command,
    parse_iso_date,
    resolve_today,
    run_kaggle,
    summarize_subprocess_error,
)

DEFAULT_OUTPUT_ROOT = Path("medal_ops")


@dataclass(frozen=True)
class ListingRow:
    ref: str
    title: str
    size: int
    last_updated: str
    download_count: int
    vote_count: int
    usability_rating: float


@dataclass(frozen=True)
class DatasetFeatures:
    ref: str
    title: str
    usability_rating: float | None
    vote_count: int | None
    download_count: int | None
    is_private: bool | None
    has_license: bool
    keyword_count: int
    subtitle_length: int
    description_length: int
    file_count: int
    has_csv: bool
    has_notebook_or_script: bool




def parse_listing_csv(raw_csv: str) -> list[ListingRow]:
    rows: list[ListingRow] = []
    reader = csv.DictReader(io.StringIO(raw_csv))
    for row in reader:
        ref = str(row.get("ref", "")).strip().lower()
        if not ref:
            continue
        try:
            rows.append(
                ListingRow(
                    ref=ref,
                    title=str(row.get("title", "")).strip(),
                    size=int(str(row.get("size", "0")).strip() or 0),
                    last_updated=str(row.get("lastUpdated", "")).strip(),
                    download_count=int(str(row.get("downloadCount", "0")).strip() or 0),
                    vote_count=int(str(row.get("voteCount", "0")).strip() or 0),
                    usability_rating=float(str(row.get("usabilityRating", "0")).strip() or 0.0),
                )
            )
        except ValueError:
            continue
    return rows


def fetch_public_listings(sort_by: str, pages: int) -> list[ListingRow]:
    all_rows: dict[str, ListingRow] = {}
    for page in range(1, pages + 1):
        raw = run_kaggle(["datasets", "list", "--sort-by", sort_by, "--page", str(page), "--csv"])
        for row in parse_listing_csv(raw):
            if row.ref not in all_rows:
                all_rows[row.ref] = row
    return list(all_rows.values())


def choose_exemplars(rows: list[ListingRow], target_rating: float, max_items: int) -> list[ListingRow]:
    above_target = [row for row in rows if row.usability_rating >= target_rating]
    if not above_target:
        above_target = sorted(rows, key=lambda row: (row.usability_rating, row.vote_count), reverse=True)
    else:
        above_target = sorted(above_target, key=lambda row: (row.vote_count, row.download_count), reverse=True)
    return above_target[:max_items]


def load_stringified_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        decoded = json.loads(payload)
        if isinstance(decoded, dict):
            return decoded
    raise ValueError(f"Unsupported metadata payload format in {path}")


def parse_files_csv(raw_csv: str) -> tuple[int, bool, bool]:
    reader = csv.DictReader(io.StringIO(raw_csv))
    names = [str(row.get("name", "")).strip().lower() for row in reader if str(row.get("name", "")).strip()]
    file_count = len(names)
    has_csv = any(name.endswith(".csv") for name in names)
    has_notebook_or_script = any(name.endswith(ext) for name in names for ext in (".ipynb", ".py", ".r", ".rmd"))
    return file_count, has_csv, has_notebook_or_script


def fetch_dataset_features(ref: str, temp_root: Path) -> DatasetFeatures:
    staging = temp_root / ref.replace("/", "__")
    staging.mkdir(parents=True, exist_ok=True)

    run_kaggle(["datasets", "metadata", ref, "-p", str(staging)])
    meta = load_stringified_metadata(staging / "dataset-metadata.json")

    files_raw = run_kaggle(["datasets", "files", ref, "--csv"])
    file_count, has_csv, has_notebook_or_script = parse_files_csv(files_raw)

    keywords = meta.get("keywords") if isinstance(meta.get("keywords"), list) else []
    licenses = meta.get("licenses") if isinstance(meta.get("licenses"), list) else []
    subtitle = str(meta.get("subtitle", "")).strip()
    description = str(meta.get("description", "")).strip()

    return DatasetFeatures(
        ref=ref,
        title=str(meta.get("title", ref)).strip(),
        usability_rating=float(meta["usabilityRating"]) if meta.get("usabilityRating") is not None else None,
        vote_count=int(meta["totalVotes"]) if meta.get("totalVotes") is not None else None,
        download_count=int(meta["totalDownloads"]) if meta.get("totalDownloads") is not None else None,
        # `datasets metadata <public-ref>` may omit isPrivate; treat omission as public.
        is_private=bool(meta["isPrivate"]) if meta.get("isPrivate") is not None else False,
        has_license=bool(licenses),
        keyword_count=len(keywords),
        subtitle_length=len(subtitle),
        description_length=len(description),
        file_count=file_count,
        has_csv=has_csv,
        has_notebook_or_script=has_notebook_or_script,
    )


def discover_local_dataset_dirs(root: Path) -> list[Path]:
    ds_root = root / "datasets"
    if not ds_root.exists():
        return []
    return sorted(path for path in ds_root.iterdir() if path.is_dir() and (path / "dataset-metadata.json").exists())


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def local_dataset_features(ds_dir: Path, root: Path) -> DatasetFeatures:
    meta = load_json_object(ds_dir / "dataset-metadata.json")
    ref = str(meta.get("id", "")).strip() or str(ds_dir.relative_to(root))
    title = str(meta.get("title", ds_dir.name)).strip()

    keywords = meta.get("keywords") if isinstance(meta.get("keywords"), list) else []
    licenses = meta.get("licenses") if isinstance(meta.get("licenses"), list) else []
    subtitle = str(meta.get("subtitle", "")).strip()
    description = str(meta.get("description", "")).strip()

    file_names = [p.name.lower() for p in ds_dir.iterdir() if p.is_file()]
    data_file_names = [name for name in file_names if name.endswith((".csv", ".parquet"))]

    is_private_raw = meta.get("isPrivate", meta.get("is_private"))
    is_private: bool | None
    if isinstance(is_private_raw, bool):
        is_private = is_private_raw
    else:
        is_private = None

    return DatasetFeatures(
        ref=ref,
        title=title,
        usability_rating=None,
        vote_count=None,
        download_count=None,
        is_private=is_private,
        has_license=bool(licenses),
        keyword_count=len(keywords),
        subtitle_length=len(subtitle),
        description_length=len(description),
        file_count=len(data_file_names),
        has_csv=any(name.endswith(".csv") for name in data_file_names),
        has_notebook_or_script=any(name.endswith(ext) for name in file_names for ext in (".ipynb", ".py", ".r", ".rmd")),
    )


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize_feature_set(features: list[DatasetFeatures]) -> dict[str, float | int]:
    count = len(features)
    if count == 0:
        return {
            "count": 0,
            "public_count": 0,
            "public_pct": 0.0,
            "license_pct": 0.0,
            "csv_pct": 0.0,
            "starter_asset_pct": 0.0,
            "keyword_median": 0.0,
            "subtitle_median": 0.0,
            "description_median": 0.0,
            "file_count_median": 0.0,
        }

    public_count = sum(1 for item in features if item.is_private is False)
    license_count = sum(1 for item in features if item.has_license)
    csv_count = sum(1 for item in features if item.has_csv)
    starter_count = sum(1 for item in features if item.has_notebook_or_script)

    return {
        "count": count,
        "public_count": public_count,
        "public_pct": round((100.0 * public_count) / count, 1),
        "license_pct": round((100.0 * license_count) / count, 1),
        "csv_pct": round((100.0 * csv_count) / count, 1),
        "starter_asset_pct": round((100.0 * starter_count) / count, 1),
        "keyword_median": round(_median([item.keyword_count for item in features]), 1),
        "subtitle_median": round(_median([item.subtitle_length for item in features]), 1),
        "description_median": round(_median([item.description_length for item in features]), 1),
        "file_count_median": round(_median([item.file_count for item in features]), 1),
    }


def build_recommendations(local_summary: dict[str, float | int], benchmark_summary: dict[str, float | int]) -> list[str]:
    recs: list[str] = []

    if float(local_summary.get("public_pct", 0.0)) < float(benchmark_summary.get("public_pct", 0.0)):
        recs.append(
            "For datasets targeting max live Kaggle usability, publish public variants (`isPrivate=false`) and reserve private mode for staging only."
        )

    if float(local_summary.get("keyword_median", 0.0)) < float(benchmark_summary.get("keyword_median", 0.0)):
        recs.append(
            "Increase keyword coverage to at least the benchmark median (typically ~4-5 validated Kaggle tags)."
        )

    if float(local_summary.get("starter_asset_pct", 0.0)) < float(benchmark_summary.get("starter_asset_pct", 0.0)):
        recs.append(
            "Ensure each dataset includes starter assets (e.g., `explore.ipynb` and/or `starter_baseline.py`)."
        )

    if float(local_summary.get("license_pct", 0.0)) < 100.0:
        recs.append(
            "Enforce explicit license metadata for every dataset (`licenses` field) to avoid platform penalties."
        )

    if float(local_summary.get("csv_pct", 0.0)) < 100.0:
        recs.append("Provide at least one CSV export per dataset for immediate Kaggle usability.")

    if not recs:
        recs.append("Local structure already matches benchmark medians. Next gains are likely from public visibility and audience engagement.")

    return recs


def format_markdown_report(
    today: date,
    target_rating: float,
    exemplar_rows: list[ListingRow],
    exemplar_features: list[DatasetFeatures],
    local_features: list[DatasetFeatures],
    benchmark_summary: dict[str, float | int],
    local_summary: dict[str, float | int],
    recommendations: list[str],
) -> str:
    top_refs = exemplar_rows[:10]
    lines = [
        "# Kaggle Usability Benchmark",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Target rating: {target_rating}",
        f"- Benchmark exemplar count: {len(exemplar_features)}",
        f"- Local dataset count: {len(local_features)}",
        "",
        "## Public Exemplars",
        "",
    ]

    if top_refs:
        for row in top_refs:
            lines.append(
                f"- `{row.ref}` rating={row.usability_rating:.3f}, votes={row.vote_count}, downloads={row.download_count}"
            )
    else:
        lines.append("- No exemplars found for the selected threshold.")

    lines.extend(
        [
            "",
            "## Benchmark Summary",
            "",
            f"- Public datasets: {benchmark_summary['public_pct']}%",
            f"- License coverage: {benchmark_summary['license_pct']}%",
            f"- CSV coverage: {benchmark_summary['csv_pct']}%",
            f"- Starter-asset coverage (`.ipynb`/`.py`): {benchmark_summary['starter_asset_pct']}%",
            f"- Median keyword count: {benchmark_summary['keyword_median']}",
            f"- Median subtitle length: {benchmark_summary['subtitle_median']}",
            f"- Median description length: {benchmark_summary['description_median']}",
            f"- Median data-file count: {benchmark_summary['file_count_median']}",
            "",
            "## Local Summary",
            "",
            f"- Public datasets: {local_summary['public_pct']}%",
            f"- License coverage: {local_summary['license_pct']}%",
            f"- CSV coverage: {local_summary['csv_pct']}%",
            f"- Starter-asset coverage (`.ipynb`/`.py`): {local_summary['starter_asset_pct']}%",
            f"- Median keyword count: {local_summary['keyword_median']}",
            f"- Median subtitle length: {local_summary['subtitle_median']}",
            f"- Median description length: {local_summary['description_median']}",
            f"- Median data-file count: {local_summary['file_count_median']}",
            "",
            "## Recommended Feature Additions",
            "",
        ]
    )
    for rec in recommendations:
        lines.append(f"- {rec}")

    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local datasets against top Kaggle usability exemplars.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for reports.")
    parser.add_argument("--today", default=None, help="Override date in YYYY-MM-DD format.")
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=10,
        help="How many pages to sample from each public listing sort (updated + votes).",
    )
    parser.add_argument(
        "--max-exemplars",
        type=int,
        default=20,
        help="Maximum public exemplars to inspect deeply.",
    )
    parser.add_argument(
        "--target-rating",
        type=float,
        default=1.0,
        help="Target Kaggle usability rating for exemplar selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_pages < 1:
        raise SystemExit("--sample-pages must be >= 1")
    if args.max_exemplars < 1:
        raise SystemExit("--max-exemplars must be >= 1")
    if args.target_rating < 0.0 or args.target_rating > 1.0:
        raise SystemExit("--target-rating must be between 0.0 and 1.0")

    root = Path(args.root).resolve()
    output_root = Path(args.output_root)
    today = resolve_today(args.today)

    sampled_rows = fetch_public_listings("updated", pages=args.sample_pages)
    sampled_rows.extend(fetch_public_listings("votes", pages=args.sample_pages))

    deduped_rows: dict[str, ListingRow] = {}
    for row in sampled_rows:
        current = deduped_rows.get(row.ref)
        if current is None or row.vote_count > current.vote_count:
            deduped_rows[row.ref] = row

    exemplars = choose_exemplars(list(deduped_rows.values()), args.target_rating, args.max_exemplars)

    exemplar_features: list[DatasetFeatures] = []
    exemplar_errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="kaggle-usability-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        for row in exemplars:
            try:
                exemplar_features.append(fetch_dataset_features(row.ref, temp_root))
            except Exception as exc:  # pragma: no cover - network+CLI surface
                exemplar_errors.append(f"{row.ref}: {exc}")

    local_dirs = discover_local_dataset_dirs(root)
    local_features = [local_dataset_features(path, root) for path in local_dirs]

    benchmark_summary = summarize_feature_set(exemplar_features)
    local_summary = summarize_feature_set(local_features)
    recommendations = build_recommendations(local_summary, benchmark_summary)

    markdown = format_markdown_report(
        today=today,
        target_rating=args.target_rating,
        exemplar_rows=exemplars,
        exemplar_features=exemplar_features,
        local_features=local_features,
        benchmark_summary=benchmark_summary,
        local_summary=local_summary,
        recommendations=recommendations,
    )

    payload = {
        "generated_on": today.isoformat(),
        "target_rating": args.target_rating,
        "sample_pages": args.sample_pages,
        "max_exemplars": args.max_exemplars,
        "benchmark_summary": benchmark_summary,
        "local_summary": local_summary,
        "recommendations": recommendations,
        "exemplar_errors": exemplar_errors,
        "exemplars": [asdict(item) for item in exemplar_features],
        "local_datasets": [asdict(item) for item in local_features],
    }

    reports_dir = output_root / "reports"
    dated_md = reports_dir / f"usability-benchmark-{today.isoformat()}.md"
    latest_md = reports_dir / "latest-usability-benchmark.md"
    dated_json = reports_dir / f"usability-benchmark-{today.isoformat()}.json"
    latest_json = reports_dir / "latest-usability-benchmark.json"

    write_text(dated_md, markdown)
    write_text(latest_md, markdown)
    write_json(dated_json, payload)
    write_json(latest_json, payload)

    print(f"Usability benchmark report written: {dated_md}")
    print(f"Latest usability benchmark report: {latest_md}")
    print(
        "Summary: "
        f"{len(exemplar_features)} exemplars, "
        f"{len(local_features)} local datasets, "
        f"{len(exemplar_errors)} exemplar fetch errors"
    )

    if exemplar_errors:
        print("Warnings: some exemplar fetches failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
