#!/usr/bin/env python3
"""Dataset usability optimizer: generates README.md per dataset and improves metadata.

What it does
------------
1. Scans all datasets/ subdirectories for CSV/Parquet files
2. For each file: computes dtype, null%, unique count, and 3 sample values
3. Writes a README.md into each dataset directory with:
   - Overview table
   - Column-by-column summary
   - Suggested use cases
4. Optionally patches dataset-metadata.json with an enriched description

Usage
-----
    python3 dataset_optimizer.py                    # generate READMEs only
    python3 dataset_optimizer.py --push             # generate + re-upload datasets
    python3 dataset_optimizer.py --dir datasets/job-postings  # single dataset

Invoked by: ./manage.sh optimize-datasets [--push]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).parent
DATASETS_DIR = ROOT / "datasets"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"

DEFAULT_DATASET_LICENSE = "GPL-3.0"
DEFAULT_AUTHOR_NAME = "Lorenzo Scaturchio"
DEFAULT_AUTHOR_BIO = (
    "Independent ML engineer building synthetic, education-first datasets "
    "for reproducible benchmarking and prototyping."
)
DEFAULT_GEOSPATIAL_COVERAGE = "Global (synthetic)"
DEFAULT_DOI = "Not assigned"
DEFAULT_COLLECTION_METHODOLOGY = (
    "Programmatic synthetic generation using seeded statistical distributions "
    "and rule-based constraints to mimic realistic structure while avoiding "
    "direct personal data."
)




def infer_temporal_coverage(meta: dict) -> tuple[str, str]:
    """Infer YYYY coverage from subtitle/description when possible."""
    subtitle = str(meta.get("subtitle", ""))
    description = str(meta.get("description", ""))
    combined = f"{subtitle} {description}"
    match = re.search(r"((?:19|20)\d{2})\s*[–-]\s*((?:19|20)\d{2})", combined)
    if not match:
        return "2020-01-01", "2025-12-31"
    start_year = match.group(1)
    end_year = match.group(2)
    return f"{start_year}-01-01", f"{end_year}-12-31"


def build_default_citation(meta: dict) -> str:
    """Build a canonical citation string for metadata completeness."""
    title = str(meta.get("title", "Dataset")).strip() or "Dataset"
    dataset_id = str(meta.get("id", "")).strip()
    if dataset_id:
        return (
            f"Scaturchio, Lorenzo (2026). {title}. Kaggle Dataset. "
            f"https://www.kaggle.com/datasets/{dataset_id}"
        )
    return f"Scaturchio, Lorenzo (2026). {title}. Kaggle Dataset."


def apply_metadata_defaults(meta: dict) -> tuple[dict, bool]:
    """Normalize metadata defaults for licensing and data-card completeness."""
    changed = False

    expected_license = [{"name": DEFAULT_DATASET_LICENSE}]
    if meta.get("licenses") != expected_license:
        meta["licenses"] = expected_license
        changed = True

    current_authors = meta.get("authors")
    if not isinstance(current_authors, list) or not current_authors:
        meta["authors"] = [{"name": DEFAULT_AUTHOR_NAME, "bio": DEFAULT_AUTHOR_BIO}]
        changed = True
    else:
        normalized_authors: list[dict[str, str]] = []
        for author in current_authors:
            if not isinstance(author, dict):
                continue
            name = str(author.get("name", "")).strip() or DEFAULT_AUTHOR_NAME
            bio = str(author.get("bio", "")).strip() or DEFAULT_AUTHOR_BIO
            normalized_authors.append({"name": name, "bio": bio})
        if not normalized_authors:
            normalized_authors = [{"name": DEFAULT_AUTHOR_NAME, "bio": DEFAULT_AUTHOR_BIO}]
        if normalized_authors != current_authors:
            meta["authors"] = normalized_authors
            changed = True

    inferred_start, inferred_end = infer_temporal_coverage(meta)
    current_coverage = meta.get("coverage")
    if isinstance(current_coverage, dict):
        normalized_coverage = {
            "temporal_start_date": str(
                current_coverage.get("temporal_start_date", inferred_start)
            ).strip()
            or inferred_start,
            "temporal_end_date": str(
                current_coverage.get("temporal_end_date", inferred_end)
            ).strip()
            or inferred_end,
            "geospatial_coverage": str(
                current_coverage.get("geospatial_coverage", DEFAULT_GEOSPATIAL_COVERAGE)
            ).strip()
            or DEFAULT_GEOSPATIAL_COVERAGE,
        }
    else:
        normalized_coverage = {
            "temporal_start_date": inferred_start,
            "temporal_end_date": inferred_end,
            "geospatial_coverage": DEFAULT_GEOSPATIAL_COVERAGE,
        }
    if current_coverage != normalized_coverage:
        meta["coverage"] = normalized_coverage
        changed = True

    doi = str(meta.get("doi", "")).strip()
    if not doi:
        meta["doi"] = DEFAULT_DOI
        changed = True

    current_provenance = meta.get("provenance")
    if isinstance(current_provenance, dict):
        sources = current_provenance.get("sources")
        if not isinstance(sources, list) or not sources:
            sources = [
                "Synthetic data generation scripts in this repository",
                "Public domain schemas and domain conventions for educational simulation",
            ]
        sources = [str(item).strip() for item in sources if str(item).strip()]
        if not sources:
            sources = [
                "Synthetic data generation scripts in this repository",
                "Public domain schemas and domain conventions for educational simulation",
            ]
        methodology = str(
            current_provenance.get("collection_methodology", DEFAULT_COLLECTION_METHODOLOGY)
        ).strip() or DEFAULT_COLLECTION_METHODOLOGY
        normalized_provenance = {
            "sources": sources,
            "collection_methodology": methodology,
        }
    else:
        normalized_provenance = {
            "sources": [
                "Synthetic data generation scripts in this repository",
                "Public domain schemas and domain conventions for educational simulation",
            ],
            "collection_methodology": DEFAULT_COLLECTION_METHODOLOGY,
        }
    if current_provenance != normalized_provenance:
        meta["provenance"] = normalized_provenance
        changed = True

    citations = meta.get("citations")
    if isinstance(citations, list):
        normalized_citations = [str(item).strip() for item in citations if str(item).strip()]
    else:
        normalized_citations = []
    if not normalized_citations:
        normalized_citations = [build_default_citation(meta)]
    if citations != normalized_citations:
        meta["citations"] = normalized_citations
        changed = True

    return meta, changed


# ── CSV Analysis ──────────────────────────────────────────────────────────────

def analyze_csv(path: Path, max_rows: int = 5000) -> dict:
    """Read up to max_rows of a CSV and return column stats."""
    columns: list[dict] = []
    rows_read = 0
    col_values: dict[str, list] = {}
    size_kb = round(path.stat().st_size / 1024, 1)

    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {
                    "file": path.name,
                    "error": "no header",
                    "columns": [],
                    "rows": 0,
                    "size_kb": size_kb,
                }
            for fn in reader.fieldnames:
                col_values[fn] = []
            for row in reader:
                if rows_read >= max_rows:
                    break
                for fn in reader.fieldnames:
                    col_values[fn].append(row.get(fn, ""))
                rows_read += 1
    except Exception as e:
        return {
            "file": path.name,
            "error": str(e),
            "columns": [],
            "rows": 0,
            "size_kb": size_kb,
        }

    for col, values in col_values.items():
        non_null = [v for v in values if v.strip() != ""]
        null_count = len(values) - len(non_null)
        null_pct = round(100 * null_count / len(values), 1) if values else 0.0
        unique_vals = set(non_null)
        n_unique = len(unique_vals)

        # Guess dtype
        dtype = _guess_dtype(non_null[:100])

        # Sample values (up to 3 unique, short)
        sample = _pick_samples(non_null, n_unique)

        columns.append({
            "name": col,
            "dtype": dtype,
            "null_pct": null_pct,
            "n_unique": n_unique,
            "samples": sample,
            "total": len(values),
        })

    return {
        "columns": columns,
        "rows": rows_read,
        "file": path.name,
        "size_kb": size_kb,
    }


def analyze_parquet(path: Path, max_rows: int = 5000) -> dict:
    """Read up to max_rows of a Parquet file and return column stats."""
    size_kb = round(path.stat().st_size / 1024, 1)
    try:
        import pandas as pd
    except ImportError:
        return {
            "file": path.name,
            "error": "pandas is required for parquet analysis",
            "columns": [],
            "rows": 0,
            "size_kb": size_kb,
        }

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {
            "file": path.name,
            "error": str(e),
            "columns": [],
            "rows": 0,
            "size_kb": size_kb,
        }

    if len(df) > max_rows:
        df = df.head(max_rows)

    columns: list[dict] = []
    for col in df.columns:
        series = df[col]
        values = [str(v) for v in series.tolist() if not pd.isna(v)]
        non_null = [v for v in values if v.strip() != ""]
        null_count = len(series) - len(non_null)
        null_pct = round(100 * null_count / len(series), 1) if len(series) else 0.0
        unique_vals = set(non_null)
        n_unique = len(unique_vals)
        dtype = _guess_dtype(non_null[:100])
        sample = _pick_samples(non_null, n_unique)
        columns.append({
            "name": str(col),
            "dtype": dtype,
            "null_pct": null_pct,
            "n_unique": n_unique,
            "samples": sample,
            "total": int(len(series)),
        })

    return {
        "columns": columns,
        "rows": int(len(df)),
        "file": path.name,
        "size_kb": size_kb,
    }


def _guess_dtype(values: list[str]) -> str:
    if not values:
        return "unknown"
    # Try int
    try:
        [int(v) for v in values if v]
        return "integer"
    except ValueError:
        pass
    # Try float
    try:
        [float(v) for v in values if v]
        return "float"
    except ValueError:
        pass
    # Check for boolean-like
    bool_vals = {"true", "false", "0", "1", "yes", "no"}
    if all(v.lower() in bool_vals for v in values if v):
        return "boolean"
    return "string"


def _pick_samples(values: list[str], n_unique: int) -> list[str]:
    """Pick up to 3 representative sample values."""
    seen: dict[str, int] = {}
    for v in values:
        v_stripped = v.strip()
        if v_stripped and len(v_stripped) < 60:
            seen[v_stripped] = seen.get(v_stripped, 0) + 1
        if len(seen) >= 10:
            break
    # Sort by frequency, pick top 3
    top = sorted(seen, key=lambda k: -seen[k])[:3]
    return top


# ── README generation ─────────────────────────────────────────────────────────

def generate_readme(ds_dir: Path, meta: dict, file_analyses: list[dict]) -> str:
    """Generate a README.md for a dataset directory."""
    title = meta.get("title", ds_dir.name)
    subtitle = meta.get("subtitle", "")
    description = meta.get("description", "")
    license_name = ""
    if meta.get("licenses"):
        license_name = meta["licenses"][0].get("name", "")
    keywords = meta.get("keywords", [])
    dataset_id = meta.get("id", "")

    lines = [f"# {title}", ""]
    if subtitle:
        lines += [f"> {subtitle}", ""]
    if license_name:
        lines += [f"**License:** {license_name}  ", ""]
    if dataset_id:
        kaggle_url = f"https://www.kaggle.com/datasets/{dataset_id}"
        lines += [f"**Kaggle:** [{dataset_id}]({kaggle_url})  ", ""]

    if description:
        lines += ["## Description", "", description, ""]

    if keywords:
        lines += ["## Tags", "", ", ".join(f"`{k}`" for k in keywords), ""]

    authors = meta.get("authors") if isinstance(meta.get("authors"), list) else []
    if authors:
        lines += ["## Authors", ""]
        for author in authors:
            if not isinstance(author, dict):
                continue
            name = str(author.get("name", "")).strip() or "Unknown"
            bio = str(author.get("bio", "")).strip()
            if bio:
                lines.append(f"- **{name}**: {bio}")
            else:
                lines.append(f"- **{name}**")
        lines.append("")

    coverage = meta.get("coverage") if isinstance(meta.get("coverage"), dict) else {}
    if coverage:
        temporal_start = str(coverage.get("temporal_start_date", "")).strip() or "n/a"
        temporal_end = str(coverage.get("temporal_end_date", "")).strip() or "n/a"
        geospatial = str(coverage.get("geospatial_coverage", "")).strip() or "n/a"
        lines += [
            "## Coverage",
            "",
            f"- Temporal: {temporal_start} to {temporal_end}",
            f"- Geospatial: {geospatial}",
            "",
        ]

    doi = str(meta.get("doi", "")).strip()
    citations = meta.get("citations") if isinstance(meta.get("citations"), list) else []
    normalized_citations = [str(item).strip() for item in citations if str(item).strip()]
    if doi or normalized_citations:
        lines += ["## DOI and Citations", ""]
        lines.append(f"- DOI: {doi or 'n/a'}")
        for citation in normalized_citations:
            lines.append(f"- {citation}")
        lines.append("")

    provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    if provenance:
        lines += ["## Provenance", ""]
        sources = provenance.get("sources") if isinstance(provenance.get("sources"), list) else []
        for source in sources:
            source_text = str(source).strip()
            if source_text:
                lines.append(f"- Source: {source_text}")
        methodology = str(provenance.get("collection_methodology", "")).strip()
        if methodology:
            lines.append(f"- Collection methodology: {methodology}")
        lines.append("")

    for analysis in file_analyses:
        file_name = analysis.get("file", "unknown-file")
        if "error" in analysis:
            lines += [f"## {file_name}", "", f"*Error reading file: {analysis['error']}*", ""]
            continue

        lines += [
            f"## {file_name}",
            "",
            f"**Rows:** {analysis['rows']:,}  |  "
            f"**Columns:** {len(analysis['columns'])}  |  "
            f"**Size:** {analysis['size_kb']:,} KB",
            "",
            "| Column | Type | Null% | Unique | Sample values |",
            "|--------|------|-------|--------|---------------|",
        ]
        for col in analysis["columns"]:
            sample_str = ", ".join(f"`{s}`" for s in col["samples"][:3]) or "—"
            null_str = f"{col['null_pct']:.1f}%"
            lines.append(
                f"| `{col['name']}` | {col['dtype']} | {null_str} | "
                f"{col['n_unique']:,} | {sample_str} |"
            )
        lines.append("")

    # Use cases (pull from existing metadata or generate generic ones)
    use_cases = _infer_use_cases(meta, file_analyses)
    if use_cases:
        lines += ["## Suggested Use Cases", ""]
        for uc in use_cases:
            lines.append(f"- {uc}")
        lines.append("")

    lines += [
        "---",
        f"*Generated by `dataset_optimizer.py` — {Path(__file__).name}*",
    ]
    return "\n".join(lines) + "\n"


def _infer_use_cases(meta: dict, file_analyses: list[dict]) -> list[str]:
    """Infer likely ML tasks from dataset metadata."""
    use_cases = []
    desc = (meta.get("description", "") + " " + meta.get("subtitle", "")).lower()
    keywords = [k.lower() for k in meta.get("keywords", [])]

    if any(w in desc + " ".join(keywords) for w in ["fraud", "anomaly"]):
        use_cases.append("Binary classification (fraud detection) with severe class imbalance")
        use_cases.append("Anomaly detection (Isolation Forest, Autoencoder)")
        use_cases.append("Threshold optimization (precision-recall tradeoff)")
    if any(w in desc + " ".join(keywords) for w in ["time series", "forecast", "sales", "price"]):
        use_cases.append("Time series forecasting with lag/rolling features")
        use_cases.append("Seasonal decomposition (trend, seasonality, residuals)")
    if any(w in desc + " ".join(keywords) for w in ["nlp", "text", "classification", "sentiment"]):
        use_cases.append("Text classification (TF-IDF, BERT embeddings)")
        use_cases.append("Named entity recognition or topic modeling")
    if any(w in desc + " ".join(keywords) for w in ["salary", "job", "employ"]):
        use_cases.append("Salary prediction (regression)")
        use_cases.append("Job category classification (multi-class)")
    if any(w in desc + " ".join(keywords) for w in ["student", "academic", "grade", "score"]):
        use_cases.append("Academic performance prediction (regression/classification)")
        use_cases.append("Feature importance analysis of student success factors")
    if any(w in desc + " ".join(keywords) for w in ["spotify", "music", "audio"]):
        use_cases.append("Music popularity prediction (regression)")
        use_cases.append("Genre clustering with audio features (k-means, UMAP)")
    if any(w in desc + " ".join(keywords) for w in ["mental health", "survey"]):
        use_cases.append("Mental health treatment prediction (classification)")
        use_cases.append("Workplace sentiment analysis")
    if not use_cases:
        use_cases.append("Exploratory data analysis and feature engineering practice")
        use_cases.append("Baseline classification or regression benchmarking")
    return use_cases


# ── Main optimizer ─────────────────────────────────────────────────────────────

def optimize_dataset(ds_dir: Path, push: bool = False) -> bool:
    """Optimize a single dataset directory."""
    meta_path = ds_dir / "dataset-metadata.json"
    if not meta_path.exists():
        print(f"  {YELLOW}SKIP{RESET} {ds_dir.name}: no dataset-metadata.json")
        return True

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  {RED}FAIL{RESET} {ds_dir.name}: invalid dataset-metadata.json ({exc})")
        return False
    if not isinstance(meta, dict):
        print(f"  {RED}FAIL{RESET} {ds_dir.name}: dataset-metadata.json must be a JSON object")
        return False

    meta, metadata_changed = apply_metadata_defaults(meta)
    if metadata_changed:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"    {GREEN}dataset-metadata.json normalized{RESET}")
    print(f"  Processing {BLUE}{ds_dir.name}{RESET}...")

    # Analyze tabular files
    csv_files = sorted(ds_dir.glob("*.csv"))
    parquet_files = sorted(ds_dir.glob("*.parquet"))

    file_analyses = []
    for f in csv_files:
        print(f"    Analyzing {f.name}... ", end="", flush=True)
        analysis = analyze_csv(f)
        if "error" in analysis:
            print(f"{RED}error: {analysis['error']}{RESET}")
        else:
            print(f"{GREEN}{analysis['rows']:,} rows, {len(analysis['columns'])} cols{RESET}")
        file_analyses.append(analysis)

    for f in parquet_files:
        print(f"    Analyzing {f.name}... ", end="", flush=True)
        analysis = analyze_parquet(f)
        if "error" in analysis:
            print(f"{RED}error: {analysis['error']}{RESET}")
        else:
            print(f"{GREEN}{analysis['rows']:,} rows, {len(analysis['columns'])} cols{RESET}")
        file_analyses.append(analysis)

    if not file_analyses:
        print(f"    {YELLOW}No CSV/Parquet files found{RESET}")

    # Generate README
    readme_path = ds_dir / "README.md"
    readme_content = generate_readme(ds_dir, meta, file_analyses)
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"    {GREEN}README.md written{RESET} ({len(readme_content):,} chars)")

    if push:
        print(f"    Pushing {ds_dir.name}... ", end="", flush=True)
        cli = kaggle_command()
        result = subprocess.run(
            [*cli, "datasets", "version", "-p", str(ds_dir),
             "-m", "Add README.md with column summaries and use cases",
             "--dir-mode", "zip"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if result.returncode == 0:
            print(f"{GREEN}OK{RESET}")
        else:
            # Try create if version fails
            result2 = subprocess.run(
                [*cli, "datasets", "create", "-p", str(ds_dir), "--dir-mode", "zip"],
                capture_output=True, text=True, cwd=str(ROOT),
            )
            if result2.returncode == 0:
                print(f"{GREEN}created{RESET}")
            else:
                msg = summarize_subprocess_error(
                    result.stdout,
                    result.stderr,
                    result2.stdout,
                    result2.stderr,
                )
                print(f"{RED}FAILED{RESET}: {msg}")
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset usability optimizer.")
    parser.add_argument("--push", action="store_true", help="Re-upload datasets after optimizing.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Optimize a single dataset directory (e.g. datasets/job-postings).")
    args = parser.parse_args(argv)

    print(f"{BLUE}=== Dataset Optimizer ==={RESET}\n")

    if args.dir:
        target_dir = args.dir if args.dir.is_absolute() else (ROOT / args.dir)
        dirs = [target_dir]
    else:
        dirs = sorted(d for d in DATASETS_DIR.iterdir() if d.is_dir())

    success = 0
    failed = 0
    for ds_dir in dirs:
        ok = optimize_dataset(ds_dir, push=args.push)
        if ok:
            success += 1
        else:
            failed += 1
        print()

    print(f"{BLUE}=== Done ==={RESET}  Optimized: {GREEN}{success}{RESET}  Failed: {RED}{failed}{RESET}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
