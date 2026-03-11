#!/usr/bin/env python3
"""Generate notebook promotion drafts for competition discussion forums.

Matches each notebook's tags/competition_sources to active competitions and
generates 2-sentence comment drafts to post in competition threads.

Usage
-----
    python3 -m kaggle_portfolio.notebooks.notebook_promoter        # print weekly promotion plan
    python3 -m kaggle_portfolio.notebooks.notebook_promoter --auto # submit via Playwright (future)

Invoked by: ./manage.sh promote-notebooks [--auto]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"

# Known competition forums paired with relevant topics
COMPETITION_TOPICS = {
    "store-sales-time-series-forecasting": {
        "url": "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/discussion",
        "topics": ["time series", "forecasting", "lightgbm", "store", "sales", "lag"],
    },
    "spaceship-titanic": {
        "url": "https://www.kaggle.com/competitions/spaceship-titanic/discussion",
        "topics": ["classification", "ensemble", "xgboost", "feature engineering", "titanic"],
    },
    "titanic": {
        "url": "https://www.kaggle.com/competitions/titanic/discussion",
        "topics": ["classification", "random forest", "feature engineering", "eda", "titanic"],
    },
    "digit-recognizer": {
        "url": "https://www.kaggle.com/competitions/digit-recognizer/discussion",
        "topics": ["cnn", "deep learning", "image", "mnist", "digit"],
    },
    "nlp-getting-started": {
        "url": "https://www.kaggle.com/competitions/nlp-getting-started/discussion",
        "topics": ["nlp", "text", "bert", "classification", "disaster", "tweets"],
    },
    "house-prices-advanced-regression-techniques": {
        "url": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/discussion",
        "topics": ["regression", "feature engineering", "house prices", "ensemble", "stacking"],
    },
}


def normalize_ref(value: str | None) -> str:
    return str(value or "").strip().lower()


def parse_ref_filter(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {
        normalize_ref(part)
        for part in raw.split(",")
        if normalize_ref(part)
    }


def load_notebooks() -> tuple[list[dict], list[str]]:
    """Load all kernel-metadata.json files (excluding dataset explorers)."""
    notebooks = []
    warnings: list[str] = []
    for meta_path in sorted(ROOT.rglob("kernel-metadata.json")):
        # Skip dataset explorer notebooks under datasets/* (path-separator agnostic).
        rel_parts = meta_path.relative_to(ROOT).parts
        if rel_parts and rel_parts[0] == "datasets":
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                warnings.append(f"{meta_path.relative_to(ROOT)}: metadata root is not an object")
                continue
            if not meta.get("id"):
                warnings.append(f"{meta_path.relative_to(ROOT)}: missing required field 'id'")
                continue
            if not meta.get("title"):
                warnings.append(f"{meta_path.relative_to(ROOT)}: missing required field 'title'")
                continue
            meta["_dir"] = str(meta_path.parent.relative_to(ROOT))
            notebooks.append(meta)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{meta_path.relative_to(ROOT)}: {exc}")
    return notebooks, warnings


def match_notebook_to_competitions(nb: dict) -> list[str]:
    """Return competition slugs this notebook is relevant to."""
    matches = []
    # Explicit competition_sources
    for comp in nb.get("competition_sources", []):
        if comp in COMPETITION_TOPICS:
            matches.append(comp)

    # Metadata keyword/title matching
    tags = [str(t).lower() for t in nb.get("tags", [])]
    keywords = [str(k).lower() for k in nb.get("keywords", [])]
    topics = [*tags, *keywords]
    title = nb.get("title", "").lower()
    nb_id = nb.get("id", "").lower()
    searchable = [title, nb_id, *topics]

    for slug, info in COMPETITION_TOPICS.items():
        if slug in matches:
            continue
        overlap = sum(
            1 for topic in info["topics"] if any(topic in field for field in searchable)
        )
        if overlap >= 2:
            matches.append(slug)

    return matches


def generate_promo_comment(nb: dict, comp_slug: str) -> str:
    """Generate a 2-sentence promotion comment."""
    title = nb.get("title", nb.get("_dir", "my notebook"))
    nb_id = nb.get("id", "")
    user = nb_id.split("/")[0] if "/" in nb_id else "lorenzoscaturchio"
    nb_slug = nb_id.split("/")[1] if "/" in nb_id else nb_id
    url = f"https://www.kaggle.com/code/{user}/{nb_slug}"

    verb_map = {
        "store-sales-time-series-forecasting": "covers LightGBM time series forecasting with lag features",
        "spaceship-titanic": "walks through the full ML pipeline with ensemble methods",
        "titanic": "covers feature engineering and model ensembling for Titanic",
        "digit-recognizer": "builds a CNN from scratch to 99%+ accuracy on MNIST",
        "nlp-getting-started": "applies BERT fine-tuning for disaster tweet classification",
        "house-prices-advanced-regression-techniques": "explores feature engineering for house price prediction",
    }
    verb = verb_map.get(comp_slug, "provides a complete ML walkthrough for this competition")

    return (
        f"I put together a notebook that {verb}: {url}\n"
        f"Happy to answer questions or discuss approaches — upvote if it's useful!"
    )


def notebook_url(nb: dict) -> str:
    nb_id = str(nb.get("id") or "")
    user = nb_id.split("/")[0] if "/" in nb_id else "lorenzoscaturchio"
    nb_slug = nb_id.split("/")[1] if "/" in nb_id else nb_id
    return f"https://www.kaggle.com/code/{user}/{nb_slug}"


def generate_manual_share_copy(nb: dict) -> str:
    title = nb.get("title", nb.get("_dir", "my notebook"))
    url = notebook_url(nb)
    return (
        f"I published a notebook on {title}: {url}\n"
        f"If you're working in this area, I'd be interested in what you'd extend or benchmark next."
    )


def filter_notebooks(
    notebooks: list[dict],
    refs: set[str],
) -> tuple[list[dict], list[str]]:
    if not refs:
        return notebooks, []

    filtered: list[dict] = []
    seen: set[str] = set()
    for nb in notebooks:
        nb_id = normalize_ref(nb.get("id"))
        slug = nb_id.split("/")[-1] if nb_id else ""
        rel_dir = normalize_ref(nb.get("_dir"))
        if nb_id in refs or slug in refs or rel_dir in refs:
            filtered.append(nb)
            if nb_id:
                seen.add(nb_id)
            if slug:
                seen.add(slug)
            if rel_dir:
                seen.add(rel_dir)

    missing = sorted(ref for ref in refs if ref not in seen)
    return filtered, missing


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Auto-submit via Playwright (not yet implemented).")
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Fail when any notebook metadata file cannot be parsed.",
    )
    parser.add_argument(
        "--refs",
        default=None,
        help="Optional comma-separated notebook refs/slugs/directories to include exactly.",
    )
    args = parser.parse_args(argv)

    print(f"{BLUE}=== Notebook Promotion Planner ==={RESET}\n")

    notebooks, warnings = load_notebooks()
    selected_refs = parse_ref_filter(args.refs)
    notebooks, missing_refs = filter_notebooks(notebooks, selected_refs)
    print(f"Loaded {len(notebooks)} notebooks\n")
    if warnings:
        print(f"{YELLOW}Skipped {len(warnings)} notebook(s) due to metadata issues:{RESET}")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  - ... and {len(warnings) - 10} more")
        print()
        if args.strict_metadata:
            return 1
    if missing_refs:
        print(f"{YELLOW}Requested refs not found:{RESET}")
        for ref in missing_refs:
            print(f"  - {ref}")
        print()

    plan: dict[str, list[dict]] = {}  # comp_slug → list of (nb, comment)
    manual_share: list[dict] = []

    for nb in notebooks:
        comps = match_notebook_to_competitions(nb)
        if not comps:
            manual_share.append(
                {
                    "notebook": nb.get("_dir"),
                    "title": nb.get("title", nb.get("_dir")),
                    "url": notebook_url(nb),
                    "comment": generate_manual_share_copy(nb),
                }
            )
            continue
        for comp in comps:
            plan.setdefault(comp, []).append({
                "notebook": nb.get("_dir"),
                "title": nb.get("title", nb.get("_dir")),
                "comment": generate_promo_comment(nb, comp),
            })

    if not plan and not manual_share:
        print(f"{YELLOW}No notebook-competition matches found.{RESET}")
        return 0

    total = sum(len(v) for v in plan.values())
    print(f"Found {total} promotion opportunities across {len(plan)} competitions.\n")
    print("Post these this week (prioritize competitions with most active discussions):\n")

    for comp_slug, entries in sorted(plan.items()):
        comp_info = COMPETITION_TOPICS[comp_slug]
        print(f"{GREEN}── {comp_slug} ──{RESET}")
        print(f"   Forum: {comp_info['url']}")
        for entry in entries:
            print(f"\n   Notebook: {entry['notebook']}")
            print(f"   Draft comment:")
            for line in entry["comment"].splitlines():
                print(f"     {line}")
        print()

    if manual_share:
        print(f"{YELLOW}Manual share targets (no competition forum match):{RESET}")
        for entry in manual_share:
            print(f"\n  Notebook: {entry['notebook']}")
            print("  Draft share copy:")
            for line in entry["comment"].splitlines():
                print(f"    {line}")
        print()

    if args.auto:
        print(f"{YELLOW}--auto not yet implemented. Post manually using the drafts above.{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
