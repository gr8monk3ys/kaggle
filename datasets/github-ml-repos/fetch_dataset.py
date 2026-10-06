#!/usr/bin/env python3
"""Snapshot the most-starred machine learning and AI repositories on GitHub.

Real data, not generated: every row comes from GitHub's public search API at
the moment this runs. Unlike the other datasets here it cannot be rebuilt
byte-for-byte later, so the CSV it writes is committed as source and this
script is named ``fetch_`` rather than ``create_`` to keep it out of
``./manage.sh build-datasets``.

Requires the GitHub CLI, signed in (``gh auth status``). Uses only the search
endpoint, paced under its limit of 30 requests a minute.

Usage:
    python fetch_dataset.py            # writes github_ml_repos.csv (~8 minutes)
    python fetch_dataset.py --max-pages 1   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "github_ml_repos.csv"

#: One search per topic. GitHub caps each search at 1,000 results (10 pages).
TOPICS = (
    "machine-learning",
    "deep-learning",
    "artificial-intelligence",
    "llm",
    "large-language-models",
    "nlp",
    "computer-vision",
    "reinforcement-learning",
    "data-science",
    "mlops",
    "generative-ai",
    "transformers",
    "pytorch",
    "tensorflow",
    "neural-network",
    "ai-agents",
    "rag",
    "diffusion-models",
    "time-series",
    "speech-recognition",
)
MIN_STARS = 500
PAUSE_SECONDS = 2.2  # 30 searches per minute, with headroom

FIELDS = (
    "repo",
    "owner",
    "owner_type",
    "description",
    "language",
    "stars",
    "forks",
    "open_issues",
    "size_kb",
    "license",
    "topics",
    "n_topics",
    "matched_topics",
    "created_at",
    "pushed_at",
    "age_days",
    "days_since_push",
    "archived",
    "is_fork",
    "has_homepage",
)


def search(topic: str, page: int) -> list[dict]:
    result = subprocess.run(
        [
            "gh", "api", "-X", "GET", "search/repositories",
            "-f", f"q=topic:{topic} stars:>={MIN_STARS}",
            "-f", "sort=stars",
            "-f", "order=desc",
            "-f", "per_page=100",
            "-f", f"page={page}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout).get("items", [])


def days_between(later: datetime, iso: str | None) -> int | None:
    if not iso:
        return None
    return (later - datetime.fromisoformat(iso.replace("Z", "+00:00"))).days


def to_row(item: dict, matched: set[str], now: datetime) -> dict:
    topics = item.get("topics") or []
    license_ = (item.get("license") or {}).get("spdx_id") or ""
    return {
        "repo": item["full_name"],
        "owner": item["owner"]["login"],
        "owner_type": item["owner"]["type"],
        "description": (item.get("description") or "").replace("\n", " ").strip(),
        "language": item.get("language") or "",
        "stars": item["stargazers_count"],
        "forks": item["forks_count"],
        "open_issues": item["open_issues_count"],
        "size_kb": item["size"],
        "license": "" if license_ == "NOASSERTION" else license_,
        "topics": "|".join(topics),
        "n_topics": len(topics),
        "matched_topics": "|".join(sorted(matched)),
        "created_at": item["created_at"],
        "pushed_at": item["pushed_at"],
        "age_days": days_between(now, item["created_at"]),
        "days_since_push": days_between(now, item["pushed_at"]),
        "archived": item["archived"],
        "is_fork": item["fork"],
        "has_homepage": bool(item.get("homepage")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args(argv)

    items: dict[str, dict] = {}
    matched: dict[str, set[str]] = {}
    for topic in TOPICS:
        for page in range(1, args.max_pages + 1):
            found = search(topic, page)
            time.sleep(PAUSE_SECONDS)
            for item in found:
                items[item["full_name"]] = item
                matched.setdefault(item["full_name"], set()).add(topic)
            if len(found) < 100:
                break
        print(f"{topic:<26} {len(items):>6} unique repos so far", flush=True)

    # Stamp the snapshot when fetching ends: a repo pushed during the run would
    # otherwise show a negative days_since_push.
    now = datetime.now(timezone.utc)
    rows = sorted(
        (to_row(item, matched[name], now) for name, item in items.items()),
        key=lambda row: (-row["stars"], row["repo"]),
    )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} repos to {OUTPUT.name} (snapshot {now:%Y-%m-%d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
