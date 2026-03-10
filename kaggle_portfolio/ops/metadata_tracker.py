#!/usr/bin/env python3
"""Track notebook metadata changes alongside vote count snapshots.

Records timestamped snapshots of kernel-metadata.json fields and live vote
counts, enabling correlation analysis between metadata tweaks (title changes,
keyword additions) and vote movement.

Usage
-----
    python3 -m kaggle_portfolio.ops.metadata_tracker snapshot              # take a snapshot
    python3 -m kaggle_portfolio.ops.metadata_tracker snapshot --dry-run    # preview without writing
    python3 -m kaggle_portfolio.ops.metadata_tracker annotate feature-engineering "Updated title for SEO"
    python3 -m kaggle_portfolio.ops.metadata_tracker report                # show metadata changes vs votes
    python3 -m kaggle_portfolio.ops.metadata_tracker report --json         # machine-readable report

Invoked by: ./manage.sh metadata-tracker <subcommand> [args...]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from kaggle_portfolio.shared.kaggle_utils import kaggle_command, summarize_subprocess_error

ROOT = Path(__file__).resolve().parents[2]
MEDAL_OPS_DIR = ROOT / "medal_ops"
LOG_PATH = MEDAL_OPS_DIR / "metadata_ab_log.json"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------

def _load_log() -> list[dict]:
    """Load the snapshot log from disk."""
    if not LOG_PATH.exists():
        return []
    try:
        data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_log(entries: list[dict]) -> None:
    """Write the snapshot log to disk."""
    MEDAL_OPS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------

def collect_metadata() -> dict[str, dict]:
    """Scan all kernel-metadata.json files and return a dict keyed by directory name."""
    results: dict[str, dict] = {}
    for meta_path in sorted(ROOT.rglob("kernel-metadata.json")):
        # Skip node_modules, .venv, etc.
        rel = meta_path.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        dir_name = str(rel.parent)
        results[dir_name] = {
            "id": meta.get("id", ""),
            "title": meta.get("title", ""),
            "keywords": meta.get("keywords", []),
            "enable_gpu": meta.get("enable_gpu", False),
            "dataset_sources": meta.get("dataset_sources", []),
            "competition_sources": meta.get("competition_sources", []),
        }
    return results


def fetch_vote_counts() -> dict[str, int]:
    """Fetch vote counts from Kaggle CLI for all owned kernels.

    Returns a dict mapping kernel slug (e.g., 'feature-engineering') to votes.
    """
    votes: dict[str, int] = {}
    try:
        cli = kaggle_command()
        result = subprocess.run(
            [*cli, "kernels", "list", "--mine", "--csv", "--page-size", "50"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return votes

        reader = csv.DictReader(io.StringIO(result.stdout))
        for row in reader:
            ref = row.get("ref", "")
            slug = ref.split("/")[-1] if "/" in ref else ref
            vote_col = next(
                (k for k in row if "vote" in k.lower() or "upvote" in k.lower()),
                None,
            )
            if vote_col:
                try:
                    votes[slug] = int(row[vote_col] or 0)
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return votes


def _merge_votes(metadata: dict[str, dict], votes: dict[str, int]) -> dict[str, dict]:
    """Merge vote counts into metadata entries, matching by slug."""
    for dir_name, entry in metadata.items():
        kernel_id = entry.get("id", "")
        slug = kernel_id.split("/")[-1] if "/" in kernel_id else dir_name
        entry["votes"] = votes.get(slug, 0)
    return metadata


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_snapshot(dry_run: bool = False, votes: dict[str, int] | None = None) -> int:
    """Take a snapshot of all metadata + votes and append to the log."""
    metadata = collect_metadata()
    if votes is None:
        votes = fetch_vote_counts()

    metadata = _merge_votes(metadata, votes)

    snapshot = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "notebooks": metadata,
        "annotation": None,
    }

    if dry_run:
        print(f"{YELLOW}DRY RUN{RESET} — would write snapshot with "
              f"{len(metadata)} notebooks")
        for name, entry in sorted(metadata.items()):
            print(f"  {name}: votes={entry.get('votes', '?')} "
                  f"title={entry.get('title', '?')[:50]}")
        return 0

    log = _load_log()
    log.append(snapshot)
    _save_log(log)
    print(f"{GREEN}Snapshot saved{RESET} — {len(metadata)} notebooks, "
          f"{sum(e.get('votes', 0) for e in metadata.values())} total votes")
    return 0


def cmd_annotate(directory: str, note: str) -> int:
    """Add an annotation to the most recent snapshot."""
    log = _load_log()
    if not log:
        print(f"{RED}No snapshots found. Run 'snapshot' first.{RESET}")
        return 1

    latest = log[-1]
    existing_annotation = latest.get("annotation") or {}
    if isinstance(existing_annotation, str):
        existing_annotation = {"_general": existing_annotation}
    existing_annotation[directory] = note
    latest["annotation"] = existing_annotation
    _save_log(log)

    print(f"{GREEN}Annotated{RESET} latest snapshot: {directory} → {note}")
    return 0


def cmd_report(as_json: bool = False) -> int:
    """Show metadata changes correlated with vote deltas across snapshots."""
    log = _load_log()
    if len(log) < 2:
        print("Need at least 2 snapshots for comparison. "
              f"Currently have {len(log)}.")
        return 0

    changes: list[dict] = []

    for i in range(1, len(log)):
        prev = log[i - 1]
        curr = log[i]
        ts = curr.get("timestamp", "?")
        annotation = curr.get("annotation")

        prev_nbs = prev.get("notebooks", {})
        curr_nbs = curr.get("notebooks", {})

        for name in sorted(set(prev_nbs) | set(curr_nbs)):
            p = prev_nbs.get(name, {})
            c = curr_nbs.get(name, {})

            vote_delta = c.get("votes", 0) - p.get("votes", 0)
            title_changed = p.get("title") != c.get("title") and p.get("title")
            keywords_changed = (
                set(p.get("keywords", [])) != set(c.get("keywords", []))
                and p.get("keywords") is not None
            )

            if vote_delta != 0 or title_changed or keywords_changed:
                entry = {
                    "timestamp": ts,
                    "notebook": name,
                    "vote_delta": vote_delta,
                    "votes_now": c.get("votes", 0),
                }
                if title_changed:
                    entry["title_from"] = p.get("title", "")
                    entry["title_to"] = c.get("title", "")
                if keywords_changed:
                    entry["keywords_added"] = sorted(
                        set(c.get("keywords", [])) - set(p.get("keywords", []))
                    )
                    entry["keywords_removed"] = sorted(
                        set(p.get("keywords", [])) - set(c.get("keywords", []))
                    )
                if isinstance(annotation, dict) and name in annotation:
                    entry["annotation"] = annotation[name]
                changes.append(entry)

    if as_json:
        print(json.dumps(changes, indent=2))
        return 0

    if not changes:
        print("No metadata or vote changes detected between snapshots.")
        return 0

    print(f"{BLUE}=== Metadata A/B Tracker Report ==={RESET}\n")
    print(f"{'Notebook':<35} {'Votes':>6} {'Delta':>7}  Changes")
    print("-" * 80)
    for ch in changes:
        name = ch["notebook"][:34]
        delta = ch["vote_delta"]
        votes = ch["votes_now"]
        delta_color = GREEN if delta > 0 else (RED if delta < 0 else RESET)
        delta_str = f"{'+' if delta > 0 else ''}{delta}"

        parts = []
        if "title_to" in ch:
            parts.append(f"title→'{ch['title_to'][:30]}'")
        if ch.get("keywords_added"):
            parts.append(f"+kw:{','.join(ch['keywords_added'][:3])}")
        if ch.get("keywords_removed"):
            parts.append(f"-kw:{','.join(ch['keywords_removed'][:3])}")
        if ch.get("annotation"):
            parts.append(f"[{ch['annotation'][:30]}]")

        desc = "  ".join(parts) if parts else "(vote change only)"
        print(f"{name:<35} {votes:>6} {delta_color}{delta_str:>7}{RESET}  {desc}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track notebook metadata changes vs vote deltas."
    )
    sub = parser.add_subparsers(dest="command")

    snap = sub.add_parser("snapshot", help="Take a metadata + vote snapshot.")
    snap.add_argument("--dry-run", action="store_true",
                      help="Preview without writing.")

    ann = sub.add_parser("annotate",
                         help="Annotate latest snapshot with a change note.")
    ann.add_argument("directory", help="Notebook directory name.")
    ann.add_argument("note", help="Description of the deliberate change.")

    rep = sub.add_parser("report", help="Show changes correlated with votes.")
    rep.add_argument("--json", action="store_true", dest="as_json",
                     help="Output as JSON.")

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        return cmd_snapshot(dry_run=args.dry_run)
    elif args.command == "annotate":
        return cmd_annotate(args.directory, args.note)
    elif args.command == "report":
        return cmd_report(as_json=args.as_json)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
