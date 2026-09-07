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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import KaggleClient, KaggleError

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------


def log_path(deps: Deps) -> Path:
    return deps.layout.output_root / "metadata_ab_log.json"


def _load_log(deps: Deps) -> list[dict]:
    """Load the snapshot log from disk."""
    path = log_path(deps)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_log(deps: Deps, entries: list[dict]) -> None:
    """Write the snapshot log to disk."""
    path = log_path(deps)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------


def collect_metadata(deps: Deps) -> dict[str, dict]:
    """Scan every kernel-metadata.json and return a dict keyed by directory name.

    Includes the ``datasets/*`` explore notebooks: they earn votes like any other
    notebook, even though ``push`` treats their directories as datasets.
    """
    results: dict[str, dict] = {}
    for rel_dir in deps.layout.kernel_metadata_dirs(include_dataset_notebooks=True):
        rel = Path(rel_dir) / "kernel-metadata.json"
        meta_path = deps.layout.root / rel
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        dir_name = rel_dir
        results[dir_name] = {
            "id": meta.get("id", ""),
            "title": meta.get("title", ""),
            "keywords": meta.get("keywords", []),
            "enable_gpu": meta.get("enable_gpu", False),
            "dataset_sources": meta.get("dataset_sources", []),
            "competition_sources": meta.get("competition_sources", []),
        }
    return results


def fetch_vote_counts(client: KaggleClient) -> dict[str, int] | None:
    """Fetch vote counts for all owned kernels, keyed by slug.

    Returns ``None`` when Kaggle could not be reached, so callers can tell
    'fetch failed' from 'genuinely zero votes' instead of recording zeros.
    """
    try:
        return {kernel.slug: kernel.total_votes for kernel in client.my_kernels()}
    except KaggleError as exc:
        print(f"{YELLOW}Vote fetch failed{RESET}: {exc}", file=sys.stderr)
        return None


def _merge_votes(
    metadata: dict[str, dict], votes: dict[str, int] | None
) -> dict[str, dict]:
    """Merge vote counts into metadata entries, matching by slug.

    When ``votes`` is None (a failed fetch), record votes as None rather than 0
    so downstream reporting can distinguish 'unknown' from 'genuinely zero'.
    """
    for dir_name, entry in metadata.items():
        if votes is None:
            entry["votes"] = None
            continue
        kernel_id = entry.get("id", "")
        slug = kernel_id.split("/")[-1] if "/" in kernel_id else dir_name
        entry["votes"] = votes.get(slug, 0)
    return metadata


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_snapshot(deps: Deps, votes: dict[str, int] | None = None) -> int:
    """Take a snapshot of all metadata + votes and append to the log."""
    metadata = collect_metadata(deps)
    votes_unavailable = False
    if votes is None:
        votes = fetch_vote_counts(deps.client)
        if votes is None:
            votes_unavailable = True

    metadata = _merge_votes(metadata, votes)

    snapshot = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "notebooks": metadata,
        "annotation": None,
        "votes_available": not votes_unavailable,
    }

    if not deps.effects:
        print(
            f"{YELLOW}DRY RUN{RESET} — would write snapshot with "
            f"{len(metadata)} notebooks"
        )
        for name, entry in sorted(metadata.items()):
            print(
                f"  {name}: votes={entry.get('votes', '?')} "
                f"title={entry.get('title', '?')[:50]}"
            )
        return 0

    if votes_unavailable:
        print(
            f"{YELLOW}Warning{RESET}: vote counts unavailable (Kaggle CLI "
            "fetch failed); recording votes as unknown for this snapshot.",
            file=sys.stderr,
        )

    log = _load_log(deps)
    log.append(snapshot)
    _save_log(deps, log)
    total_votes = sum((e.get("votes") or 0) for e in metadata.values())
    print(
        f"{GREEN}Snapshot saved{RESET} — {len(metadata)} notebooks, "
        f"{total_votes} total votes"
    )
    return 0


def cmd_annotate(deps: Deps, directory: str, note: str) -> int:
    """Add an annotation to the most recent snapshot."""
    log = _load_log(deps)
    if not log:
        print(f"{RED}No snapshots found. Run 'snapshot' first.{RESET}")
        return 1

    latest = log[-1]
    existing_annotation = latest.get("annotation") or {}
    if isinstance(existing_annotation, str):
        existing_annotation = {"_general": existing_annotation}
    existing_annotation[directory] = note
    latest["annotation"] = existing_annotation
    _save_log(deps, log)

    print(f"{GREEN}Annotated{RESET} latest snapshot: {directory} → {note}")
    return 0


def cmd_report(deps: Deps, as_json: bool = False) -> int:
    """Show metadata changes correlated with vote deltas across snapshots."""
    log = _load_log(deps)
    if len(log) < 2:
        print(f"Need at least 2 snapshots for comparison. Currently have {len(log)}.")
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

            c_votes = c.get("votes")
            p_votes = p.get("votes")
            votes_known = c_votes is not None and p_votes is not None
            vote_delta = (c_votes - p_votes) if votes_known else 0
            title_changed = p.get("title") != c.get("title") and p.get("title")
            keywords_changed = (
                set(p.get("keywords", [])) != set(c.get("keywords", []))
                and p.get("keywords") is not None
            )

            if (votes_known and vote_delta != 0) or title_changed or keywords_changed:
                entry = {
                    "timestamp": ts,
                    "notebook": name,
                    "vote_delta": vote_delta if votes_known else None,
                    "votes_now": c_votes,
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
        if delta is None:
            delta_color = YELLOW
            delta_str = "?"
        else:
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
        votes_str = f"{votes:>6}" if votes is not None else f"{'?':>6}"
        print(f"{name:<35} {votes_str} {delta_color}{delta_str:>7}{RESET}  {desc}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track notebook metadata changes vs vote deltas."
    )
    sub = parser.add_subparsers(dest="command")

    snap = sub.add_parser("snapshot", help="Take a metadata + vote snapshot.")
    snap.add_argument("--dry-run", action="store_true", help="Preview without writing.")

    ann = sub.add_parser(
        "annotate", help="Annotate latest snapshot with a change note."
    )
    ann.add_argument("directory", help="Notebook directory name.")
    ann.add_argument("note", help="Description of the deliberate change.")

    rep = sub.add_parser("report", help="Show changes correlated with votes.")
    rep.add_argument(
        "--json", action="store_true", dest="as_json", help="Output as JSON."
    )

    args = parser.parse_args(argv)
    deps = deps or Deps.resolve(effects=not getattr(args, "dry_run", False))

    if args.command == "snapshot":
        return cmd_snapshot(deps)
    elif args.command == "annotate":
        return cmd_annotate(deps, args.directory, args.note)
    elif args.command == "report":
        return cmd_report(deps, as_json=args.as_json)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
