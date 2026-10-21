#!/usr/bin/env python3
"""Medal operations tooling for Kaggle Grandmaster execution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_portfolio.shared.clock import resolve_today
from kaggle_portfolio.shared import reports
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import (
    AUTH_REJECTED_HINT,
    KaggleAuthRejected,
    KaggleClient,
    KaggleError,
)
from kaggle_portfolio.shared.errors import CommandError
from kaggle_portfolio.ops.tracker import (
    apply_tracker_sync,
    extract_first_int,
    parse_active_competitions,
    parse_last_updated,
    parse_progress_metrics,
    parse_tier_requirements,
)


DEFAULT_TRACKER_PATH = Path("docs/reports/grandmaster-tracker.md")
DEFAULT_OUTPUT_ROOT = Path("medal_ops")
UTC = getattr(datetime, "UTC", timezone.utc)


def find_key(keys: list[str], predicates: list[str]) -> str | None:
    for wanted in predicates:
        for key in keys:
            if wanted in key.lower():
                return key
    return None


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
            raise CommandError(
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
            raise CommandError(
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


def fetch_live_kaggle_metrics(client: KaggleClient) -> dict[str, Any]:
    if not client.available():
        raise CommandError(
            "kaggle CLI not found. Install and authenticate it, then rerun sync."
        )

    # The client paginates and strips Kaggle's banner lines; the metric
    # parsers below work on the raw CLI rows it keeps.
    kernels = client.my_kernels()
    datasets = client.my_datasets()
    entered = client.entered_competitions()

    kernels_rows = [k.raw for k in kernels]
    datasets_rows = [d.raw for d in datasets]
    entered_rows = [c.raw for c in entered]
    kernels_columns = list(kernels_rows[0]) if kernels_rows else []
    datasets_columns = list(datasets_rows[0]) if datasets_rows else []

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


def build_snapshot(content: str, today: date) -> dict[str, Any]:
    tracker_last_updated = parse_last_updated(content)
    stale_days = (today - tracker_last_updated).days if tracker_last_updated else None

    requirements = parse_tier_requirements(content)
    competitions = parse_progress_metrics(content, "Competitions")
    notebooks = parse_progress_metrics(content, "Notebooks")
    datasets = parse_progress_metrics(content, "Datasets")
    discussion = parse_progress_metrics(content, "Discussion")
    active_competitions = parse_active_competitions(content, today)

    def gap(
        current: dict[str, Any], req: dict[str, int], key: str, req_key: str
    ) -> int | None:
        if key not in current or req_key not in req:
            return None
        return max(0, req[req_key] - int(current[key]))

    category_summary = {
        "competitions": {
            **competitions,
            "gold_goal": requirements.get("competitions", {}).get("grandmaster_gold"),
            "gold_gap": gap(
                competitions,
                requirements.get("competitions", {}),
                "gold",
                "grandmaster_gold",
            ),
            "expert_bronze_goal": requirements.get("competitions", {}).get(
                "expert_bronze"
            ),
            "expert_bronze_gap": gap(
                competitions,
                requirements.get("competitions", {}),
                "bronze",
                "expert_bronze",
            ),
        },
        "notebooks": {
            **notebooks,
            "gold_goal": requirements.get("notebooks", {}).get("grandmaster_gold"),
            "gold_gap": gap(
                notebooks, requirements.get("notebooks", {}), "gold", "grandmaster_gold"
            ),
            "expert_bronze_goal": requirements.get("notebooks", {}).get(
                "expert_bronze"
            ),
            "expert_bronze_gap": gap(
                notebooks, requirements.get("notebooks", {}), "bronze", "expert_bronze"
            ),
        },
        "datasets": {
            **datasets,
            "gold_goal": requirements.get("datasets", {}).get("grandmaster_gold"),
            "gold_gap": gap(
                datasets, requirements.get("datasets", {}), "gold", "grandmaster_gold"
            ),
            "expert_bronze_goal": requirements.get("datasets", {}).get("expert_bronze"),
            "expert_bronze_gap": gap(
                datasets, requirements.get("datasets", {}), "bronze", "expert_bronze"
            ),
        },
        "discussion": {
            **discussion,
            "gold_goal": requirements.get("discussion", {}).get("grandmaster_gold"),
            "gold_gap": gap(
                discussion,
                requirements.get("discussion", {}),
                "gold",
                "grandmaster_gold",
            ),
            "total_goal": requirements.get("discussion", {}).get("grandmaster_total"),
            "total_gap": gap(
                discussion,
                requirements.get("discussion", {}),
                "total_posts",
                "grandmaster_total",
            ),
            "expert_bronze_goal": requirements.get("discussion", {}).get(
                "expert_bronze"
            ),
            "expert_bronze_gap": gap(
                discussion,
                requirements.get("discussion", {}),
                "bronze",
                "expert_bronze",
            ),
        },
    }

    active_competitions_json = [
        {
            "competition": deadline.competition,
            "deadline_raw": deadline.deadline_raw,
            "deadline_date": deadline.deadline_date.isoformat()
            if deadline.deadline_date
            else None,
            "days_to_deadline": deadline.days_to_deadline,
            "teams": deadline.teams,
            "difficulty": deadline.difficulty,
            "strategy": deadline.strategy,
        }
        for deadline in active_competitions
    ]

    return {
        "generated_on": today.isoformat(),
        "tracker_last_updated": tracker_last_updated.isoformat()
        if tracker_last_updated
        else None,
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


def delta(
    current: dict[str, Any], previous: dict[str, Any] | None, path: tuple[str, ...]
) -> int | None:
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


def top_actions(snapshot: dict[str, Any]) -> list[str]:
    """Next steps that can earn medals: an honest tracker, real competitions, better work."""
    actions: list[str] = []
    stale_days = snapshot.get("tracker_stale_days")
    if isinstance(stale_days, int) and stale_days > 7:
        actions.append(
            f"Run `./manage.sh sync` to refresh the tracker ({stale_days} days stale)."
        )

    upcoming = [
        c
        for c in snapshot.get("active_competitions", [])
        if isinstance(c.get("days_to_deadline"), int) and c["days_to_deadline"] >= 0
    ]
    if not upcoming:
        actions.append(
            "Enter a Featured or Research competition; those are the ones that award medals."
        )

    actions.append(
        "Publish one notebook or dataset with a clear, reproducible result "
        "before starting another."
    )
    return actions


def _fmt_delta(value: int | None) -> str:
    if value is None:
        return "—"
    return f"+{value}" if value > 0 else str(value)


def generate_digest(snapshots: list[dict[str, Any]]) -> str:
    """Compose a one-message daily Grandmaster digest (Markdown) from snapshot history.

    Pure function: takes the chronological snapshot list, returns a Markdown string.
    """
    if not snapshots:
        return "No snapshots available yet — run `medal_ops sync` first."

    current = snapshots[-1]
    previous = snapshots[-2] if len(snapshots) >= 2 else None
    lines = [
        f"\U0001f4ca *Grandmaster digest — {current.get('generated_on', '?')}*",
        "",
    ]

    if previous is None:
        lines.append("_First snapshot recorded — deltas start next run._")
    else:
        movers = [
            ("Comp entered", ("categories", "competitions", "entered")),
            ("Notebook votes", ("categories", "notebooks", "total_votes")),
            ("Dataset votes", ("categories", "datasets", "total_votes")),
            ("Discussion posts", ("categories", "discussion", "total_posts")),
        ]
        bits = []
        for label, path in movers:
            d = delta(current, previous, path)
            if d:
                bits.append(f"{label} {_fmt_delta(d)}")
        lines.append(
            "*Since last snapshot:* " + (", ".join(bits) if bits else "no change")
        )

    upcoming = [
        c
        for c in current.get("active_competitions", [])
        if isinstance(c.get("days_to_deadline"), int) and c["days_to_deadline"] >= 0
    ]
    if upcoming:
        nearest = min(upcoming, key=lambda c: c["days_to_deadline"])
        lines.append(
            f"*Nearest deadline:* {nearest.get('competition', '?')} "
            f"in {nearest['days_to_deadline']}d"
        )
    else:
        lines.append("*Nearest deadline:* none tracked")

    actions = top_actions(current)
    if actions:
        lines.append(f"*Top action today:* {actions[0]}")

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
        lines.append(
            "- Competition entries pulled: n/a (no entered column found in competitions list output)"
        )

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


def has_kaggle_credentials(client: KaggleClient) -> tuple[bool, list[str]]:
    """Whether Kaggle credentials are resolvable, and where they came from."""
    state = client.credentials()
    return bool(state), list(state.sources)


def run_preflight_checks(
    client: KaggleClient,
    tracker_path: Path,
    output_root: Path,
    today: date,
    require_kaggle: bool,
    max_stale_days: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []
    snapshot: dict[str, Any] | None = None

    if not tracker_path.exists():
        errors.append(f"Tracker file not found: {tracker_path}")
    else:
        try:
            tracker_content = tracker_path.read_text(encoding="utf-8")
            snapshot = build_snapshot(tracker_content, today)
            infos.append(f"Tracker readable: {tracker_path}")

            requirements = parse_tier_requirements(tracker_content)
            if not requirements:
                errors.append(
                    "Tier requirements table could not be parsed from tracker."
                )

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

    kaggle_cli_available = client.available()
    if kaggle_cli_available:
        infos.append("kaggle CLI available.")
    elif require_kaggle:
        errors.append("kaggle CLI not found.")
    else:
        warnings.append("kaggle CLI not found; live sync is unavailable.")

    creds_ok, creds_paths = has_kaggle_credentials(client)
    if creds_ok:
        joined = ", ".join(str(path) for path in creds_paths)
        infos.append(f"Kaggle credentials found: {joined}")
    elif require_kaggle:
        errors.append(
            "Kaggle credentials not found (`~/.kaggle/kaggle.json` or local `kaggle.json`)."
        )
    else:
        warnings.append("Kaggle credentials not found for live sync.")

    # Present is not the same as accepted: an expired key passed every check
    # above while the live sync behind it failed with a 401 for weeks. When
    # the caller requires Kaggle, ask Kaggle, using the call sync makes first.
    if require_kaggle and kaggle_cli_available and creds_ok:
        try:
            client.my_kernels()
            infos.append("Kaggle accepted the credentials (kernels list --mine).")
        except KaggleAuthRejected as exc:
            errors.append(str(exc))
        except KaggleError as exc:
            warnings.append(f"Kaggle credential check could not complete: {exc}")

    return {
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "snapshot": snapshot,
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
    if any("kaggle CLI not found" in item for item in warnings + errors):
        recommended.append("Install and authenticate the Kaggle CLI.")
    if any("credentials not found" in item.lower() for item in warnings + errors):
        recommended.append(
            "Add Kaggle credentials to `~/.kaggle/kaggle.json` (chmod 600)."
        )
    if any(item.startswith(AUTH_REJECTED_HINT) for item in errors):
        recommended.append(AUTH_REJECTED_HINT)

    if not recommended and status == "READY":
        recommended.append("Preflight passed. Run `./manage.sh sync --dry-run`.")
    elif not recommended:
        recommended.append(
            "Resolve listed issues, then rerun `./manage.sh doctor --strict`."
        )

    lines.append("## Recommended Next Step")
    lines.append("")
    lines.extend(f"- {item}" for item in recommended)
    lines.append("")

    return "\n".join(lines)


def add_shared_cli_args(
    parser: argparse.ArgumentParser, *, is_subparser: bool = False
) -> None:
    # The shared flags are registered on both the top-level parser and each
    # subparser so they are accepted before OR after the subcommand. On the
    # subparsers we suppress the defaults: without this, a subparser default
    # would overwrite a value already parsed from before the subcommand (e.g.
    # ``medal_ops --output-root X digest`` would silently reset to the
    # default). With SUPPRESS the subparser only sets these attributes when the
    # flag is actually given after the subcommand; otherwise the top-level
    # parser's value (given or default) is preserved.
    tracker_default = argparse.SUPPRESS if is_subparser else str(DEFAULT_TRACKER_PATH)
    output_default = argparse.SUPPRESS if is_subparser else str(DEFAULT_OUTPUT_ROOT)
    today_default = argparse.SUPPRESS if is_subparser else None
    parser.add_argument(
        "--tracker",
        default=tracker_default,
        help="Path to grandmaster tracker markdown file.",
    )
    parser.add_argument(
        "--output-root",
        default=output_default,
        help="Output directory for history and reports.",
    )
    parser.add_argument(
        "--today", default=today_default, help="Override date in YYYY-MM-DD format."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle medal operations CLI.")
    add_shared_cli_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync tracker metrics from live Kaggle CLI data and record a history snapshot.",
    )
    add_shared_cli_args(sync_parser, is_subparser=True)
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate sync report without writing the tracker or a snapshot.",
    )
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run preflight checks for tracker health and environment readiness.",
    )
    add_shared_cli_args(doctor_parser, is_subparser=True)
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
    digest_parser = subparsers.add_parser(
        "digest", help="Print a one-message daily Grandmaster digest to stdout."
    )
    add_shared_cli_args(digest_parser, is_subparser=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parse_args(argv)
    deps = deps or Deps.resolve(
        output_root=getattr(args, "output_root", None),
        today=getattr(args, "today", None),
        effects=not bool(getattr(args, "dry_run", False)),
    )
    today = resolve_today(args.today)
    output_root = Path(args.output_root)
    tracker_path = Path(args.tracker)
    history_dir = output_root / "history"

    if args.command == "doctor":
        if int(args.max_stale_days) < 0:
            raise CommandError("--max-stale-days must be >= 0")
        checks = run_preflight_checks(
            client=deps.client,
            tracker_path=tracker_path,
            output_root=output_root,
            today=today,
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
        deps.emitter.emit(reports.DOCTOR, report)

        errors = checks["errors"]
        warnings = checks["warnings"]
        print(f"Summary: {len(errors)} error(s), {len(warnings)} warning(s)")
        # The report file is not always kept (CI redirects stdout to a log and
        # uploads only that), so the reasons go to stderr as well as the report.
        for item in errors:
            print(f"doctor error: {item}", file=sys.stderr)
        for item in warnings:
            print(f"doctor warning: {item}", file=sys.stderr)

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
        raise CommandError(f"Tracker file not found: {tracker_path}")

    content = tracker_path.read_text(encoding="utf-8")

    if args.command == "digest":
        snapshots = load_all_snapshots(history_dir)
        if not snapshots:
            snapshots = [build_snapshot(content, today)]
        print(generate_digest(snapshots))
        return 0

    if args.command == "sync":
        live = fetch_live_kaggle_metrics(deps.client)
        updated_content, changes = apply_tracker_sync(content, today, live)

        # deps.effects, not args.dry_run: the dispatcher already resolved the
        # flag, and a second reading of it is how the two drifted apart before.
        snapshot_path = None
        if deps.effects:
            if updated_content != content:
                tracker_path.write_text(updated_content, encoding="utf-8")
            # The snapshot history is what `digest` reads its deltas from.
            snapshot_path = write_snapshot(
                history_dir, build_snapshot(updated_content, today)
            )

        report = generate_sync_markdown(
            tracker_path, today, live, changes, args.dry_run
        )
        deps.emitter.emit(reports.SYNC, report)

        if args.dry_run:
            print("Dry-run mode: tracker and snapshot history were not modified.")
        elif updated_content != content:
            print(f"Tracker updated: {tracker_path}")
        else:
            print("Tracker already up to date with pulled metrics.")
        if snapshot_path is not None:
            print(f"Snapshot written: {snapshot_path}")
        return 0

    raise CommandError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
