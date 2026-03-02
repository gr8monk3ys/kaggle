#!/usr/bin/env python3
"""Publish pipeline for datasets with quality gates and draft awareness."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import dataset_usability
from kaggle_utils import kaggle_command, summarize_subprocess_error


BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
RESET = "\033[0m"

ROOT = Path(__file__).resolve().parent
DEFAULT_MIN_SCORE = 85


@dataclass(frozen=True)
class PublishCandidate:
    rel_path: str
    dir_path: Path
    dataset_ref: str | None
    score: int
    score_10: int
    tier: str
    live_state: str  # draft | live | unknown
    eligible: bool
    blocked_reasons: list[str]




def parse_live_refs_csv(raw_csv: str) -> set[str]:
    lines = raw_csv.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("ref,"):
            start_idx = idx
            break
    if start_idx is None:
        return set()

    refs: set[str] = set()
    csv_payload = "\n".join(lines[start_idx:]) + "\n"
    reader = csv.DictReader(io.StringIO(csv_payload))
    for row in reader:
        ref = str(row.get("ref", "")).strip().lower()
        if ref:
            refs.add(ref)
    return refs


def _run_dataset_list(args: list[str]) -> tuple[set[str] | None, str | None]:
    result = subprocess.run([*kaggle_command(), "datasets", "list", *args], capture_output=True, text=True)
    if result.returncode != 0:
        return None, summarize_subprocess_error(result.stdout, result.stderr)
    return parse_live_refs_csv(result.stdout), None


def fetch_live_refs(owner: str) -> tuple[set[str] | None, str | None]:
    owner = owner.strip().lower()
    if not owner:
        return None, "owner is required"

    refs: set[str] = set()
    errors: list[str] = []

    mine_refs, mine_err = _run_dataset_list(["--mine", "--csv"])
    if mine_refs is not None:
        refs.update(ref for ref in mine_refs if ref.startswith(f"{owner}/"))
    elif mine_err:
        errors.append(f"--mine: {mine_err}")

    search_refs, search_err = _run_dataset_list(["-s", owner, "--csv"])
    if search_refs is not None:
        refs.update(ref for ref in search_refs if ref.startswith(f"{owner}/"))
    elif search_err:
        errors.append(f"-s {owner}: {search_err}")

    if refs:
        return refs, None
    if errors:
        return None, "; ".join(errors)
    return None, "no dataset refs returned"


def classify_live_state(dataset_ref: str | None, live_refs: set[str] | None) -> str:
    if live_refs is None:
        return "unknown"
    if not dataset_ref:
        return "unknown"
    return "live" if dataset_ref.strip().lower() in live_refs else "draft"


def build_candidates(
    root: Path,
    *,
    min_score: int,
    live_refs: set[str] | None,
) -> list[PublishCandidate]:
    dirs = dataset_usability.discover_dataset_dirs(root)
    candidates: list[PublishCandidate] = []
    for ds_dir in dirs:
        scored = dataset_usability.score_dataset(ds_dir, root=root)
        live_state = classify_live_state(scored.dataset_ref, live_refs)

        blocked: list[str] = []
        if not scored.dataset_ref:
            blocked.append("missing dataset metadata id")
        if scored.score < min_score:
            blocked.append(f"score {scored.score} < min-score {min_score}")

        candidates.append(
            PublishCandidate(
                rel_path=str(ds_dir.relative_to(root)),
                dir_path=ds_dir,
                dataset_ref=scored.dataset_ref,
                score=scored.score,
                score_10=scored.score_10,
                tier=scored.tier,
                live_state=live_state,
                eligible=not blocked,
                blocked_reasons=blocked,
            )
        )
    return candidates


def infer_owner(candidates: list[PublishCandidate]) -> str | None:
    owners: dict[str, int] = {}
    for item in candidates:
        ref = (item.dataset_ref or "").strip().lower()
        if "/" not in ref:
            continue
        owner = ref.split("/", 1)[0]
        owners[owner] = owners.get(owner, 0) + 1
    if not owners:
        return None
    return max(owners.items(), key=lambda row: row[1])[0]


def select_targets(
    candidates: list[PublishCandidate],
    *,
    draft_only: bool,
    max_items: int,
) -> list[PublishCandidate]:
    selected = [
        item
        for item in candidates
        if item.eligible and (not draft_only or item.live_state == "draft")
    ]
    if max_items > 0:
        selected = selected[:max_items]
    return selected


def publish_dataset(candidate: PublishCandidate) -> tuple[bool, str]:
    cli = kaggle_command()
    version = subprocess.run(
        [
            *cli,
            "datasets",
            "version",
            "-p",
            str(candidate.dir_path),
            "-m",
            "publish pipeline update",
            "--dir-mode",
            "zip",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if version.returncode == 0:
        return True, "updated"

    create = subprocess.run(
        [*cli, "datasets", "create", "-p", str(candidate.dir_path), "--dir-mode", "zip"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if create.returncode == 0:
        return True, "created"

    return False, summarize_subprocess_error(
        version.stdout,
        version.stderr,
        create.stdout,
        create.stderr,
    )


def build_ui_sync_command(
    dataset_refs: list[str],
    *,
    headed: bool,
    timeout_ms: int,
    manual_login: bool,
) -> list[str]:
    script = ROOT / "pi-automation" / "scripts" / "dataset_metadata_sync.py"
    cmd = [
        sys.executable,
        str(script),
        "--apply",
        "--timeout-ms",
        str(timeout_ms),
    ]
    if headed:
        cmd.append("--headed")
    if not manual_login:
        cmd.append("--no-manual-login")
    for ref in dataset_refs:
        cmd.extend(["--dataset-ref", ref])
    return cmd


def run_ui_metadata_sync(
    dataset_refs: list[str],
    *,
    headed: bool,
    timeout_ms: int,
    manual_login: bool,
) -> tuple[bool, str, str, str]:
    cmd = build_ui_sync_command(
        dataset_refs,
        headed=headed,
        timeout_ms=timeout_ms,
        manual_login=manual_login,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    ok = result.returncode == 0
    detail = "updated" if ok else summarize_subprocess_error(result.stdout, result.stderr)
    return ok, detail, result.stdout, result.stderr


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish local datasets with score and draft gates.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--owner", default=None, help="Kaggle owner for draft/live lookup.")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE, help="Minimum local usability score.")
    parser.add_argument("--all", action="store_true", help="Include both live and draft datasets.")
    parser.add_argument("--max-items", type=int, default=0, help="Maximum items to process (0 means all).")
    parser.add_argument("--apply", action="store_true", help="Actually publish selected datasets.")
    parser.add_argument(
        "--sync-ui-metadata",
        action="store_true",
        help="After successful --apply publish, sync UI-only metadata fields via Playwright.",
    )
    parser.add_argument("--ui-sync-headed", action="store_true", help="Run UI sync browser in headed mode.")
    parser.add_argument("--ui-sync-timeout-ms", type=int, default=20000, help="Playwright timeout for UI sync.")
    parser.add_argument(
        "--ui-sync-manual-login",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow interactive Kaggle login during UI sync.",
    )
    parser.add_argument("--report-json", default=None, help="Optional output path for publish report JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_score < 0 or args.min_score > 100:
        raise SystemExit("--min-score must be between 0 and 100")
    if args.max_items < 0:
        raise SystemExit("--max-items must be >= 0")
    if args.ui_sync_timeout_ms < 1:
        raise SystemExit("--ui-sync-timeout-ms must be >= 1")
    if args.sync_ui_metadata and not args.apply:
        raise SystemExit("--sync-ui-metadata requires --apply")

    root = Path(args.root).resolve()
    print(f"{BLUE}=== Dataset Publish Pipeline ==={RESET}")

    # First pass to infer owner if not supplied.
    seed_candidates = build_candidates(root, min_score=args.min_score, live_refs=None)
    if not seed_candidates:
        raise SystemExit("No dataset directories found under datasets/")

    owner = (args.owner or infer_owner(seed_candidates) or "").strip().lower()
    draft_only = not args.all

    live_refs: set[str] | None = None
    if owner:
        live_refs, live_error = fetch_live_refs(owner)
        if live_error:
            print(f"{YELLOW}Warning:{RESET} live listing lookup failed for '{owner}': {live_error}")
            live_refs = None
        else:
            print(f"Live lookup: {len(live_refs)} refs for owner '{owner}'")
    elif draft_only:
        print(f"{YELLOW}Warning:{RESET} unable to infer owner; draft/live state unavailable.")

    candidates = build_candidates(root, min_score=args.min_score, live_refs=live_refs)

    if draft_only and live_refs is None:
        print(
            f"{RED}Cannot run in draft-only mode without live listing data.{RESET} "
            "Pass --all or provide valid --owner credentials."
        )
        return 1

    print("")
    print(f"{'Dataset':<34} {'Score':>5} {'State':<8} {'Eligible':<8} {'Ref'}")
    print("-" * 110)
    for item in candidates:
        eligible = "yes" if item.eligible else "no"
        ref = item.dataset_ref or "n/a"
        print(f"{item.rel_path[:34]:<34} {item.score:>5} {item.live_state:<8} {eligible:<8} {ref}")
        if item.blocked_reasons:
            print(f"  {YELLOW}blocked:{RESET} {'; '.join(item.blocked_reasons)}")

    targets = select_targets(candidates, draft_only=draft_only, max_items=args.max_items)
    print("")
    print(
        f"Selected targets: {len(targets)} "
        f"(mode={'draft-only' if draft_only else 'all'}, min-score={args.min_score})"
    )

    if not args.apply:
        print(f"{YELLOW}Dry run only.{RESET} Re-run with --apply to publish selected datasets.")
        if args.report_json:
            payload = {
                "root": str(root),
                "owner": owner,
                "mode": "draft-only" if draft_only else "all",
                "apply": False,
                "min_score": args.min_score,
                "selected_count": len(targets),
                "candidates": [
                    {
                        "rel_path": item.rel_path,
                        "dataset_ref": item.dataset_ref,
                        "score": item.score,
                        "score_10": item.score_10,
                        "tier": item.tier,
                        "live_state": item.live_state,
                        "eligible": item.eligible,
                        "blocked_reasons": item.blocked_reasons,
                    }
                    for item in candidates
                ],
                "results": [],
                "ui_sync": {"requested": bool(args.sync_ui_metadata), "status": "skipped"},
            }
            report_path = Path(args.report_json).resolve()
            write_json(report_path, payload)
            print(f"Report written: {report_path}")
        return 0

    if not targets:
        print("No eligible targets to publish.")
        if args.report_json:
            payload = {
                "root": str(root),
                "owner": owner,
                "mode": "draft-only" if draft_only else "all",
                "apply": True,
                "min_score": args.min_score,
                "selected_count": 0,
                "candidates": [
                    {
                        "rel_path": item.rel_path,
                        "dataset_ref": item.dataset_ref,
                        "score": item.score,
                        "score_10": item.score_10,
                        "tier": item.tier,
                        "live_state": item.live_state,
                        "eligible": item.eligible,
                        "blocked_reasons": item.blocked_reasons,
                    }
                    for item in candidates
                ],
                "results": [],
                "ui_sync": {"requested": bool(args.sync_ui_metadata), "status": "skipped", "reason": "no targets"},
            }
            report_path = Path(args.report_json).resolve()
            write_json(report_path, payload)
            print(f"Report written: {report_path}")
        return 0

    success = 0
    failed = 0
    results: list[dict] = []
    for item in targets:
        print(f"Publishing {item.rel_path}... ", end="", flush=True)
        ok, detail = publish_dataset(item)
        if ok:
            success += 1
            print(f"{GREEN}{detail}{RESET}")
        else:
            failed += 1
            print(f"{RED}failed{RESET}: {detail}")
        results.append(
            {
                "rel_path": item.rel_path,
                "dataset_ref": item.dataset_ref,
                "ok": ok,
                "detail": detail,
            }
        )

    print("")
    print(f"Publish results: {GREEN}{success} succeeded{RESET}, {RED}{failed} failed{RESET}")

    ui_sync_payload: dict = {"requested": bool(args.sync_ui_metadata), "status": "skipped"}
    ui_sync_failed = False
    if args.sync_ui_metadata:
        refs = sorted(
            {
                str(item["dataset_ref"]).strip().lower()
                for item in results
                if item["ok"] and item["dataset_ref"]
            }
        )
        if not refs:
            print("UI metadata sync skipped: no successful dataset refs to sync.")
            ui_sync_payload = {"requested": True, "status": "skipped", "reason": "no successful refs"}
        else:
            print(f"Running UI metadata sync for {len(refs)} dataset(s)...")
            ok, detail, sync_stdout, sync_stderr = run_ui_metadata_sync(
                refs,
                headed=args.ui_sync_headed,
                timeout_ms=args.ui_sync_timeout_ms,
                manual_login=args.ui_sync_manual_login,
            )
            ui_sync_payload = {
                "requested": True,
                "status": "ok" if ok else "failed",
                "detail": detail,
                "dataset_refs": refs,
            }
            if sync_stdout.strip():
                print(sync_stdout.strip())
            if sync_stderr.strip():
                print(sync_stderr.strip())
            if ok:
                print(f"{GREEN}UI metadata sync completed.{RESET}")
            else:
                print(f"{RED}UI metadata sync failed:{RESET} {detail}")
                ui_sync_failed = True

    if args.report_json:
        payload = {
            "root": str(root),
            "owner": owner,
            "mode": "draft-only" if draft_only else "all",
            "apply": True,
            "min_score": args.min_score,
            "selected_count": len(targets),
            "publish": {
                "success": success,
                "failed": failed,
            },
            "candidates": [
                {
                    "rel_path": item.rel_path,
                    "dataset_ref": item.dataset_ref,
                    "score": item.score,
                    "score_10": item.score_10,
                    "tier": item.tier,
                    "live_state": item.live_state,
                    "eligible": item.eligible,
                    "blocked_reasons": item.blocked_reasons,
                }
                for item in candidates
            ],
            "results": results,
            "ui_sync": ui_sync_payload,
        }
        report_path = Path(args.report_json).resolve()
        write_json(report_path, payload)
        print(f"Report written: {report_path}")

    return 0 if failed == 0 and not ui_sync_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
