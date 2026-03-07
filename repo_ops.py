#!/usr/bin/env python3
"""Repository-level preflight and safe live smoke checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent

BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
RESET = "\033[0m"


@dataclass(frozen=True)
class Step:
    name: str
    cmd: list[str]


def echo_step_header(name: str) -> None:
    print(f"{BLUE}=== {name} ==={RESET}")


def run_steps(steps: list[Step], *, cwd: Path = ROOT) -> int:
    failures: list[tuple[str, int]] = []
    for step in steps:
        echo_step_header(step.name)
        result = subprocess.run(step.cmd, cwd=str(cwd), capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode == 0:
            print(f"{GREEN}[ok]{RESET} {step.name}")
        else:
            print(f"{RED}[fail]{RESET} {step.name} (exit {result.returncode})")
            failures.append((step.name, result.returncode))
        print("")

    if failures:
        joined = ", ".join(f"{name}={code}" for name, code in failures)
        print(f"{RED}Repo ops failed:{RESET} {joined}")
        return 1

    print(f"{GREEN}Repo ops passed.{RESET}")
    return 0


def build_preflight_steps(args: argparse.Namespace) -> list[Step]:
    output_root = str(Path(args.output_root))

    doctor_cmd = [sys.executable, str(ROOT / "medal_ops.py"), "--output-root", output_root]
    if args.today:
        doctor_cmd.extend(["--today", args.today])
    doctor_cmd.extend(["doctor", "--max-stale-days", str(args.max_stale_days)])
    if args.strict_doctor:
        doctor_cmd.append("--strict")
    if args.require_kaggle:
        doctor_cmd.append("--require-kaggle")
    if args.kernels_csv:
        doctor_cmd.extend(["--kernels-csv", args.kernels_csv])
    if args.datasets_csv:
        doctor_cmd.extend(["--datasets-csv", args.datasets_csv])
    if args.competitions_csv:
        doctor_cmd.extend(["--competitions-csv", args.competitions_csv])

    quality_cmd = [sys.executable, str(ROOT / "notebook_quality.py"), "--output-root", output_root]
    if args.today:
        quality_cmd.extend(["--today", args.today])
    quality_cmd.extend(
        [
            "--scope",
            "all",
            "--min-score",
            str(args.min_quality_score),
            "--fail-under-threshold",
        ]
    )

    dataset_cmd = [sys.executable, str(ROOT / "dataset_usability.py"), "--output-root", output_root]
    if args.today:
        dataset_cmd.extend(["--today", args.today])
    dataset_cmd.extend(["--strict", "--fail-under", str(args.min_dataset_usability_score)])

    draft_cmd = [
        sys.executable,
        str(ROOT / "discussion_scheduler.py"),
        "--health-check",
        "--max-overdue-scheduled",
        str(args.max_overdue_scheduled),
        "--max-days-until-next-post",
        str(args.max_days_until_next_post),
    ]

    steps = [
        Step("metadata-validate", ["bash", str(ROOT / "manage.sh"), "validate"]),
        Step("doctor", doctor_cmd),
        Step("notebook-quality", quality_cmd),
        Step("dataset-usability", dataset_cmd),
        Step("draft-ops", draft_cmd),
    ]
    if not args.no_pytest:
        steps.append(Step("pytest", [sys.executable, "-m", "pytest", "-q"]))
    return steps


def build_smoke_live_steps(args: argparse.Namespace) -> list[Step]:
    auth_cmd = [sys.executable, str(ROOT / "kaggle_auth_doctor.py"), "--strict"]
    if args.owner:
        auth_cmd.extend(["--expected-owner", args.owner])

    publish_cmd = [
        sys.executable,
        str(ROOT / "dataset_publish_pipeline.py"),
        "--max-items",
        str(args.limit),
        "--report-json",
        args.report_json,
    ]
    if args.owner:
        publish_cmd.extend(["--owner", args.owner])
    if args.include_live_datasets:
        publish_cmd.append("--all")

    campaign_cmd = [
        sys.executable,
        str(ROOT / "campaign_execute.py"),
        "--dry-run",
        "--limit",
        str(args.limit),
        "--no-respect-schedule",
    ]

    discussion_cmd = [
        sys.executable,
        str(ROOT / "pi-automation" / "scripts" / "discussion_post.py"),
        "--smoke-test",
    ]
    if args.check_discussion_login:
        discussion_cmd.append("--check-login")

    steps = [Step("auth-doctor", auth_cmd)]
    if not args.no_publish:
        steps.append(Step("publish-datasets-dry-run", publish_cmd))
    if not args.no_campaign:
        steps.append(Step("campaign-execute-dry-run", campaign_cmd))
    if not args.no_discussion:
        steps.append(Step("discussion-post-smoke", discussion_cmd))
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repository operations for local preflight and live smoke checks.")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="Run the core repo gates in one command.")
    preflight.add_argument("--output-root", default="/tmp/kaggle-preflight", help="Output root for generated reports.")
    preflight.add_argument("--today", default=None, help="Optional YYYY-MM-DD override for deterministic runs.")
    preflight.add_argument("--max-stale-days", type=int, default=30, help="Doctor stale-content threshold.")
    preflight.add_argument("--strict-doctor", action="store_true", help="Fail preflight on doctor warnings, not only errors.")
    preflight.add_argument("--require-kaggle", action="store_true", help="Require live Kaggle access in doctor.")
    preflight.add_argument("--kernels-csv", default=None, help="Optional exported kernels CSV for doctor.")
    preflight.add_argument("--datasets-csv", default=None, help="Optional exported datasets CSV for doctor.")
    preflight.add_argument("--competitions-csv", default=None, help="Optional exported competitions CSV for doctor.")
    preflight.add_argument("--min-quality-score", type=int, default=95, help="Notebook quality threshold.")
    preflight.add_argument("--min-dataset-usability-score", type=int, default=85, help="Dataset usability threshold.")
    preflight.add_argument("--max-overdue-scheduled", type=int, default=0, help="Allowed overdue scheduled drafts.")
    preflight.add_argument("--max-days-until-next-post", type=int, default=14, help="Allowed gap to next scheduled post.")
    preflight.add_argument("--no-pytest", action="store_true", help="Skip the full pytest run.")

    smoke = sub.add_parser("smoke-live", help="Safely exercise live Kaggle publish/post prerequisites without mutating state.")
    smoke.add_argument("--owner", default=None, help="Expected Kaggle owner slug.")
    smoke.add_argument("--limit", type=int, default=1, help="Max items to inspect in dry-run checks.")
    smoke.add_argument(
        "--report-json",
        default="/tmp/kaggle-live-smoke-dataset-publish.json",
        help="Output path for dataset publish dry-run report.",
    )
    smoke.add_argument("--include-live-datasets", action="store_true", help="Inspect all datasets, not only draft candidates.")
    smoke.add_argument("--no-publish", action="store_true", help="Skip dataset publish dry-run.")
    smoke.add_argument("--no-campaign", action="store_true", help="Skip campaign queue dry-run.")
    smoke.add_argument("--no-discussion", action="store_true", help="Skip discussion posting smoke test.")
    smoke.add_argument(
        "--check-discussion-login",
        action="store_true",
        help="Open Playwright and verify Kaggle login without posting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preflight":
        return run_steps(build_preflight_steps(args))
    if args.command == "smoke-live":
        return run_steps(build_smoke_live_steps(args))
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
