#!/usr/bin/env python3
"""Repository-level preflight and safe live smoke checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.errors import CommandError


ROOT = Path(__file__).resolve().parents[2]

BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
RESET = "\033[0m"


@dataclass(frozen=True)
class Step:
    """One preflight/smoke step.

    ``run`` is an in-process callable returning an exit code; ``cmd`` is an argv
    vector for the few steps that genuinely need their own process (pytest, and
    the Playwright scripts). Exactly one is set.
    """

    name: str
    cmd: list[str] | None = None
    run: Callable[[], int] | None = None


def echo_step_header(name: str) -> None:
    print(f"{BLUE}=== {name} ==={RESET}")


def run_step(step: Step, *, cwd: Path) -> int:
    """Run one step and return its exit code.

    In-process steps are isolated here: a step that raises CommandError fails
    that step, not the whole run. Before this, a step could terminate preflight
    by raising SystemExit from inside its own main().
    """
    if step.run is not None:
        try:
            return int(step.run() or 0)
        except CommandError as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return exc.exit_code
    # Output is inherited, not captured: a long preflight should stream rather
    # than look hung. Capture was an artifact of the subprocess mechanism.
    return subprocess.run(step.cmd or [], cwd=str(cwd)).returncode


def run_steps(steps: list[Step], *, cwd: Path = ROOT) -> int:
    failures: list[tuple[str, int]] = []
    for step in steps:
        echo_step_header(step.name)
        code = run_step(step, cwd=cwd)
        if code == 0:
            print(f"{GREEN}[ok]{RESET} {step.name}")
        else:
            print(f"{RED}[fail]{RESET} {step.name} (exit {code})")
            failures.append((step.name, code))
        print("")

    if failures:
        joined = ", ".join(f"{name}={code}" for name, code in failures)
        print(f"{RED}Repo ops failed:{RESET} {joined}")
        return 1

    print(f"{GREEN}Repo ops passed.{RESET}")
    return 0


def build_preflight_steps(args: argparse.Namespace, deps: Deps) -> list[Step]:
    """Assemble preflight as a list of callables.

    Each step used to be an argv vector launched as its own interpreter, and the
    first of them shelled back into `bash manage.sh validate` — a third-level
    process re-entering the very module that spawned it. They are ordinary calls
    now, so a failing step reports as a failing step rather than as an exit code
    recovered from a grandchild.
    """
    from kaggle_portfolio import manage_commands
    from kaggle_portfolio.datasets import dataset_usability
    from kaggle_portfolio.ops import discussion_scheduler, medal_ops
    from kaggle_portfolio.quality import notebook_quality

    step_deps = deps.with_output_root(args.output_root)
    # --output-root is still forwarded explicitly: each module resolves it from its
    # own argparse, so handing them step_deps alone would silently write reports
    # into the repo instead of the scratch dir.
    out = ["--output-root", str(args.output_root)]
    today = ["--today", args.today] if args.today else []

    doctor_argv = [
        *out,
        *today,
        "doctor",
        "--max-stale-days",
        str(args.max_stale_days),
    ]
    if args.strict_doctor:
        doctor_argv.append("--strict")
    if args.require_kaggle:
        doctor_argv.append("--require-kaggle")
    for flag, value in (
        ("--kernels-csv", args.kernels_csv),
        ("--datasets-csv", args.datasets_csv),
        ("--competitions-csv", args.competitions_csv),
    ):
        if value:
            doctor_argv.extend([flag, value])

    quality_argv = [
        *out,
        *today,
        "--scope",
        "all",
        "--min-score",
        str(args.min_quality_score),
        "--fail-under-threshold",
    ]
    dataset_argv = [
        *out,
        *today,
        "--strict",
        "--fail-under",
        str(args.min_dataset_usability_score),
    ]
    draft_argv = [
        "--health-check",
        "--max-overdue-scheduled",
        str(args.max_overdue_scheduled),
        "--max-days-until-next-post",
        str(args.max_days_until_next_post),
        *today,
    ]

    steps = [
        Step("metadata-validate", run=lambda: manage_commands.cmd_validate([])),
        Step("doctor", run=lambda: medal_ops.main(doctor_argv, deps=step_deps)),
        Step(
            "notebook-quality",
            run=lambda: notebook_quality.main(quality_argv, deps=step_deps),
        ),
        Step(
            "dataset-usability",
            run=lambda: dataset_usability.main(dataset_argv, deps=step_deps),
        ),
        Step(
            "draft-ops",
            run=lambda: discussion_scheduler.main(draft_argv, deps=step_deps),
        ),
    ]
    if not args.no_pytest:
        steps.append(Step("pytest", [sys.executable, "-m", "pytest", "-q"]))
    return steps


def build_smoke_live_steps(args: argparse.Namespace, deps: Deps) -> list[Step]:
    """Live, non-mutating smoke checks.

    The discussion step stays a subprocess: it drives Playwright, which must not
    become reachable from a kaggle_portfolio import.
    """
    from kaggle_portfolio.campaigns import campaign_execute
    from kaggle_portfolio.datasets import dataset_publish_pipeline
    from kaggle_portfolio.ops import kaggle_auth_doctor

    auth_argv = ["--strict"]
    if args.owner:
        auth_argv.extend(["--expected-owner", args.owner])

    publish_argv = [
        "--max-items",
        str(args.limit),
        "--report-json",
        args.report_json,
    ]
    if args.owner:
        publish_argv.extend(["--owner", args.owner])
    if args.include_live_datasets:
        publish_argv.append("--all")

    campaign_argv = ["--dry-run", "--limit", str(args.limit), "--no-respect-schedule"]

    discussion_cmd = [
        sys.executable,
        str(deps.layout.pi_scripts / "discussion_post.py"),
        "--smoke-test",
    ]
    if args.check_discussion_login:
        discussion_cmd.append("--check-login")

    steps = [
        Step("auth-doctor", run=lambda: kaggle_auth_doctor.main(auth_argv, deps=deps))
    ]
    if not args.no_publish:
        steps.append(
            Step(
                "publish-datasets-dry-run",
                run=lambda: dataset_publish_pipeline.main(publish_argv, deps=deps),
            )
        )
    if not args.no_campaign:
        steps.append(
            Step(
                "campaign-execute-dry-run",
                run=lambda: campaign_execute.main(campaign_argv, deps=deps),
            )
        )
    if not args.no_discussion:
        steps.append(Step("discussion-post-smoke", discussion_cmd))
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repository operations for local preflight and live smoke checks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser(
        "preflight", help="Run the core repo gates in one command."
    )
    preflight.add_argument(
        "--output-root",
        default=None,
        help="Output root for generated reports (default: a scratch dir outside the repo).",
    )
    preflight.add_argument(
        "--today",
        default=None,
        help="Optional YYYY-MM-DD override for deterministic runs.",
    )
    preflight.add_argument(
        "--max-stale-days", type=int, default=30, help="Doctor stale-content threshold."
    )
    preflight.add_argument(
        "--strict-doctor",
        action="store_true",
        help="Fail preflight on doctor warnings, not only errors.",
    )
    preflight.add_argument(
        "--require-kaggle",
        action="store_true",
        help="Require live Kaggle access in doctor.",
    )
    preflight.add_argument(
        "--kernels-csv", default=None, help="Optional exported kernels CSV for doctor."
    )
    preflight.add_argument(
        "--datasets-csv",
        default=None,
        help="Optional exported datasets CSV for doctor.",
    )
    preflight.add_argument(
        "--competitions-csv",
        default=None,
        help="Optional exported competitions CSV for doctor.",
    )
    preflight.add_argument(
        "--min-quality-score", type=int, default=95, help="Notebook quality threshold."
    )
    preflight.add_argument(
        "--min-dataset-usability-score",
        type=int,
        default=85,
        help="Dataset usability threshold.",
    )
    preflight.add_argument(
        "--max-overdue-scheduled",
        type=int,
        default=0,
        help="Allowed overdue scheduled drafts.",
    )
    preflight.add_argument(
        "--max-days-until-next-post",
        type=int,
        default=14,
        help="Allowed gap to next scheduled post.",
    )
    preflight.add_argument(
        "--no-pytest", action="store_true", help="Skip the full pytest run."
    )

    smoke = sub.add_parser(
        "smoke-live",
        help="Safely exercise live Kaggle publish/post prerequisites without mutating state.",
    )
    smoke.add_argument("--owner", default=None, help="Expected Kaggle owner slug.")
    smoke.add_argument(
        "--limit", type=int, default=1, help="Max items to inspect in dry-run checks."
    )
    smoke.add_argument(
        "--report-json",
        default=None,
        help="Output path for dataset publish dry-run report.",
    )
    smoke.add_argument(
        "--include-live-datasets",
        action="store_true",
        help="Inspect all datasets, not only draft candidates.",
    )
    smoke.add_argument(
        "--no-publish", action="store_true", help="Skip dataset publish dry-run."
    )
    smoke.add_argument(
        "--no-campaign", action="store_true", help="Skip campaign queue dry-run."
    )
    smoke.add_argument(
        "--no-discussion",
        action="store_true",
        help="Skip discussion posting smoke test.",
    )
    smoke.add_argument(
        "--check-discussion-login",
        action="store_true",
        help="Open Playwright and verify Kaggle login without posting.",
    )
    return parser


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    deps = deps or Deps.resolve(today=getattr(args, "today", None))

    # Defaults that depend on the machine, not on argparse: the old literals were
    # POSIX-only paths baked into the parser.
    if getattr(args, "output_root", None) is None:
        args.output_root = str(deps.layout.scratch_dir("kaggle-preflight"))
    if getattr(args, "report_json", None) is None:
        args.report_json = str(
            deps.layout.scratch_dir("kaggle-live-smoke") / "dataset-publish.json"
        )

    if args.command == "preflight":
        return run_steps(build_preflight_steps(args, deps))
    if args.command == "smoke-live":
        return run_steps(build_smoke_live_steps(args, deps))
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
