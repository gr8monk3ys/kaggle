from __future__ import annotations

import sys
from types import SimpleNamespace

import repo_ops


def test_build_preflight_steps_includes_expected_defaults():
    parser = repo_ops.build_parser()
    args = parser.parse_args(["preflight"])

    steps = repo_ops.build_preflight_steps(args)

    assert [step.name for step in steps] == [
        "metadata-validate",
        "doctor",
        "notebook-quality",
        "dataset-usability",
        "draft-ops",
        "pytest",
    ]
    assert steps[0].cmd == ["bash", str(repo_ops.ROOT / "manage.sh"), "validate"]
    assert steps[1].cmd[:4] == [sys.executable, str(repo_ops.ROOT / "medal_ops.py"), "--output-root", "/tmp/kaggle-preflight"]
    assert "--strict" not in steps[1].cmd
    assert "--fail-under-threshold" in steps[2].cmd
    assert "--fail-under" in steps[3].cmd


def test_build_preflight_steps_can_skip_pytest_and_pass_csv_args():
    parser = repo_ops.build_parser()
    args = parser.parse_args(
        [
            "preflight",
            "--no-pytest",
            "--today",
            "2026-03-07",
            "--kernels-csv",
            "kernels.csv",
            "--datasets-csv",
            "datasets.csv",
            "--competitions-csv",
            "competitions.csv",
            "--strict-doctor",
            "--require-kaggle",
        ]
    )

    steps = repo_ops.build_preflight_steps(args)
    assert [step.name for step in steps][-1] == "draft-ops"
    doctor_cmd = steps[1].cmd
    assert "--today" in doctor_cmd and "2026-03-07" in doctor_cmd
    assert "--strict" in doctor_cmd
    assert "--require-kaggle" in doctor_cmd
    assert "--kernels-csv" in doctor_cmd and "kernels.csv" in doctor_cmd
    assert "--datasets-csv" in doctor_cmd and "datasets.csv" in doctor_cmd
    assert "--competitions-csv" in doctor_cmd and "competitions.csv" in doctor_cmd


def test_build_smoke_live_steps_include_expected_checks():
    parser = repo_ops.build_parser()
    args = parser.parse_args(["smoke-live", "--owner", "lorenzoscaturchio", "--check-discussion-login"])

    steps = repo_ops.build_smoke_live_steps(args)

    assert [step.name for step in steps] == [
        "auth-doctor",
        "publish-datasets-dry-run",
        "campaign-execute-dry-run",
        "discussion-post-smoke",
    ]
    assert "--expected-owner" in steps[0].cmd
    assert "--report-json" in steps[1].cmd
    assert "--dry-run" in steps[2].cmd
    assert "--smoke-test" in steps[3].cmd
    assert "--check-login" in steps[3].cmd


def test_build_smoke_live_steps_respect_skip_flags():
    parser = repo_ops.build_parser()
    args = parser.parse_args(["smoke-live", "--no-publish", "--no-campaign"])

    steps = repo_ops.build_smoke_live_steps(args)

    assert [step.name for step in steps] == ["auth-doctor", "discussion-post-smoke"]


def test_run_steps_returns_failure_when_any_step_fails(monkeypatch):
    results = iter(
        [
            SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            SimpleNamespace(returncode=2, stdout="", stderr="boom"),
        ]
    )

    monkeypatch.setattr(repo_ops.subprocess, "run", lambda *args, **kwargs: next(results))

    rc = repo_ops.run_steps(
        [
            repo_ops.Step("first", ["echo", "first"]),
            repo_ops.Step("second", ["echo", "second"]),
        ]
    )

    assert rc == 1
