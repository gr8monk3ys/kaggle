"""Tests for preflight and smoke-live.

These used to assert on argv vectors — that a flag string appeared somewhere in a
list destined for a subprocess. Since the steps are in-process calls now, they
assert on what actually happens instead: which steps run, that a failing step
fails only itself, and that a step raising CommandError is reported rather than
taking the whole run down.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kaggle_portfolio.ops import repo_ops
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.errors import CommandError
from kaggle_portfolio.shared.kaggle_client import FakeKaggleClient


def _args(*argv: str):
    return repo_ops.build_parser().parse_args(list(argv))


def _deps(tmp_path) -> Deps:
    return Deps.for_test(tmp_path, client=FakeKaggleClient())


class TestPreflightComposition:
    def test_default_step_order(self, tmp_path):
        steps = repo_ops.build_preflight_steps(_args("preflight"), _deps(tmp_path))
        assert [s.name for s in steps] == [
            "metadata-validate",
            "doctor",
            "notebook-quality",
            "dataset-usability",
            "draft-ops",
            "pytest",
        ]

    def test_metadata_validate_is_an_in_process_call_not_a_bash_round_trip(
        self, tmp_path
    ):
        steps = repo_ops.build_preflight_steps(_args("preflight"), _deps(tmp_path))
        validate = steps[0]
        assert validate.run is not None
        assert validate.cmd is None

    def test_only_pytest_remains_a_subprocess(self, tmp_path):
        steps = repo_ops.build_preflight_steps(_args("preflight"), _deps(tmp_path))
        spawning = [s.name for s in steps if s.cmd is not None]
        assert spawning == ["pytest"]

    def test_no_pytest_drops_the_only_subprocess(self, tmp_path):
        steps = repo_ops.build_preflight_steps(
            _args("preflight", "--no-pytest"), _deps(tmp_path)
        )
        assert [s.name for s in steps][-1] == "draft-ops"
        assert all(s.cmd is None for s in steps)

    def test_reports_go_to_the_requested_output_root(self, tmp_path):
        out = tmp_path / "scratch"
        args = _args("preflight", "--no-pytest", "--output-root", str(out))
        repo_ops.build_preflight_steps(args, _deps(tmp_path))
        # The steps carry it via forwarded argv; assert the parser honoured it
        # rather than falling back to the machine default.
        assert args.output_root == str(out)

    def test_output_root_default_is_outside_the_repo(self, tmp_path):
        deps = _deps(tmp_path)
        scratch = deps.layout.scratch_dir("kaggle-preflight")
        assert not str(scratch).startswith(str(deps.layout.root))


class TestSmokeLiveComposition:
    def test_includes_expected_checks(self, tmp_path):
        args = _args(
            "smoke-live", "--owner", "lorenzoscaturchio", "--check-discussion-login"
        )
        args.report_json = str(tmp_path / "r.json")
        steps = repo_ops.build_smoke_live_steps(args, _deps(tmp_path))
        assert [s.name for s in steps] == [
            "auth-doctor",
            "publish-datasets-dry-run",
            "campaign-execute-dry-run",
            "discussion-post-smoke",
        ]

    def test_discussion_smoke_stays_a_subprocess(self, tmp_path):
        """Playwright must not become reachable from a kaggle_portfolio import."""
        args = _args("smoke-live")
        args.report_json = str(tmp_path / "r.json")
        steps = repo_ops.build_smoke_live_steps(args, _deps(tmp_path))
        discussion = [s for s in steps if s.name == "discussion-post-smoke"][0]
        assert discussion.cmd is not None
        assert discussion.cmd[-1] == "--smoke-test"

    def test_respects_skip_flags(self, tmp_path):
        args = _args("smoke-live", "--no-publish", "--no-campaign")
        args.report_json = str(tmp_path / "r.json")
        steps = repo_ops.build_smoke_live_steps(args, _deps(tmp_path))
        assert [s.name for s in steps] == ["auth-doctor", "discussion-post-smoke"]


class TestStepExecution:
    def test_a_failing_step_fails_the_run(self):
        rc = repo_ops.run_steps(
            [
                repo_ops.Step("first", run=lambda: 0),
                repo_ops.Step("second", run=lambda: 2),
            ]
        )
        assert rc == 1

    def test_all_passing_steps_succeed(self):
        assert repo_ops.run_steps([repo_ops.Step("only", run=lambda: 0)]) == 0

    def test_a_step_raising_command_error_fails_only_that_step(self, capsys):
        """The point of the change: one bad step no longer kills the run."""
        ran_after = []

        def boom():
            raise CommandError("tracker is missing", exit_code=3)

        rc = repo_ops.run_steps(
            [
                repo_ops.Step("explodes", run=boom),
                repo_ops.Step("later", run=lambda: ran_after.append(True) or 0),
            ]
        )
        out = capsys.readouterr()
        assert rc == 1
        assert ran_after == [True], "the run must continue past a failing step"
        assert "[fail] explodes" in out.out.replace("\x1b[0;31m", "").replace(
            "\x1b[0m", ""
        )
        assert "tracker is missing" in out.err

    def test_the_step_exit_code_is_reported(self, capsys):
        repo_ops.run_steps([repo_ops.Step("x", run=lambda: 7)])
        assert "exit 7" in capsys.readouterr().out

    def test_subprocess_steps_still_work(self, monkeypatch):
        monkeypatch.setattr(
            repo_ops.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
        )
        assert repo_ops.run_steps([repo_ops.Step("echo", cmd=["echo", "hi"])]) == 0


class TestDispatch:
    def test_unknown_command_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            repo_ops.main(["nope"], deps=_deps(tmp_path))
