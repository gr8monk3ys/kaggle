"""The competition-lab harness: fetching data, recording a run, submitting.

The per-competition benchmarks are tested in ``test_competition_lab.py``; this
covers what wraps them — ``runner`` and ``competition_lab.main`` — against the
fake Kaggle, so nothing here downloads or submits for real.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from kaggle_portfolio.notebooks import competition_lab as lab
from kaggle_portfolio.notebooks.competition_lab import runner
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.errors import CommandError
from kaggle_portfolio.shared.kaggle_client import (
    FakeKaggleClient,
    KaggleError,
    PushOutcome,
)


@pytest.fixture(autouse=True)
def _reset_layout():
    """The lab root is a module-level value; never let one test's root leak."""
    yield
    runner.set_layout(None)


class _ZipDownloadClient(FakeKaggleClient):
    """A fake whose download drops a zip, as the real CLI does."""

    def download_competition(self, slug: str, dest: Path) -> Path:
        super().download_competition(slug, dest)
        with zipfile.ZipFile(dest / f"{slug}.zip", "w") as zf:
            zf.writestr("train.csv", "id,y\n1,0\n")
            zf.writestr("test.csv", "id\n2\n")
        return dest


class _FailingDownloadClient(FakeKaggleClient):
    def download_competition(self, slug: str, dest: Path) -> Path:
        raise KaggleError("403 - accept the competition rules first")


class _RejectingSubmitClient(FakeKaggleClient):
    def submit(self, slug: str, file: Path, message: str) -> PushOutcome:
        self.calls.append(("submit", (slug, file, message)))
        return PushOutcome(False, "Submission limit reached")


def _result(submission_path: Path | None = None) -> runner.LabResult:
    return runner.LabResult(
        competition="titanic",
        metric_name="accuracy",
        best_model="logreg",
        best_score=0.8123456789,
        benchmark_rows=[
            {"model": "logreg", "score": 0.81235},
            {"model": "rf", "score": 0.80001},
        ],
        submission_path=submission_path,
    )


# -- _ensure_data -------------------------------------------------------------


def test_ensure_data_downloads_into_the_deps_lab_root_and_extracts_zips(tmp_path):
    client = _ZipDownloadClient()
    deps = Deps.for_test(tmp_path, client=client)

    data_dir = runner._ensure_data(deps, "titanic")

    assert data_dir == tmp_path / ".competition_lab" / "titanic" / "data"
    assert sorted(p.name for p in data_dir.glob("*.csv")) == ["test.csv", "train.csv"]
    assert client.calls == [("download_competition", ("titanic", data_dir))]


def test_ensure_data_uses_the_cache_and_skips_the_download(tmp_path):
    client = FakeKaggleClient()
    deps = Deps.for_test(tmp_path, client=client)
    cached = tmp_path / ".competition_lab" / "titanic" / "data"
    cached.mkdir(parents=True)
    (cached / "train.csv").write_text("id\n1\n", encoding="utf-8")

    assert runner._ensure_data(deps, "titanic") == cached
    assert client.calls == []


def test_ensure_data_force_download_refetches_a_warm_cache(tmp_path):
    client = FakeKaggleClient()
    deps = Deps.for_test(tmp_path, client=client)
    cached = tmp_path / ".competition_lab" / "titanic" / "data"
    cached.mkdir(parents=True)
    (cached / "train.csv").write_text("id\n1\n", encoding="utf-8")

    runner._ensure_data(deps, "titanic", force_download=True)

    assert [name for name, _ in client.calls] == ["download_competition"]


def test_ensure_data_turns_a_kaggle_failure_into_a_command_error(tmp_path):
    deps = Deps.for_test(tmp_path, client=_FailingDownloadClient())

    with pytest.raises(CommandError, match="Failed to download titanic: 403"):
        runner._ensure_data(deps, "titanic")


# -- recording a run ------------------------------------------------------------


def test_save_summary_writes_latest_benchmark_json_under_the_lab_root(tmp_path):
    runner.set_layout(Deps.for_test(tmp_path).layout)
    submission = tmp_path / "sub.csv"

    runner._save_summary(_result(submission))

    written = tmp_path / ".competition_lab" / "titanic" / "latest_benchmark.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload == {
        "competition": "titanic",
        "metric_name": "accuracy",
        "best_model": "logreg",
        "best_score": 0.812346,
        "benchmarks": [
            {"model": "logreg", "score": 0.81235},
            {"model": "rf", "score": 0.80001},
        ],
        "submission_path": str(submission),
    }


def test_save_summary_records_no_submission_as_null(tmp_path):
    runner.set_layout(Deps.for_test(tmp_path).layout)

    runner._save_summary(_result())

    written = tmp_path / ".competition_lab" / "titanic" / "latest_benchmark.json"
    assert json.loads(written.read_text(encoding="utf-8"))["submission_path"] is None


def test_safe_slug_is_filename_safe():
    assert runner._safe_slug("LightGBM + XGB (blend)") == "lightgbm---xgb--blend"


# -- submitting -------------------------------------------------------------------


def test_submit_raises_when_kaggle_rejects_the_file(tmp_path):
    deps = Deps.for_test(tmp_path, client=_RejectingSubmitClient())

    with pytest.raises(CommandError, match="Submission failed: Submission limit"):
        runner._submit(deps, "titanic", tmp_path / "sub.csv", "msg")


# -- main: the whole CLI path -----------------------------------------------------


def _fake_benchmark(calls: list, submission: Path | None):
    def bench(data_dir, folds, write_submission=False):
        calls.append((data_dir, folds, write_submission))
        return _result(submission if write_submission else None)

    return bench


def test_main_runs_the_benchmark_records_it_and_does_not_submit(
    tmp_path, monkeypatch, capsys
):
    client = FakeKaggleClient()
    deps = Deps.for_test(tmp_path, client=client)
    calls: list = []
    monkeypatch.setattr(
        lab, "load_benchmark", lambda slug: _fake_benchmark(calls, None)
    )

    assert lab.main(["titanic", "--cv-folds", "3"], deps=deps) == 0

    data_dir = tmp_path / ".competition_lab" / "titanic" / "data"
    assert calls == [(data_dir, 3, False)]
    assert (
        tmp_path / ".competition_lab" / "titanic" / "latest_benchmark.json"
    ).exists()
    assert [name for name, _ in client.calls] == ["download_competition"]
    assert "Best: " in capsys.readouterr().out


def test_main_submit_writes_and_submits_the_best_file(tmp_path, monkeypatch):
    client = FakeKaggleClient()
    deps = Deps.for_test(tmp_path, client=client)
    submission = tmp_path / "submission_logreg.csv"
    calls: list = []
    monkeypatch.setattr(
        lab, "load_benchmark", lambda slug: _fake_benchmark(calls, submission)
    )

    assert lab.main(["titanic", "--submit"], deps=deps) == 0

    # --submit implies writing the submission even without --write-submission.
    assert calls[0][2] is True
    submits = [args for name, args in client.calls if name == "submit"]
    assert submits == [
        (
            "titanic",
            submission,
            "Local logreg baseline via competition-lab (accuracy=0.81235)",
        )
    ]


def test_main_submit_without_a_submission_file_is_an_error(tmp_path, monkeypatch):
    client = FakeKaggleClient()
    deps = Deps.for_test(tmp_path, client=client)
    monkeypatch.setattr(lab, "load_benchmark", lambda slug: _fake_benchmark([], None))

    with pytest.raises(CommandError, match="No submission file was generated"):
        lab.main(["titanic", "--submit"], deps=deps)
    assert "submit" not in [name for name, _ in client.calls]
