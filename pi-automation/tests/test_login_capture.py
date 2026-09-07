"""Tests for login_capture.py session-capture entry point."""

import sys
from contextlib import contextmanager
from pathlib import Path

import login_capture


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["login_capture.py", *argv])
    return login_capture.main()


def test_dry_run_skips_browser_and_succeeds(monkeypatch, capsys, tmp_path):
    state = tmp_path / "state.json"

    def _boom(args):
        raise AssertionError("browser must not open in dry-run")

    monkeypatch.setattr(login_capture.kb, "open_kaggle_browser", _boom)
    rc = _run(monkeypatch, ["--dry-run", "--storage-state", str(state)])

    assert rc == 0
    assert str(state) in capsys.readouterr().out


def test_successful_capture_reports_storage_path(monkeypatch, capsys, tmp_path):
    state = tmp_path / "state.json"

    @contextmanager
    def _fake_browser(args):
        assert Path(args.storage_state) == state
        yield object()

    monkeypatch.setattr(login_capture.kb, "open_kaggle_browser", _fake_browser)
    rc = _run(monkeypatch, ["--storage-state", str(state)])

    assert rc == 0
    assert "[done]" in capsys.readouterr().out


def test_failed_login_returns_nonzero(monkeypatch, capsys):
    @contextmanager
    def _fake_browser(args):
        raise RuntimeError("Kaggle login still appears unauthenticated after manual login.")
        yield  # pragma: no cover

    monkeypatch.setattr(login_capture.kb, "open_kaggle_browser", _fake_browser)
    rc = _run(monkeypatch, [])

    assert rc == 1
    assert "[failed]" in capsys.readouterr().out
