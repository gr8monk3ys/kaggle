import importlib.util
import json
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "discussion_post.py"


def _load_discussion_post_module(monkeypatch):
    sync_api = types.ModuleType("playwright.sync_api")

    class DummyTimeout(Exception):
        pass

    class DummyPlaywrightContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    sync_api.sync_playwright = lambda: DummyPlaywrightContext()
    sync_api.TimeoutError = DummyTimeout

    playwright_mod = types.ModuleType("playwright")
    playwright_mod.sync_api = sync_api

    monkeypatch.setitem(sys.modules, "playwright", playwright_mod)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)

    module_name = f"discussion_post_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_require_kaggle_login_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("KAGGLE_EMAIL", raising=False)
    monkeypatch.delenv("KAGGLE_PASSWORD", raising=False)
    module = _load_discussion_post_module(monkeypatch)

    with pytest.raises(EnvironmentError, match="KAGGLE_EMAIL"):
        module.require_kaggle_login_env()


def test_main_exits_fast_when_login_env_missing(monkeypatch):
    monkeypatch.delenv("KAGGLE_EMAIL", raising=False)
    monkeypatch.delenv("KAGGLE_PASSWORD", raising=False)
    module = _load_discussion_post_module(monkeypatch)

    messages = []
    monkeypatch.setattr(module, "notify_safe", lambda message: messages.append(message))

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
    assert messages
    assert "KAGGLE_EMAIL" in messages[0]


def test_smoke_test_validates_postable_item_without_posting(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("KAGGLE_EMAIL", raising=False)
    monkeypatch.delenv("KAGGLE_PASSWORD", raising=False)
    module = _load_discussion_post_module(monkeypatch)

    drafts = tmp_path / "discussion-drafts.md"
    drafts.write_text(
        "\n".join(
            [
                "## Draft 1: Test Draft",
                "**Target forum:** General",
                "",
                "### Test Draft",
                "",
                "Body content here.",
            ]
        ),
        encoding="utf-8",
    )
    queue_path = tmp_path / "discussion_queue.json"
    queue_path.write_text(
        json.dumps(
            [
                {
                    "id": "draft_001",
                    "title": "Test Draft",
                    "forum_url": "https://www.kaggle.com/discussions/general",
                    "body_file": "discussion-drafts.md",
                    "body_section": "Draft 1",
                    "status": "scheduled",
                    "scheduled_after": "2099-01-01T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "QUEUE_PATH", queue_path)

    rc = module.smoke_test()

    assert rc == 0
    captured = capsys.readouterr()
    assert "Smoke candidate: Test Draft" in captured.out
    assert "Discussion smoke test passed" in captured.out


def test_main_smoke_test_queue_only_does_not_require_login_env(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("KAGGLE_EMAIL", raising=False)
    monkeypatch.delenv("KAGGLE_PASSWORD", raising=False)
    module = _load_discussion_post_module(monkeypatch)

    drafts = tmp_path / "discussion-drafts.md"
    drafts.write_text(
        "\n".join(
            [
                "## Draft 1: Test Draft",
                "**Target forum:** General",
                "",
                "### Test Draft",
                "",
                "Body content here.",
            ]
        ),
        encoding="utf-8",
    )
    queue_path = tmp_path / "discussion_queue.json"
    queue_path.write_text(
        json.dumps(
            [
                {
                    "id": "draft_001",
                    "title": "Test Draft",
                    "forum_url": "https://www.kaggle.com/discussions/general",
                    "body_file": "discussion-drafts.md",
                    "body_section": "Draft 1",
                    "status": "ready",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "QUEUE_PATH", queue_path)

    with pytest.raises(SystemExit) as exc:
        module.main(["--smoke-test"])

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Smoke candidate: Test Draft" in captured.out
    assert "Discussion smoke test passed" in captured.out


def test_smoke_test_succeeds_when_queue_has_no_postable_items(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("KAGGLE_EMAIL", raising=False)
    monkeypatch.delenv("KAGGLE_PASSWORD", raising=False)
    module = _load_discussion_post_module(monkeypatch)

    queue_path = tmp_path / "discussion_queue.json"
    queue_path.write_text(json.dumps([{"id": "done", "status": "posted"}]), encoding="utf-8")

    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "QUEUE_PATH", queue_path)

    rc = module.smoke_test()

    assert rc == 0
    captured = capsys.readouterr()
    assert "No postable discussion items found" in captured.out
