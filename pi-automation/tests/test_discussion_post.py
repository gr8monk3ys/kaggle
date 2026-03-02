import importlib.util
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
