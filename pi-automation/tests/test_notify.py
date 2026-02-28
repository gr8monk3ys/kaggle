import os
import pytest
from unittest.mock import patch

# Add scripts to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def test_send_calls_telegram_api(monkeypatch):
    """Happy path: send() posts to Telegram API with correct payload."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

    with patch("notify.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        import notify
        notify.send("Hello from test")

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "test-token" in call_args[0][0]
    assert call_args[1]["json"]["text"] == "Hello from test"
    assert call_args[1]["json"]["chat_id"] == "123456"


def test_send_raises_on_missing_token(monkeypatch):
    """Missing TELEGRAM_BOT_TOKEN raises EnvironmentError."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

    import importlib
    import notify
    importlib.reload(notify)

    with pytest.raises(EnvironmentError, match="TELEGRAM_BOT_TOKEN"):
        notify.send("Should fail")


def test_send_prints_stderr_on_api_failure(monkeypatch, capsys):
    """Non-200 response from Telegram API prints error to stderr and does not raise."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

    with patch("notify.requests.post") as mock_post:
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        import importlib
        import notify
        importlib.reload(notify)
        notify.send("This will fail")

    captured = capsys.readouterr()
    assert "400" in captured.err or "Bad Request" in captured.err
