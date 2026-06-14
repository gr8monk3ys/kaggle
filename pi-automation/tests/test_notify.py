import pytest
import requests
from unittest.mock import patch


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


def test_send_handles_request_exception(monkeypatch, capsys):
    """Network/request failures should be logged without raising."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

    with patch("notify.requests.post", side_effect=requests.RequestException("timeout")):
        import importlib
        import notify
        importlib.reload(notify)
        notify.send("Transient failure")

    captured = capsys.readouterr()
    assert "request failed" in captured.err.lower()


def test_send_redacts_token_in_request_exception(monkeypatch, capsys):
    """A request failure must not leak the bot token into stderr."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token-123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

    exc = requests.RequestException(
        "HTTPSConnectionPool: url=https://api.telegram.org/botsecret-token-123/sendMessage failed"
    )
    with patch("notify.requests.post", side_effect=exc):
        import importlib
        import notify
        importlib.reload(notify)
        notify.send("leaky")

    captured = capsys.readouterr()
    assert "secret-token-123" not in captured.err
    assert "***" in captured.err
