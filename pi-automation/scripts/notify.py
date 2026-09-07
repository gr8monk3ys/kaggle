import os
import sys
import requests


def _redact(text: str, token: str) -> str:
    """Replace the bot token with '***' so it never reaches logs."""
    return text.replace(token, "***") if token else text


def send(message: str) -> None:
    """Send a Telegram message via Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    Raises EnvironmentError if token is missing.
    Prints to stderr on API failure (does not raise).
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN environment variable is not set")

    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"Telegram API request failed: {_redact(str(exc), token)}", file=sys.stderr)
        return

    if response.status_code != 200:
        print(
            f"Telegram API error {response.status_code}: {_redact(response.text, token)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send a Telegram notification")
    parser.add_argument("message", help="Message to send")
    args = parser.parse_args()
    send(args.message)
