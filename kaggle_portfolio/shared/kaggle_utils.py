#!/usr/bin/env python3
"""Shared utilities for Kaggle CLI operations, date parsing, and subprocess handling.

Consolidates duplicated helpers that were previously copy-pasted across
medal_ops, dataset_usability, notebook_pipeline, competition_scout,
kaggle_auth_doctor, dataset_publish_pipeline, dataset_optimizer,
dataset_usability_benchmark, campaign_pack, discussion_scheduler, and
notebook_quality.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    name: str = "kaggle",
    *,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Return a configured logger with console (and optional file) output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(console)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Kaggle CLI helpers
# ---------------------------------------------------------------------------


def kaggle_cli_path() -> str | None:
    """Return the best available Kaggle CLI executable path, if any.

    Resolution order:
    1. ``KAGGLE_CLI_BIN`` environment variable
    2. ``kaggle`` on ``PATH``
    3. a ``kaggle`` sibling next to the active Python executable
    """
    candidates: list[Path] = []

    env_override = os.environ.get("KAGGLE_CLI_BIN", "").strip()
    if env_override:
        candidates.append(Path(env_override).expanduser())

    which_path = shutil.which("kaggle")
    if which_path:
        candidates.append(Path(which_path))

    exe_path = Path(sys.executable)
    candidates.extend(
        [
            exe_path.parent / "kaggle",
            exe_path.parent / "kaggle.exe",
            exe_path.resolve().parent / "kaggle",
            exe_path.resolve().parent / "kaggle.exe",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _kaggle_cli_module_available() -> bool:
    """Return whether the ``kaggle.cli`` module can be imported.

    ``find_spec`` imports the parent ``kaggle`` package, which may fail in
    several ways: ``ModuleNotFoundError`` when it is not installed, other
    ``ImportError`` subtypes when a transitive dependency is missing, and
    ``OSError`` in versions that touch credentials at import time. For a
    boolean availability probe, any such failure means "not usable".
    """
    try:
        return importlib.util.find_spec("kaggle.cli") is not None
    except (ImportError, OSError):
        return False


def has_kaggle_cli() -> bool:
    """Return whether a usable Kaggle CLI is available."""
    return kaggle_cli_path() is not None or _kaggle_cli_module_available()


def kaggle_command() -> list[str]:
    """Return a runnable Kaggle CLI command prefix."""
    binary = kaggle_cli_path()
    if binary:
        return [binary]
    if _kaggle_cli_module_available():
        return [sys.executable, "-m", "kaggle.cli"]
    return ["kaggle"]


def run_kaggle(args: list[str], **subprocess_kwargs: Any) -> str:
    """Run a Kaggle CLI command and return stdout.

    Raises ``RuntimeError`` with a human-readable summary on failure.
    """
    result = subprocess.run(
        [*kaggle_command(), *args],
        capture_output=True,
        text=True,
        **subprocess_kwargs,
    )
    if result.returncode != 0:
        raise RuntimeError(summarize_subprocess_error(result.stdout, result.stderr))
    return result.stdout


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_iso_date(text: str | None) -> date | None:
    """Parse a ``YYYY-MM-DD`` string into a :class:`date`, or ``None``."""
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def resolve_today(today_override: str | None) -> date:
    """Return *today_override* as a date, or today's date when ``None``."""
    if not today_override:
        return date.today()
    parsed = parse_iso_date(today_override)
    if not parsed:
        raise SystemExit(f"Invalid --today value: {today_override}")
    return parsed


# ---------------------------------------------------------------------------
# Subprocess error summarisation
# ---------------------------------------------------------------------------


def summarize_subprocess_error(*chunks: str) -> str:
    """Return a compact, human-meaningful error summary from subprocess output.

    Prefers lines that contain common error keywords.  Falls back to the
    last non-blank line.
    """
    lines: list[str] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        return "unknown error"
    preferred = [
        line
        for line in lines
        if re.search(
            r"(error|unauthorized|forbidden|denied|failed|exception|traceback)",
            line,
            re.IGNORECASE,
        )
    ]
    return (preferred[-1] if preferred else lines[-1])[:220]


# ---------------------------------------------------------------------------
# Retry with exponential back-off
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable: tuple[type[BaseException], ...] = (RuntimeError, subprocess.SubprocessError),
) -> Any:
    """Decorator: retry a function with exponential back-off.

    >>> @retry(max_attempts=3)
    ... def flaky():
    ...     run_kaggle(["datasets", "list"])
    """

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger = logging.getLogger("kaggle.retry")
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s), retrying in %.1fs",
                        attempt,
                        max_attempts,
                        fn.__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
