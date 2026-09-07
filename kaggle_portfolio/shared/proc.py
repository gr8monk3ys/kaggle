#!/usr/bin/env python3
"""General process and logging helpers, independent of Kaggle.

These are the two things left over from ``kaggle_utils`` once the Kaggle CLI
moved behind :mod:`kaggle_portfolio.shared.kaggle_client` and the date helpers
moved to :mod:`kaggle_portfolio.shared.clock`.
"""

from __future__ import annotations

import logging
import re

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_ERROR_HINT = re.compile(
    r"(error|unauthorized|forbidden|denied|failed|exception|traceback)", re.IGNORECASE
)


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


def summarize_output(*chunks: str) -> str:
    """Return a compact, human-meaningful summary of command output.

    Prefers a line naming an error; falls back to the last non-blank line.
    """
    lines = [
        line.strip() for chunk in chunks for line in chunk.splitlines() if line.strip()
    ]
    if not lines:
        return "unknown error"
    preferred = [line for line in lines if _ERROR_HINT.search(line)]
    return (preferred[-1] if preferred else lines[-1])[:220]
