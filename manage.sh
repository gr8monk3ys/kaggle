#!/usr/bin/env bash
set -euo pipefail

KAGGLE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_ROOT="${KAGGLE_DIR}"
export KAGGLE_DIR
export PYTHONPATH="${MODULE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="python3"
if [[ -x "${KAGGLE_DIR}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${KAGGLE_DIR}/.venv/bin/python3"
    # Fallback only: a kaggle binary already on the caller's PATH must win.
    export PATH="${PATH}:${KAGGLE_DIR}/.venv/bin"
fi

exec "${PYTHON_BIN}" -m kaggle_portfolio.cli "$@"
