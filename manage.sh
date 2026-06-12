#!/usr/bin/env bash
set -euo pipefail

KAGGLE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_ROOT="${KAGGLE_DIR}"
export KAGGLE_DIR
export PYTHONPATH="${MODULE_ROOT}:${PYTHONPATH:-}"

exec python3 -m kaggle_portfolio.cli "$@"
