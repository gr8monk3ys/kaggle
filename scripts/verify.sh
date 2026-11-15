#!/usr/bin/env bash
# Single local entrypoint for the gating CI job (.github/workflows/ci.yml -> quality).
# CI installs the dependencies and then runs exactly this script, so a green
# `make verify` (or `bash scripts/verify.sh`) locally means a green `quality` check.
#
# Deterministic by design: no Kaggle credentials, no network, no wall-clock SLAs.
# Content freshness (tracker staleness) and live Kaggle probes live in the
# scheduled Medal Ops Health monitor, not here.
#
# Needs: python (3.11 in CI) with pytest pytest-cov pytest-mock requests pyyaml
# numpy pandas scikit-learn "kaggle<2", plus ripgrep (rg).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python3}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN="python"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${WORK_DIR}/pycache}"

step() { printf '\n==> %s\n' "$*"; }

step "Security guardrails"
if git ls-files --error-unmatch kaggle.json >/dev/null 2>&1; then
  echo "::error file=kaggle.json::Remove committed Kaggle credentials."
  exit 1
fi
if rg -n --glob '*.py' --glob '*.sh' --glob '!tests/**' --glob '!.venv/**' '/[U]sers/' .; then
  echo "::error::Hardcoded /Users paths found in scripts."
  exit 1
fi
if rg -n --glob '*.py' --glob '*.sh' --glob '!tests/**' --glob '!.venv/**' 'trust_remote_code\s*=\s*True' .; then
  echo "::error::Avoid enabling trust_remote_code unless explicitly reviewed."
  exit 1
fi

step "Script syntax checks"
bash -n manage.sh
# shellcheck disable=SC2046
"${PYTHON_BIN}" -m py_compile $(git ls-files -- '*.py' | sort)

step "Build datasets"
bash manage.sh build-datasets

step "Repository preflight (no pytest; freshness SLAs belong to Medal Ops Health)"
bash manage.sh preflight --no-pytest

step "Unit tests with coverage"
"${PYTHON_BIN}" -m pytest -q --cov=. --cov-report=term-missing:skip-covered --cov-config=.coveragerc

step "Script smoke tests"
"${PYTHON_BIN}" datasets/ai-research-trends/create_dataset.py >"${WORK_DIR}/ai-research.log"
"${PYTHON_BIN}" datasets/programming-benchmarks/create_dataset.py >"${WORK_DIR}/programming-benchmarks.log"
"${PYTHON_BIN}" -m kaggle_portfolio.ops.medal_ops --output-root "${WORK_DIR}/medal_ops_ci" digest >"${WORK_DIR}/digest.log"
"${PYTHON_BIN}" -m kaggle_portfolio.notebooks.competition_entry --help >"${WORK_DIR}/comp-entry.log"

step "verify: all checks passed"
