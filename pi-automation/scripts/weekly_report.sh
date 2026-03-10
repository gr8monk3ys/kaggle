#!/bin/bash
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

(cd "${REPO}" && python3 -m kaggle_portfolio.ops.medal_ops --output-root /tmp/health --today "${TODAY}" \
    weekly-plan) 2>&1 | tee /tmp/weekly.log

REPORT=$(head -60 /tmp/health/reports/latest-weekly-plan.md 2>/dev/null || echo "Report not generated.")
python3 /scripts/notify.py "📅 *Weekly Plan — ${TODAY}*

${REPORT}"
