#!/bin/bash
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

doctor_status="✅ PASS"
quality_status="✅ PASS"

(cd "${REPO}" && python3 -m kaggle_portfolio.ops.medal_ops --output-root /tmp/health --today "${TODAY}" \
    doctor --strict) 2>&1 | tee /tmp/doctor.log || doctor_status="❌ FAIL"

(cd "${REPO}" && python3 -m kaggle_portfolio.quality.notebook_quality --output-root /tmp/health --today "${TODAY}" \
    --scope all --min-score 95 --fail-under-threshold) 2>&1 | tee /tmp/quality.log \
    || quality_status="❌ FAIL"

python3 /scripts/notify.py "📊 *Kaggle Health — ${TODAY}*
Doctor: ${doctor_status}
Quality: ${quality_status}"
