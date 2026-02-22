#!/bin/bash
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

doctor_status="✅ PASS"
quality_status="✅ PASS"

python3 "${REPO}/medal_ops.py" --output-root /tmp/health --today "${TODAY}" \
    doctor --strict 2>&1 | tee /tmp/doctor.log || doctor_status="❌ FAIL"

python3 "${REPO}/notebook_quality.py" --output-root /tmp/health --today "${TODAY}" \
    --scope all --min-score 95 --fail-under-threshold 2>&1 | tee /tmp/quality.log \
    || quality_status="❌ FAIL"

python3 /scripts/notify.py "📊 *Kaggle Health — ${TODAY}*
Doctor: ${doctor_status}
Quality: ${quality_status}"
