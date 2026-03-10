#!/bin/bash
set -euo pipefail
REPO="${REPO_PATH:-/repo}"
TODAY=$(date -u +%Y-%m-%d)

if (cd "${REPO}" && python3 -m kaggle_portfolio.ops.medal_ops --output-root /tmp/health --today "${TODAY}" \
    sync) 2>&1 | tee /tmp/sync.log; then
    python3 /scripts/notify.py "🔄 *Sync complete — ${TODAY}*"
else
    python3 /scripts/notify.py "❌ *Sync failed — ${TODAY}*"
fi
