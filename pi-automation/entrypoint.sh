#!/bin/bash
set -euo pipefail

# Source .env if mounted
if [ -f /run/secrets/.env ]; then
    set -a; source /run/secrets/.env; set +a
fi

# Write kaggle.json from env vars
mkdir -p /root/.kaggle
cat > /root/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USERNAME:-}","key":"${KAGGLE_KEY:-}"}
EOF
chmod 600 /root/.kaggle/kaggle.json

echo "kaggle-autobot starting cron..."
cron -f
