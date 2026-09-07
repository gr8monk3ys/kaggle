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

# cron gives jobs only SHELL, PATH, HOME and LOGNAME — never the container's
# environment. Without this, REPO_PATH, QUEUE_PATH, KAGGLE_EMAIL, KAGGLE_PASSWORD
# and TELEGRAM_* are all unset for every scheduled job, while `docker exec` tests
# pass because exec DOES inherit the container env. That is why deadline_alert.py
# has been dying at import on its 6-hourly run: with REPO_PATH unset it inserted
# "/" on sys.path and could not import kaggle_portfolio.
#
# Render the environment where cron can read it, before cron starts.
printenv | grep -E '^(REPO_PATH|QUEUE_PATH|KAGGLE_|TELEGRAM_|DISCUSSION_|FLYWHEEL_)' \
    | sed 's/^/export /' > /etc/cron-env.sh
chmod 0644 /etc/cron-env.sh

echo "kaggle-autobot starting cron..."
cron -f
