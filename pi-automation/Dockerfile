FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /scripts
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install chromium --with-deps

COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh 2>/dev/null || true

COPY crontab /etc/cron.d/kaggle-autobot
RUN chmod 0644 /etc/cron.d/kaggle-autobot && crontab /etc/cron.d/kaggle-autobot

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
