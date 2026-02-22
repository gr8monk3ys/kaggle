# Deploying kaggle-autobot to Raspberry Pi

## Prerequisites

- Pi running Docker + Docker Compose v2
- Repo cloned: `git clone <repo-url> && cd kaggle`
- Kaggle credentials from https://www.kaggle.com/settings → API → Create New Token
- Telegram bot token from @BotFather on Telegram
- Your Telegram chat ID: start a chat with the bot, then visit
  `https://api.telegram.org/bot{TOKEN}/getUpdates` and find `"id"` in the result

## One-time setup

```bash
cd pi-automation
cp .env.example .env
nano .env          # fill in all values
```

## Build and start

```bash
docker compose -f pi-automation/docker-compose.yml up -d --build
```

First build takes 10-15 minutes on Pi 4 (Playwright Chromium install).

## Verify

```bash
# Check it started
docker logs kaggle-autobot

# Test Telegram
docker exec kaggle-autobot python3 /scripts/notify.py "hello from Pi"

# Test deadline check
docker exec kaggle-autobot python3 /scripts/deadline_alert.py

# Force a discussion post (ignore schedule)
docker exec -e FORCE_POST=1 kaggle-autobot python3 /scripts/discussion_post.py
```

## Updating

After `git pull`:
```bash
# Scripts changed → rebuild
docker compose -f pi-automation/docker-compose.yml up -d --build

# Only tracker/drafts changed → restart (repo is mounted live)
docker compose -f pi-automation/docker-compose.yml restart kaggle-autobot
```

## Logs

```bash
docker exec kaggle-autobot tail -f /var/log/cron.log
```
