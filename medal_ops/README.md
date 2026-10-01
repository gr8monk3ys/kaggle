# Medal Ops

Generated execution artifacts for Kaggle medal progress.

## Commands

```bash
./manage.sh sync
./manage.sh digest
./manage.sh doctor
./manage.sh quality
```

Equivalent direct usage:

```bash
python3 -m kaggle_portfolio.ops.medal_ops sync
python3 -m kaggle_portfolio.ops.medal_ops sync --dry-run
python3 -m kaggle_portfolio.ops.medal_ops digest
python3 -m kaggle_portfolio.ops.medal_ops doctor
python3 -m kaggle_portfolio.ops.medal_ops doctor --strict --require-kaggle
python3 -m kaggle_portfolio.quality.notebook_quality --min-score 70 --scope all
python3 -m kaggle_portfolio.quality.notebook_quality --min-score 70 --fix-target-score 85 --fix-top-actions 4 --scope all
```

`sync` writes the tracker and a history snapshot; `--dry-run` writes neither.
`digest` compares the latest two snapshots.
Use `doctor` before sync to validate tracker health and environment readiness.

## Output

- `medal_ops/history/snapshot-*.json`: point-in-time metrics snapshots, written by `sync`.
- `medal_ops/reports/latest-sync.md`: most recent live sync report.
- `medal_ops/reports/latest-doctor.md`: most recent preflight report.
- `medal_ops/reports/latest-notebook-quality.md`: most recent notebook quality scorecard.
- `medal_ops/reports/latest-notebook-quality-fixes.md`: prioritized per-notebook fix checklist.

## Scheduled Health Checks

- `.github/workflows/medal-ops-health.yml` runs daily and on manual dispatch.
- On failure, it opens/updates a tracking issue automatically.
- It currently uses `doctor --strict --max-stale-days 30`.
- It also runs `python -m kaggle_portfolio.quality.notebook_quality --fail-under-threshold` with default `--min-score 95`.
- Manual dispatch supports `mode`, `max_stale_days`, and `min_quality_score` inputs.
- Set repository secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` for live-mode checks.

## Inputs

- `docs/reports/grandmaster-tracker.md` is the primary source of truth.
- Keep that tracker updated for accurate reports.
