# Medal Ops

Generated execution artifacts for Kaggle medal progress.

## Commands

```bash
./manage.sh scorecard
./manage.sh weekly-plan
./manage.sh pace
./manage.sh sync
./manage.sh sync-template
./manage.sh doctor
./manage.sh quality
```

Equivalent direct usage:

```bash
python3 medal_ops.py scorecard
python3 medal_ops.py weekly-plan
python3 medal_ops.py pace
python3 medal_ops.py sync
python3 medal_ops.py sync --dry-run
python3 medal_ops.py sync-template
python3 medal_ops.py doctor
python3 medal_ops.py doctor --strict --kernels-csv kernels.csv --datasets-csv datasets.csv --competitions-csv competitions.csv
python3 medal_ops.py sync --kernels-csv kernels.csv --datasets-csv datasets.csv --competitions-csv competitions.csv --dry-run
python3 notebook_quality.py --min-score 70 --scope all
python3 notebook_quality.py --min-score 70 --fix-target-score 85 --fix-top-actions 4 --scope all
```

CSV sync is useful when Kaggle CLI/network access is unavailable.
Use `sync-template` to scaffold CSV inputs and an export helper script.
Use `doctor` before sync to validate tracker health, environment readiness, and CSV inputs.

## Output

- `medal_ops/history/snapshot-*.json`: point-in-time metrics snapshots.
- `medal_ops/reports/latest-scorecard.md`: most recent scorecard.
- `medal_ops/reports/latest-weekly-plan.md`: most recent weekly plan.
- `medal_ops/reports/latest-pace.md`: most recent velocity/ETA analysis.
- `medal_ops/reports/latest-sync.md`: most recent live sync report.
- `medal_ops/reports/latest-doctor.md`: most recent preflight report.
- `medal_ops/reports/latest-notebook-quality.md`: most recent notebook quality scorecard.
- `medal_ops/reports/latest-notebook-quality-fixes.md`: prioritized per-notebook fix checklist.

## Scheduled Health Checks

- `.github/workflows/medal-ops-health.yml` runs daily and on manual dispatch.
- On failure, it opens/updates a tracking issue automatically.
- It currently uses `doctor --strict --max-stale-days 30`.
- It also runs `notebook_quality.py --fail-under-threshold` with default `--min-score 95`.
- Manual dispatch supports `mode`, `max_stale_days`, and `min_quality_score` inputs.
- Set repository secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` for live-mode checks.

## Inputs

- `grandmaster-tracker.md` is the primary source of truth.
- Keep that tracker updated for accurate reports.
