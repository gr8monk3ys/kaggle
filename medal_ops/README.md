# Medal Ops

Progress records for the tracker in `docs/reports/grandmaster-tracker.md`.

```bash
./manage.sh doctor          # tracker parses? how stale? does Kaggle accept the key?
./manage.sh sync --dry-run  # preview live counts against the tracker
./manage.sh sync            # write them into the tracker and record a snapshot
./manage.sh digest          # one-message summary of change since the last snapshot
```

- `history/snapshot-*.json` — written by `sync`, committed, so change over time
  survives between machines.
- `reports/` — the latest doctor and sync reports. Local only (gitignored).

The weekly `medal-ops-health.yml` workflow runs `doctor` and a dry-run `sync`,
opening an issue when they fail and closing it on recovery. `telemetry.yml`
runs the real `sync` and proposes the result as a pull request.
