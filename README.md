# Kaggle Portfolio

[![CI](https://github.com/gr8monk3ys/kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/gr8monk3ys/kaggle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-lorenzoscaturchio-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/lorenzoscaturchio)

The notebooks, datasets and competition entries behind
[kaggle.com/lorenzoscaturchio](https://www.kaggle.com/lorenzoscaturchio), plus
`kaggle_portfolio/`, a small tested package that validates and publishes them,
scouts competitions, and keeps an honest record of progress.

Where things stand, and the gap to each next tier:
[docs/reports/grandmaster-tracker.md](./docs/reports/grandmaster-tracker.md).

## Layout

| Path | What lives there |
|------|------------------|
| `projects/competitions/` | One folder per competition entry: the `.ipynb` and its `kernel-metadata.json` |
| `projects/educational/` | Teaching notebooks, one per folder |
| `datasets/` | One folder per published dataset: metadata, README, `explore.ipynb`, and the script that produces the data |
| `kaggle_portfolio/` | The package behind `./manage.sh` |
| `docs/reports/` | The tracker and the competition scout report |
| `docs/adr/` | Architecture decisions, including [why nothing here votes, follows or posts automatically](./docs/adr/0007-no-automated-engagement.md) |
| `medal_ops/history/` | Snapshots `sync` records, so `digest` can show change over time |

The `.ipynb` files are the source for every notebook. Edit them directly and run
them on Kaggle before publishing a claim about results.

## Setup

Linux, macOS or WSL with Python 3.11:

```bash
git clone https://github.com/gr8monk3ys/kaggle.git && cd kaggle
python3 -m venv .venv && . .venv/bin/activate
pip install pytest pytest-cov pytest-mock requests pyyaml numpy pandas scikit-learn "kaggle<2"
./manage.sh build-datasets      # dataset CSVs are generated, not committed
python -m pytest -q
```

For anything that talks to Kaggle, create a token at
[kaggle.com/settings](https://www.kaggle.com/settings) → API → *Create New
Token*, save it as `~/.kaggle/kaggle.json`, and `chmod 600` it. Never commit it.

## Commands

`./manage.sh help` lists everything. The ones that matter:

**Publish**

| Command | Does |
|---------|------|
| `validate [dir]` | Checks metadata (required fields, title and keyword limits, declared files present, no credentials) |
| `push <dir>` | Validates, then publishes one notebook or dataset |
| `push-nb` / `push-ds` / `push-all` | The same for every notebook, every dataset, or both |
| `build-datasets [name ...]` | Regenerates dataset CSVs from each seeded `create_dataset.py` |
| `publish-datasets [--apply]` | Publishes datasets that pass the usability gate (dry run without `--apply`) |
| `dataset-usability` | Scores each dataset against Kaggle's usability checklist |

A dataset is never published without the files its metadata declares; the Kaggle
client refuses an incomplete folder whichever command asks.

**Compete**

| Command | Does |
|---------|------|
| `scout [--update]` | Ranks live boards twice: for competition medals (Featured and Research only) and for notebook votes (any active board) |
| `create-competition-entry <slug>` | Scaffolds a starter notebook and metadata for a competition |
| `competition-lab <slug>` | Runs a local cross-validated benchmark and can write or submit a submission |
| `leaderboard record\|report` | Records your rank on entered competitions over time |
| `link-competition <dir> <slug>` | Attaches a notebook to a competition and republishes it |

**Progress**

| Command | Does |
|---------|------|
| `sync [--dry-run]` | Pulls live notebook, dataset and competition counts into the tracker and records a snapshot |
| `digest` | One-message summary of change since the last snapshot |
| `doctor` | Checks the tracker parses, how stale it is, and whether Kaggle accepts the credentials |
| `auth-doctor` | Checks the key, dataset ownership and upload authorisation (exit 3 means Kaggle rejected the key) |
| `status` / `votes` | Live notebook and dataset listing, with distance to each medal |
| `preflight` | The CI gate: validate, doctor, dataset usability, tests |

## Datasets

Most datasets here are synthetic and say so in their README; their CSVs are
rebuilt byte for byte by `./manage.sh build-datasets`, so they are not
committed. `datasets/github-ml-repos/` is the exception: a real snapshot from
GitHub's API (`fetch_dataset.py`, needs `gh auth login`), which cannot be
regenerated later, so its CSV is committed as source.

## Automation

| Workflow | When | What |
|----------|------|------|
| [`ci.yml`](./.github/workflows/ci.yml) | Every push and PR | Builds the datasets, runs `preflight --no-pytest`, the test suite with coverage, and smoke runs |
| [`medal-ops-health.yml`](./.github/workflows/medal-ops-health.yml) | Weekly and on push to `main` | Doctor, dataset usability and a dry-run sync; opens an issue on failure and closes it on recovery |
| [`telemetry.yml`](./.github/workflows/telemetry.yml) | Weekly | Live sync and scout refresh, proposed as a PR |
| [`live-smoke.yml`](./.github/workflows/live-smoke.yml) | Manual | Authenticated, non-mutating publish checks |
| `security-baseline.yml`, `semgrep.yml` | Scheduled | Security scans |

The scheduled jobs need a `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` +
`KAGGLE_KEY`) repository secret. If Kaggle rejects the key, they warn and run
offline instead of failing; rotate the secret to restore live data.

## Tests

```bash
./manage.sh build-datasets
./manage.sh preflight --no-pytest
python -m pytest -q
```

The suite runs offline: Kaggle is replaced by an in-memory fake client, so no
test needs credentials or a network.

## License

[MIT](./LICENSE). Copyright (c) 2026 gr8monk3ys.
