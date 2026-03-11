# GitHub Repository Metrics Dataset (5K+ Repos)

> 5,500 synthetic repositories with popularity, activity, maintenance, and community-health signals for software analytics.

**Kaggle dataset:** [lorenzoscaturchio/github-repo-metrics](https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics)  
**Companion notebook:** [Github Repo Metrics Explorer V2](https://www.kaggle.com/code/lorenzoscaturchio/github-repo-metrics-explorer-v2)  
**License:** GPL-3.0

## Overview

This dataset is designed for practical repository analytics: predicting stars, segmenting projects by health, and exploring how engineering hygiene relates to traction. It combines product-style signals, collaboration signals, and governance signals in one compact table.

All records are synthetic, but the schema is intentionally realistic enough for:
- tabular regression and classification
- software engineering analytics demos
- portfolio projects around open source quality and popularity
- downstream feature-importance or explainability examples

## Why This Dataset Is Useful

- A single tabular file with **5,500 repositories** and **29 features** makes it easy to benchmark tabular models quickly.
- Mixes **popularity**, **activity**, and **project health** signals in one place instead of forcing feature joins.
- Strong fit for software-engineering ML tasks, ranking experiments, and open-source health scoring.
- Synthetic generation keeps the structure realistic while avoiding maintenance overhead from live GitHub API collection.

## Quick Start

```python
import pandas as pd

df = pd.read_csv("github_repos.csv")
print(df.shape)
print(df[["language", "stars", "forks", "has_ci", "test_coverage"]].head())
```

Common starter tasks:

- Predict `stars` or bucketed popularity tiers.
- Score repository health from maintenance and community features.
- Compare language ecosystems by stars, CI usage, and contributor counts.
- Model dormant vs active projects using `last_commit_date`, issues, PRs, and releases.

## Quick Facts

| Property | Value |
|---|---|
| File | `github_repos.csv` |
| Rows | `5,500` |
| Columns | `29` |
| Coverage | `2014-01-01` to `2025-12-31` |
| Languages | `12` |
| Geography | `Global (synthetic)` |

## Best First Analyses

1. Predict `stars`, `forks`, or `watchers` from repository metadata.
2. Classify healthy vs at-risk repositories from governance and maintenance signals.
3. Compare languages by traction, release cadence, and contributor depth.
4. Measure how README size, CI, or test coverage correlate with community growth.
5. Build explainable tree models for software-product analytics.

## Column Guide

### Repository identity

- `repo_name`, `language`, `description`, `license`, `topics`
- `created_date`, `last_commit_date`, `default_branch`

### Popularity and activity

- `stars`, `forks`, `watchers`
- `open_issues`, `closed_issues`
- `open_pull_requests`, `merged_pull_requests`
- `contributors`, `commits`, `releases`

### Community health and maintenance

- `readme_length`, `has_ci`, `test_coverage`
- `has_code_of_conduct`, `has_contributing_guide`
- `has_wiki`, `has_pages`, `has_discussions`
- `is_archived`, `is_fork`, `size_kb`

## Linked Assets

- Dataset page: <https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics>
- Explore notebook: <https://www.kaggle.com/code/lorenzoscaturchio/github-repo-metrics-explorer-v2>

## Modeling Notes

- Popularity targets are intentionally long-tailed, so log transforms are often useful.
- Governance features such as `has_ci`, `has_contributing_guide`, and `has_code_of_conduct` are meant to support repository-health segmentation.
- `topics` and `description` can be used for lightweight NLP features if you want to combine structured and text signals.

## Provenance

- Generated from repository scripts in this project
- Built from public schema conventions and OSS platform patterns
- Intended for education, benchmarking, demos, and exploratory research

## Notes and Caveats

- This dataset is **synthetic**. It is designed to preserve realistic relationships between repo attributes, not to mirror a specific live GitHub snapshot.
- `topics` is useful for lightweight NLP or multi-label experiments, but it should be treated as simulated metadata rather than canonical GitHub taxonomy.
- `test_coverage` is intentionally sparse when CI is absent, which makes it useful for missingness-aware modeling.

## Changelog

- 2026-03-08: tightened the README, clarified the benchmark tasks, and added a quick-start workflow.

## Citation

Scaturchio, Lorenzo (2026). *GitHub Repository Metrics Dataset (5K+ Repos).* Kaggle Dataset. <https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics>
