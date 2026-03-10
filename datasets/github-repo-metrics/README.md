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

## Citation

Scaturchio, Lorenzo (2026). *GitHub Repository Metrics Dataset (5K+ Repos).* Kaggle Dataset. <https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics>
