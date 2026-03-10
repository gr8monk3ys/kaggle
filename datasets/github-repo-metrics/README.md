# GitHub Repository Metrics Dataset (5K+ Repos)

> 5,500 synthetic repositories with popularity, activity, and community-health signals for software analytics.

**Kaggle dataset:** [lorenzoscaturchio/github-repo-metrics](https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics)  
**Companion notebook:** [Github Repo Metrics Explorer V2](https://www.kaggle.com/code/lorenzoscaturchio/github-repo-metrics-explorer-v2)  
**License:** GPL-3.0

## Why this dataset is useful

This dataset is designed for practical software-analytics work: predicting stars or forks, segmenting repositories by health, and exploring how engineering signals relate to popularity. It combines traction metrics, maintenance signals, and community features in one table so you can move directly from EDA to modeling.

All records are synthetic but structurally realistic. The table covers repositories created from 2014 through 2025 and includes both product-style indicators like releases and README length and governance indicators like CI, contributing guides, code of conduct, wiki, and discussions.

## File Summary

- `github_repos.csv`
- Rows: `5,500`
- Columns: `29`
- File size: `0.99 MB`
- Coverage: `2014-01-01` to `2025-12-31`
- Geography: `Global (synthetic)`

## Column Groups

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

## Practical Use Cases

- Popularity regression for stars, forks, or watchers
- Repository-health classification and segmentation
- Language-mix analysis across software categories
- CI/CD adoption studies and OSS governance exploration
- Feature-importance demos for tabular models in software engineering contexts

## Linked Kaggle Assets

- Dataset page: <https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics>
- Explore notebook: <https://www.kaggle.com/code/lorenzoscaturchio/github-repo-metrics-explorer-v2>

## Provenance

- Synthetic data generated from repository scripts in this project
- Built from public schema conventions and OSS platform patterns
- Intended for education, benchmarking, demos, and exploratory research

## Citation

Scaturchio, Lorenzo (2026). *GitHub Repository Metrics Dataset (5K+ Repos).* Kaggle Dataset. <https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics>
