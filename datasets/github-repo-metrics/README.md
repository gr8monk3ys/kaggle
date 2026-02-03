---
title: "GitHub Repository Metrics Dataset (5K+ Repos)"
description: "Synthetic but realistic dataset of 5,500 GitHub repositories with stars, forks, issues, contributors, CI/CD status, test coverage, and more. Features realistic correlations between metrics."
license: CC0-1.0
tags:
  - github
  - open source
  - software engineering
  - repository analytics
  - popularity prediction
---

# GitHub Repository Metrics Dataset

![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)
![Repos: 5,500](https://img.shields.io/badge/Repos-5%2C500-blue.svg)
![Features: 29](https://img.shields.io/badge/Features-29-green.svg)
![Languages: 19](https://img.shields.io/badge/Languages-19-orange.svg)

## Overview

A synthetic dataset of **5,500 GitHub repositories** with realistic metrics, correlations, and distributions. Designed for practicing **popularity prediction**, **language trend analysis**, **open source health assessment**, and **software engineering analytics**.

The data mirrors real GitHub distributions including power-law star counts, language popularity trends, and realistic correlations between community and code health metrics.

---

## Quick Start

```python
import pandas as pd
import numpy as np

# On Kaggle
df = pd.read_csv('/kaggle/input/github-repo-metrics/github_repos.csv')

# Quick exploration
print(f"Total repos: {len(df):,}")
print(f"Languages: {df['language'].nunique()}")
print(f"\nStar distribution:")
for p in [50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {df['stars'].quantile(p/100):,.0f}")

# Top languages by median stars
print("\nMedian stars by language (top 10):")
print(df.groupby('language')['stars'].median().sort_values(ascending=False).head(10))

# Star prediction features
df['log_stars'] = np.log1p(df['stars'])
print(f"\nCorrelation with log(stars):")
numeric = df.select_dtypes(include=[np.number])
print(numeric.corrwith(df['log_stars']).sort_values(ascending=False).head(10))
```

---

## Dataset Description

### Content

The dataset contains 29 features per repository, capturing code, community, and health metrics.

| Column | Type | Description |
|--------|------|-------------|
| `repo_name` | string | Repository name |
| `language` | string | Primary programming language (19 languages) |
| `description` | string | Repository description |
| `stars` | int | GitHub stars (0-200K, power-law distributed) |
| `forks` | int | Number of forks (~10-30% of stars) |
| `watchers` | int | Number of watchers |
| `open_issues` | int | Currently open issues |
| `closed_issues` | int | Total closed issues |
| `open_pull_requests` | int | Open PRs |
| `merged_pull_requests` | int | Merged PRs |
| `contributors` | int | Number of contributors |
| `commits` | int | Total commits |
| `releases` | int | Number of releases |
| `license` | string | License type (MIT, Apache-2.0, GPL-3.0, etc.) |
| `topics` | string | Pipe-separated topic tags |
| `created_date` | date | Repository creation date |
| `last_commit_date` | date | Date of most recent commit |
| `readme_length` | int | README character count |
| `has_ci` | int | Has CI/CD configured |
| `test_coverage` | float | Test coverage percentage (NaN if no CI) |
| `has_code_of_conduct` | int | Has code of conduct |
| `has_contributing_guide` | int | Has contributing guidelines |
| `has_wiki` | int | Has wiki enabled |
| `has_pages` | int | Has GitHub Pages |
| `has_discussions` | int | Has discussions enabled |
| `default_branch` | string | Default branch name |
| `is_archived` | int | Repository is archived |
| `is_fork` | int | Repository is a fork |
| `size_kb` | int | Repository size in KB |

### Built-in Correlations

The data includes realistic statistical relationships:

- **Stars and forks**: Forks are ~10-30% of stars
- **Stars and language**: Rust and Python repos tend to get more stars
- **Age and stars**: Older repos accumulate more stars
- **Stars and CI**: Popular repos are more likely to have CI/CD
- **README length and popularity**: Popular repos have longer READMEs
- **Activity and archival**: Inactive repos are more likely to be archived
- **Contributors and commits**: More contributors means more commits

### Language Distribution

Reflects real GitHub trends (2024): Python (18%), JavaScript (16%), TypeScript (10%), Java (9%), Go (7%), Rust (6%), and 13 other languages.

---

## Use Cases

| # | Use Case | Type | Key Features |
|---|----------|------|-------------|
| 1 | **Star Prediction** | Regression | forks, watchers, contributors, README length |
| 2 | **Language Trend Analysis** | Visualization / Stats | language, stars, created_date |
| 3 | **Open Source Health Scoring** | Feature Engineering | has_ci, test_coverage, has_code_of_conduct |
| 4 | **Repository Classification** | Multi-class Classification | topics, description, language |
| 5 | **Contributor Analysis** | Correlation / Regression | contributors, commits, merged_pull_requests |
| 6 | **License Analysis** | Exploratory Analysis | license, stars, language |
| 7 | **Activity Prediction** | Binary Classification | last_commit_date, is_archived, age |
| 8 | **Feature Engineering Practice** | Tabular ML | All columns -- derive meaningful features |
| 9 | **Anomaly Detection** | Unsupervised Learning | outlier stars, unusual metrics combinations |
| 10 | **Topic Network Analysis** | Graph Analysis | topics co-occurrence patterns |

### Related Kaggle Competitions

This dataset lets you practice techniques from:
- [Predict Student Performance](https://www.kaggle.com/competitions/predict-student-performance-from-game-play) -- tabular prediction
- [Tabular Playground Series](https://www.kaggle.com/competitions?search=tabular+playground) -- general tabular modeling
- [Google Research - Identify Contrails](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming) -- working with metadata features

---

## File Structure

```
github-repo-metrics/
  github_repos.csv         # 5,500 repositories with 29 features
  create_dataset.py        # Generation script
  explore.ipynb            # Exploration notebook with EDA & star prediction model
  dataset-metadata.json    # Kaggle dataset metadata
  kernel-metadata.json     # Kaggle notebook metadata
```

---

## Citation

```
@dataset{github_repo_metrics_2025,
  title={GitHub Repository Metrics Dataset},
  author={Lorenzo Scaturchio},
  year={2025},
  url={https://www.kaggle.com/datasets/lorenzoscaturchio/github-repo-metrics}
}
```

## License

CC0 1.0 Universal -- Public Domain.

---

**If you found this dataset useful, please upvote! It helps others in the community discover it.**
