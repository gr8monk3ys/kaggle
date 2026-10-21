# GitHub ML & AI Repositories (2026 Snapshot)

**Kaggle:** [lorenzoscaturchio/github-ml-ai-repositories-2026](https://www.kaggle.com/datasets/lorenzoscaturchio/github-ml-ai-repositories-2026) (not yet published)

## Description

Every machine learning and AI repository on GitHub with at least 500 stars
across 20 topics, pulled from GitHub's public search API on 2026-10-01. This is
real data, not generated: 7,550 repositories, one row each.

## Files

| File | Rows | Contents |
|------|------|----------|
| `github_ml_repos.csv` | 7,550 | Stars, forks, issues, size, language, license, topics, dates and derived age/staleness |

## Columns

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `repo` | string | 0.0% | 7,550 | affaan-m/ECC, NousResearch/hermes-agent, deepseek-ai/deepseek-harness |
| `owner` | string | 0.0% | 5,568 | affaan-m, NousResearch, deepseek-ai |
| `owner_type` | string | 0.0% | 2 | User, Organization |
| `description` | string | 0.3% | 7,528 | The agent harness performanc, The agent that grows with yo, DeepSeek Harness: Everything |
| `language` | string | 8.9% | 65 | JavaScript, Python, TypeScript |
| `stars` | integer | 0.0% | 3,955 | 270541, 250546, 241623 |
| `forks` | integer | 0.0% | 1,860 | 40449, 53599, 29015 |
| `open_issues` | integer | 0.0% | 657 | 241, 48034, 0 |
| `size_kb` | integer | 0.0% | 6,699 | 54255, 1119035, 255632 |
| `license` | string | 23.0% | 25 | MIT, Apache-2.0, AGPL-3.0 |
| `topics` | string | 0.0% | 7,500 | ai-agents/anthropic/claude/c, ai/ai-agent/ai-agents/anthro, ai-agents/cordis/dsh/dsh-plu |
| `n_topics` | integer | 0.0% | 20 | 8, 13, 4 |
| `matched_topics` | string | 0.0% | 643 | ai-agents/llm, ai-agents, deep-learning/machine-learni |
| `created_at` | datetime | 0.0% | 7,550 | 2026-01-18T00:51:51Z, 2025-07-22T22:22:28Z, 2026-08-13T11:56:32Z |
| `pushed_at` | datetime | 0.0% | 7,526 | 2026-09-30T18:45:04Z, 2026-10-01T17:08:04Z, 2026-09-29T09:41:57Z |
| `age_days` | integer | 0.0% | 3,424 | 256, 435, 49 |
| `days_since_push` | integer | 0.0% | 1,921 | 0, 2, 213 |
| `archived` | boolean | 0.0% | 2 | False, True |
| `is_fork` | boolean | 0.0% | 1 | False |
| `has_homepage` | boolean | 0.0% | 2 | True, False |

## What the starter notebook finds

`explore.ipynb` computes these from the data:

- Attention is concentrated: the top 10% of repositories hold 61% of all stars
  (Gini 0.69).
- Agents took over: among popular repos created in 2026 so far, 66% are tagged
  `ai-agents`, up from 5% of those created in 2023.
- TypeScript's share of new popular repos rose from 9% (2023) to 25% (2026).
- 43% have not been pushed to in a year; 5% are archived.
- 70% use a permissive license, 23% have none or an unclear one, including 19
  of the 100 most-starred.

## Caveats

- GitHub search returns at most 1,000 results per query, so `ai-agents`, `llm`,
  `deep-learning`, `machine-learning` and `pytorch` are capped at their 1,000
  most-starred repos. Per-topic counts are lower bounds.
- Repositories created this year have had less time to collect stars.

## Reproducing

`python fetch_dataset.py` (needs the GitHub CLI, signed in) takes about eight
minutes and writes a fresh snapshot. A new run gives different numbers; that is
the point of dating the snapshot.

## License and source

Repository metadata from GitHub's public REST API. Descriptions belong to their
authors; use is subject to GitHub's Terms of Service.

## Suggested Use Cases

Predicting stars from early signals, clustering repositories by topic sets,
studying maintenance decay, and comparing against later snapshots.

## Tags

artificial intelligence, programming, software, exploratory data analysis, deep learning, tabular
