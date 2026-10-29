# GitHub ML & AI Repositories (2026 Snapshot)

**Kaggle:** [lorenzoscaturchio/github-ml-ai-repositories-2026](https://www.kaggle.com/datasets/lorenzoscaturchio/github-ml-ai-repositories-2026) (not yet published)

## Description

Machine learning and AI repositories on GitHub with at least 500 stars across
20 topics, pulled from GitHub's public search API on 2026-10-01. This is real
data, not generated: 7,550 repositories, one row each. Fifteen topics are
complete down to 500 stars. The five largest hit the API's 1,000-result cap,
and four of them are complete only down to a higher floor (see Caveats).

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

`explore.ipynb` computes these from the full CSV:

- Attention is concentrated: the top 10% of repositories hold 61% of all stars
  (Gini 0.69).
- Agents took over: among popular repos created in 2026 so far, 66% are tagged
  `ai-agents`, up from 5% of those created in 2023.
- TypeScript's share of new popular repos rose from 9% (2023) to 25% (2026).
- 43% have not been pushed to in a year; 5% are archived.
- 70% use a permissive license, 23% have none or an unclear one, including 19
  of the 100 most-starred.

The agent and TypeScript trends hold on the complete subset described below.
The `llm` trend and the concentration figures do not: they depend on the
uneven star floor.

## Caveats

- **The star floor is not 500 for every topic.** GitHub search returns at most
  1,000 results per query, sorted by stars, so the five largest topics stop at
  their 1,000th repo. The lowest star count each one reaches in the CSV:
  `pytorch` 953, `llm` 1,594, `deep-learning` 1,659, `machine-learning` 1,794.
  `ai-agents` reaches 501, so it is only barely cut off. An `llm` repo with
  600 stars is missing, but a `rag` repo with 600 stars is included.
- **What that changes.** Restricting to repos with at least 1,794 stars, the
  level at which every topic is complete (3,429 repos), gives these figures,
  measured offline on this CSV with pandas 3.1:

  | Figure | Full CSV | Complete subset (≥1,794 stars) |
  |---|---:|---:|
  | `llm` share of repos created in 2023 / 2026 | 29% / 24% | 54% / 54% |
  | `ai-agents` share, 2023 / 2026 | 5% / 66% | 5.5% / 54% |
  | TypeScript share, 2023 / 2026 | 9% / 25% | 10% / 28% |
  | Top 10% share of stars; Gini | 61%; 0.69 | 51%; 0.59 |
  | Not pushed to in a year | 43% | 34% |
  | Permissive / none or unclear license | 70% / 23% | 71% / 22% |

  The full CSV understates `llm` by about half, because it drops `llm` repos
  below 1,594 stars but keeps every other kind down to 500. On the complete
  subset, `llm` holds steady at about 54% of each year's new repos, and the
  agent rise is smaller than the headline figure. The concentration figures
  depend on where the floor sits, so treat them as properties of this sample,
  not of GitHub.
- **Staleness mixes in young repos.** A repo created less than a year ago
  cannot be a year stale. Among the 6,504 repos older than a year, 50% have
  not been pushed to in a year.
- Repositories created this year have had less time to collect stars, so the
  2026 cohort is the survivors of a shorter race.

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
