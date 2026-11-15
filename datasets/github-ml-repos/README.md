# GitHub ML & AI Repositories (2026 Snapshot)

**Kaggle:** [lorenzoscaturchio/github-ml-ai-repositories-2026](https://www.kaggle.com/datasets/lorenzoscaturchio/github-ml-ai-repositories-2026)

## Description

Machine learning and AI repositories on GitHub with at least 500 stars across
20 topics, pulled from GitHub's public search API on 2026-11-15. This is real
data, not generated: 7,636 repositories, one row each. Fifteen topics are
complete down to 500 stars. The five largest hit the API's 1,000-result cap,
and each of them is complete only down to a higher floor (see Caveats).

## Files

| File | Rows | Contents |
|------|------|----------|
| `github_ml_repos.csv` | 7,636 | Stars, forks, issues, size, language, license, topics, dates and derived age/staleness |

## Columns

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `repo` | string | 0.0% | 7,636 | affaan-m/ECC, deepseek-ai/deepseek-harness, NousResearch/hermes-agent |
| `owner` | string | 0.0% | 5,618 | affaan-m, deepseek-ai, NousResearch |
| `owner_type` | string | 0.0% | 2 | User, Organization |
| `description` | string | 0.2% | 7,615 | The agent harness performanc, DeepSeek Harness: Everything, The agent that grows with yo |
| `language` | string | 8.7% | 64 | JavaScript, TypeScript, Python |
| `stars` | integer | 0.0% | 4,087 | 303748, 281572, 263612 |
| `forks` | integer | 0.0% | 1,901 | 44771, 34821, 59426 |
| `open_issues` | integer | 0.0% | 676 | 282, 0, 53304 |
| `size_kb` | integer | 0.0% | 6,818 | 62275, 389740, 1462859 |
| `license` | string | 22.8% | 25 | MIT, AGPL-3.0, Apache-2.0 |
| `topics` | string | 0.0% | 7,586 | ai-agents/anthropic/claude/c, ai-agents/cordis/dsh/dsh-plu, ai/ai-agent/ai-agents/anthro |
| `n_topics` | integer | 0.0% | 20 | 8, 4, 13 |
| `matched_topics` | string | 0.0% | 643 | ai-agents/llm, ai-agents, deep-learning/machine-learni |
| `created_at` | datetime | 0.0% | 7,636 | 2026-01-18T00:51:51Z, 2026-08-13T11:56:32Z, 2025-07-22T22:22:28Z |
| `pushed_at` | datetime | 0.0% | 7,626 | 2026-11-15T06:48:46Z, 2026-11-13T07:56:25Z, 2026-11-15T20:36:06Z |
| `age_days` | integer | 0.0% | 3,480 | 301, 94, 480 |
| `days_since_push` | integer | 0.0% | 1,933 | 0, 2, 17 |
| `archived` | boolean | 0.0% | 2 | False, True |
| `is_fork` | boolean | 0.0% | 1 | False |
| `has_homepage` | boolean | 0.0% | 2 | True, False |

## What the starter notebook finds

`explore.ipynb` computes these from the full CSV:

- Attention is concentrated: the top 10% of repositories hold 62% of all stars
  (Gini 0.70).
- Agents took over: among popular repos created in 2026 so far, 61% are tagged
  `ai-agents`, up from 5% of those created in 2023.
- TypeScript's share of new popular repos rose from 9% (2023) to 24% (2026).
- 43% have not been pushed to in a year; 5% are archived.
- 70% use a permissive license, 23% have none or an unclear one, including 17
  of the 100 most-starred.

The agent and TypeScript trends hold on the complete subset described below.
The `llm` trend and the concentration figures do not: they depend on the
uneven star floor.

## Caveats

- **The star floor is not 500 for every topic.** GitHub search returns at most
  1,000 results per query, sorted by stars, so the five largest topics stop at
  their 1,000th repo. The lowest star count each one reaches in the CSV:
  `ai-agents` 708, `pytorch` 962, `deep-learning` 1,673, `llm` 1,779,
  `machine-learning` 1,800. An `llm` repo with 600 stars is missing, but a
  `rag` repo with 600 stars is included.
- **What that changes.** Restricting to repos with at least 1,800 stars, the
  level at which every topic is complete (3,589 repos), gives these figures,
  measured offline on this CSV with pandas 3.1:

  | Figure | Full CSV | Complete subset (≥1,800 stars) |
  |---|---:|---:|
  | `llm` share of repos created in 2023 / 2026 | 27% / 24% | 54% / 51% |
  | `ai-agents` share, 2023 / 2026 | 5% / 61% | 6% / 55% |
  | TypeScript share, 2023 / 2026 | 9% / 24% | 10% / 27% |
  | Top 10% share of stars; Gini | 62%; 0.70 | 52%; 0.60 |
  | Not pushed to in a year | 43% | 33% |
  | Permissive / none or unclear license | 70% / 23% | 71% / 22% |

  The full CSV understates `llm` by about half, because it drops `llm` repos
  below 1,779 stars but keeps the fifteen uncapped topics down to 500. On the
  complete subset, `llm` stays between 51% and 56% of each year's new repos,
  and the agent rise is smaller than the headline figure. The concentration figures
  depend on where the floor sits, so treat them as properties of this sample,
  not of GitHub.
- **Staleness mixes in young repos.** A repo created less than a year ago
  cannot be a year stale. Among the 6,587 repos older than a year, 49% have
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
