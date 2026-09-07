# AI & Data Jobs: Skills + Salaries (2024-2026)

> 36K listings plus company profiles, job-skill edges, and salary benchmarks

**License:** GPL-3.0  

**Kaggle:** [lorenzoscaturchio/ai-data-jobs-skills-salaries-2024-2026](https://www.kaggle.com/datasets/lorenzoscaturchio/ai-data-jobs-skills-salaries-2024-2026)  

## Description

A multi-table synthetic jobs market dataset covering 36,000 AI and data job listings from January 2024 through March 2026. Focuses on the roles people actually search for right now: Data Scientist, ML Engineer, AI Engineer, LLM Engineer, Data Engineer, Analytics Engineer, MLOps, BI, Research, and AI Product roles across North America, Europe, and APAC.

The dataset includes five linked tables: jobs, companies, normalized job-skill edges, salary benchmarks, and monthly skill-demand trends. Compensation is modeled in USD with realistic effects from seniority, region, industry, remote policy, and company size. GenAI / LLM roles become more common in 2025-2026, and startups trade higher equity incidence for lower cash compensation.

Built for salary prediction, market trend analysis, skill-gap analysis, retrieval and recommendation systems, compensation benchmarking, and tabular or graph ML on hiring data. All records are synthetic and generated for educational and prototyping use.

## Tags

`jobs`, `business`, `regression`, `classification`, `nlp`

## Authors

- **Lorenzo Scaturchio**: Independent ML engineer building synthetic, education-first datasets for reproducible benchmarking and prototyping.

## Coverage

- Temporal: 2024-01-01 to 2026-03-31
- Geospatial: North America, Europe, APAC (synthetic)

## DOI and Citations

- DOI: Not assigned
- Scaturchio, Lorenzo (2026). AI & Data Jobs: Skills + Salaries (2024-2026). Kaggle Dataset. https://www.kaggle.com/datasets/lorenzoscaturchio/ai-data-jobs-skills-salaries-2024-2026

## Provenance

- Source: Synthetic data generation scripts in this repository
- Source: Public domain job-market schemas and compensation conventions for educational simulation
- Collection methodology: Programmatic synthetic generation using seeded probabilistic rules to simulate realistic relationships between compensation, location, company type, seniority, and skill demand while avoiding direct personal or employer data.

## companies.csv

**Rows:** 280  |  **Columns:** 15  |  **Size:** 33.9 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `company_id` | string | 0.0% | 280 | `COMP0001`, `COMP0002`, `COMP0003` |
| `company_name` | string | 0.0% | 280 | `Apex Media`, `Northstar Bio`, `Signal Consumer` |
| `industry` | string | 0.0% | 10 | `finance`, `retail`, `consulting` |
| `company_type` | string | 0.0% | 27 | `bank`, `advisory`, `marketplace` |
| `company_size` | string | 0.0% | 5 | `medium`, `small`, `startup` |
| `funding_stage` | string | 0.0% | 8 | `series-c`, `series-a`, `late-stage` |
| `hq_city` | string | 0.0% | 18 | `Amsterdam, Netherlands`, `Sydney, Australia`, `New York, NY` |
| `country` | string | 0.0% | 11 | `United States`, `India`, `Australia` |
| `region` | string | 0.0% | 3 | `North America`, `Europe`, `APAC` |
| `employee_count_estimate` | integer | 0.0% | 251 | `401`, `33797`, `2783` |
| `remote_first` | integer | 0.0% | 2 | `0`, `1` |
| `ai_maturity_score` | float | 0.0% | 211 | `59.5`, `47.7`, `53.8` |
| `benefits_score` | float | 0.0% | 194 | `59.8`, `75.7`, `77.3` |
| `glassdoor_like_rating` | float | 0.0% | 121 | `3.69`, `3.56`, `4.03` |
| `hiring_velocity_index` | integer | 0.0% | 62 | `44`, `68`, `61` |

## job_skills.csv

**Rows:** 5,000  |  **Columns:** 6  |  **Size:** 14,522.8 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `job_id` | string | 0.0% | 569 | `JOB000001`, `JOB000008`, `JOB000005` |
| `skill` | string | 0.0% | 70 | `Python`, `Azure`, `BigQuery` |
| `skill_category` | string | 0.0% | 13 | `data-platform`, `cloud`, `analytics` |
| `importance` | string | 0.0% | 3 | `core`, `strong`, `nice_to_have` |
| `proficiency_level` | string | 0.0% | 3 | `advanced`, `intermediate`, `basic` |
| `is_genai_skill` | integer | 0.0% | 2 | `0`, `1` |

## jobs.csv

**Rows:** 5,000  |  **Columns:** 37  |  **Size:** 19,784.9 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `job_id` | string | 0.0% | 5,000 | `JOB000001`, `JOB000002`, `JOB000003` |
| `company_id` | string | 0.0% | 280 | `COMP0131`, `COMP0189`, `COMP0241` |
| `company_name` | string | 0.0% | 280 | `Signal Academy`, `Foundry Advisory 189`, `Apex Learning 241` |
| `posted_date` | string | 0.0% | 817 | `2025-07-22`, `2024-11-08`, `2025-10-27` |
| `job_title` | string | 0.0% | 12 | `Data Engineer`, `LLM Engineer`, `MLOps Engineer` |
| `role_family` | string | 0.0% | 8 | `analytics`, `modeling`, `genai-apps` |
| `seniority` | string | 0.0% | 5 | `mid`, `senior`, `entry` |
| `employment_type` | string | 0.0% | 3 | `full_time`, `contract`, `internship` |
| `remote_type` | string | 0.0% | 3 | `remote`, `hybrid`, `onsite` |
| `city` | string | 0.0% | 18 | `New York, NY`, `San Francisco, CA`, `Seattle, WA` |
| `country` | string | 0.0% | 11 | `United States`, `India`, `United Kingdom` |
| `region` | string | 0.0% | 3 | `North America`, `Europe`, `APAC` |
| `industry` | string | 0.0% | 10 | `software`, `consulting`, `finance` |
| `company_size` | string | 0.0% | 5 | `medium`, `small`, `startup` |
| `funding_stage` | string | 0.0% | 8 | `series-c`, `series-a`, `late-stage` |
| `company_type` | string | 0.0% | 27 | `edtech`, `developer-tools`, `consumer` |
| `salary_currency` | string | 0.0% | 1 | `USD` |
| `salary_min_usd` | integer | 0.0% | 265 | `139000`, `183000`, `71000` |
| `salary_max_usd` | integer | 0.0% | 352 | `191000`, `257000`, `91000` |
| `salary_mid_usd` | integer | 0.0% | 306 | `104000`, `157000`, `212000` |
| `bonus_target_pct` | float | 0.0% | 127 | `0.093`, `0.079`, `0.077` |
| `equity_offered` | integer | 0.0% | 2 | `0`, `1` |
| `experience_min_years` | integer | 0.0% | 10 | `2`, `5`, `3` |
| `experience_max_years` | integer | 0.0% | 10 | `8`, `9`, `6` |
| `education_required` | string | 0.0% | 4 | `master`, `bachelor`, `phd` |
| `visa_sponsorship` | integer | 0.0% | 2 | `0`, `1` |
| `ai_focus_area` | string | 0.0% | 45 | `batch-pipelines`, `operations`, `governance` |
| `primary_skill_cluster` | string | 0.0% | 8 | `analytics`, `ml`, `llm` |
| `cloud_stack` | string | 0.0% | 6 | `Azure|Databricks`, `Azure`, `GCP|BigQuery` |
| `llm_stack` | string | 76.9% | 4 | `Anthropic API|DSPy|pgvector`, `OpenAI API|vLLM|LLM Evaluation`, `OpenAI API|LangChain|Pinecone` |
| `required_skills` | string | 0.0% | 4,999 | `SQL|Python|Statistics|Excel|Azure|Pandas|Looker|Tableau`, `SQL|BigQuery|Looker|Data Modeling|dbt|Snowflake|GCP`, `Python|Airflow|Azure|Databricks|Data Modeling|Snowflake|SQL` |
| `skills_count` | integer | 0.0% | 7 | `8`, `9`, `10` |
| `applications_30d` | integer | 0.0% | 92 | `53`, `57`, `28` |
| `days_open` | integer | 0.0% | 62 | `32`, `28`, `38` |
| `is_filled` | integer | 0.0% | 2 | `1`, `0` |
| `hiring_urgency` | string | 0.0% | 3 | `medium`, `low`, `high` |
| `description` | string | 0.0% | 5,000 | — |

## salary_benchmarks.csv

**Rows:** 618  |  **Columns:** 10  |  **Size:** 44.3 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `job_title` | string | 0.0% | 12 | `Data Scientist`, `Data Engineer`, `Data Analyst` |
| `country` | string | 0.0% | 10 | `United States`, `India`, `United Kingdom` |
| `seniority` | string | 0.0% | 5 | `mid`, `senior`, `entry` |
| `remote_type` | string | 0.0% | 3 | `hybrid`, `remote`, `onsite` |
| `sample_jobs` | integer | 0.0% | 129 | `20`, `21`, `27` |
| `salary_p10_usd` | integer | 0.0% | 465 | `124000`, `96000`, `125600` |
| `salary_median_usd` | integer | 0.0% | 294 | `143000`, `151500`, `105000` |
| `salary_p90_usd` | integer | 0.0% | 494 | `171000`, `121000`, `166400` |
| `bonus_target_pct_median` | float | 0.0% | 72 | `0.093`, `0.094`, `0.09` |
| `visa_share` | float | 0.0% | 194 | `0.15`, `0.238`, `0.111` |

## skill_demand_monthly.csv

**Rows:** 1,080  |  **Columns:** 7  |  **Size:** 51.9 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `year_month` | string | 0.0% | 27 | `2024-01`, `2024-02`, `2024-03` |
| `skill` | string | 0.0% | 40 | `Python`, `SQL`, `Azure` |
| `skill_category` | string | 0.0% | 12 | `analytics`, `data-platform`, `ml` |
| `job_count` | integer | 0.0% | 422 | `678`, `466`, `300` |
| `median_salary_mid_usd` | integer | 0.0% | 116 | `147000`, `141000`, `140000` |
| `remote_share` | float | 0.0% | 182 | `0.416`, `0.391`, `0.457` |
| `share_of_postings` | float | 0.0% | 836 | `0.8794`, `0.6044`, `0.3891` |

## Suggested Use Cases

- Text classification (TF-IDF, BERT embeddings)
- Named entity recognition or topic modeling
- Salary prediction (regression)
- Job category classification (multi-class)

---
*Generated by `dataset_optimizer.py` — dataset_optimizer.py*
