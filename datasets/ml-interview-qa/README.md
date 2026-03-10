# ML/DS Interview Questions & Answers (500+)

> 502 synthetic machine learning and data science interview questions with detailed answers, topic labels, and difficulty levels.

**Kaggle dataset:** [lorenzoscaturchio/ml-interview-qa](https://www.kaggle.com/datasets/lorenzoscaturchio/ml-interview-qa)  
**License:** GPL-3.0

## Overview

This dataset is structured for technical interview preparation and NLP workflows. It covers 10 categories including statistics, ML theory, deep learning, NLP, computer vision, system design, SQL, Python, feature engineering, and A/B testing.

Each row contains:
- a unique question
- an expert-style answer
- a coarse category
- a difficulty label
- company tags
- fine-grained topic tags

The content is synthetic and intended for study, retrieval, classification, and RAG prototyping.

## Quick Facts

| Property | Value |
|---|---|
| File | `ml_interview_questions.csv` |
| Rows | `502` |
| Columns | `8` |
| Categories | `10` |
| Difficulty levels | `3` |
| Coverage | `2020-01-01` to `2025-12-31` |
| Geography | `Global (synthetic)` |

## Recommended Use Cases

1. Interview-prep search or filtering by category, topic, or company.
2. Text classification by category or difficulty.
3. Embedding and semantic-search demos.
4. Retrieval-augmented generation prototypes for technical Q&A.
5. LLM evaluation or fine-tuning experiments on concise expert answers.

## Schema

| Column | Meaning |
|---|---|
| `id` | unique question identifier |
| `question` | interview question text |
| `answer` | detailed answer written in expert style |
| `category` | broad topic area |
| `difficulty` | `easy`, `medium`, or `hard` |
| `company_tags` | pipe-separated companies associated with the question style |
| `topic_tags` | pipe-separated fine-grained topic labels |
| `answer_length` | answer word count |

## Practical Notes

- `company_tags` is useful for targeted interview-prep slices.
- `topic_tags` supports multi-label or retrieval experiments.
- `answer_length` makes it easy to separate flashcard-style answers from longer explanations.
- Because the content is synthetic, it is safer for educational benchmarking than scraped interview corpora.

## Provenance

- Synthetic data generation scripts in this repository
- Public schema conventions and domain patterns for educational simulation
- Programmatic generation with seeded constraints to preserve structure while avoiding personal data

## Citation

Scaturchio, Lorenzo (2026). *ML/DS Interview Questions & Answers (500+).* Kaggle Dataset. <https://www.kaggle.com/datasets/lorenzoscaturchio/ml-interview-qa>
