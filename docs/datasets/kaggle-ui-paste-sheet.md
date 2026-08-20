# Kaggle UI Paste Sheet

Use this when Kaggle asks for `Sources`, `Collection methodology`, and column descriptors in the dataset UI.

## Common Provenance Text

### Sources

- `Synthetic data generation scripts in this repository`
- `Public domain schemas and domain conventions for educational simulation`

### Collection Methodology

`Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data.`

## Dataset Checklist

### AI/ML Research Papers Trends (3K+ Papers)

- Dataset ref: `lorenzoscaturchio/ai-ml-research-papers-trends`
- Metadata file: `datasets/ai-research-trends/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `ai_research_papers.csv`
- Field count: `15`

### Credit Card Fraud Detection (200K Transactions)

- Dataset ref: `lorenzoscaturchio/credit-card-fraud-detection-synthetic`
- Metadata file: `datasets/credit-card-fraud/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `credit_card_transactions.csv`
- Field count: `36`

### Synthetic E-Commerce Customer Behavior Dataset

- Dataset ref: `lorenzoscaturchio/ecommerce-behavior`
- Metadata file: `datasets/ecommerce-behavior/dataset-metadata.json`
- Column descriptors copy from:
  - `resources[0].schema.fields` for `customers.csv`
  - `resources[1].schema.fields` for `products.csv`
  - `resources[2].schema.fields` for `transactions.csv`
  - `resources[3].schema.fields` for `sessions.csv`
  - `resources[4].schema.fields` for `reviews.csv`
- Resource files:
  - `customers.csv` -> `10` fields
  - `products.csv` -> `11` fields
  - `transactions.csv` -> `11` fields
  - `sessions.csv` -> `10` fields
  - `reviews.csv` -> `8` fields

### GitHub Repository Metrics Dataset (5K+ Repos)

- Dataset ref: `lorenzoscaturchio/github-repo-metrics`
- Metadata file: `datasets/github-repo-metrics/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `github_repos.csv`
- Field count: `29`

### Job Postings: NLP & Salary Prediction (15K)

- Dataset ref: `lorenzoscaturchio/job-postings-nlp-salary-prediction`
- Metadata file: `datasets/job-postings/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `job_postings.csv`
- Field count: `16`

### Mental Health in Tech Survey (5K Responses)

- Dataset ref: `lorenzoscaturchio/mental-health-in-tech-survey-5k`
- Metadata file: `datasets/mental-health-tech/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `mental_health_tech.csv`
- Field count: `27`

### ML/DS Interview Questions & Answers (Curated)

- Dataset ref: `lorenzoscaturchio/ml-interview-qa`
- Metadata file: `datasets/ml-interview-qa/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `ml_interview_questions.csv`
- Field count: `7`

### Programming Language Benchmarks Dataset

- Dataset ref: `lorenzoscaturchio/programming-language-benchmarks`
- Metadata file: `datasets/programming-benchmarks/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `language_benchmarks.csv`
- Field count: `11`

### Spotify Tracks: Audio Features (50K Songs)

- Dataset ref: `lorenzoscaturchio/spotify-tracks-audio-features-50k`
- Metadata file: `datasets/spotify-tracks/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `spotify_tracks.csv`
- Field count: `21`

### Student Academic Performance (10K Students)

- Dataset ref: `lorenzoscaturchio/student-academic-performance-dataset`
- Metadata file: `datasets/student-performance/dataset-metadata.json`
- Column descriptors copy from: `resources[0].schema.fields`
- Resource file: `students.csv`
- Field count: `25`

## Fastest Workflow

1. Open the dataset page in Kaggle.
2. Paste the common `Sources`.
3. Paste the common `Collection methodology`.
4. Open the matching `dataset-metadata.json`.
5. Copy the `resources[].schema.fields[]` entries for that dataset's files into the Kaggle column descriptor UI.
6. Save, then move to the next dataset.

## Important Note

The current Kaggle CLI dataset version flow does not appear to publish `resources.schema.fields`, `authors`, `coverage`, or `provenance`. That is why these sections still need manual UI entry even though they already exist in this repo.
