# Dataset Metadata Guide

This repo already stores Kaggle-ready dataset descriptors and provenance in each `dataset-metadata.json`.

## Where Kaggle Reads These Fields

Column descriptors live under:

```json
"resources": [
  {
    "path": "file.csv",
    "description": "Short file summary",
    "schema": {
      "fields": [
        {
          "name": "column_name",
          "title": "Human Title",
          "description": "What the column means",
          "type": "string"
        }
      ]
    }
  }
]
```

Provenance lives under:

```json
"provenance": {
  "sources": [
    "Synthetic data generation scripts in this repository",
    "Public domain schemas and domain conventions for educational simulation"
  ],
  "collection_methodology": "Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data."
}
```

## Current Datasets

Each of the following already includes:
- column descriptors in `resources[].schema.fields[]`
- provenance in `provenance`
- author metadata in `authors`
- temporal/geospatial context in `coverage`

| Dataset | Metadata File |
|---|---|
| AI/ML Research Papers Trends | `datasets/ai-research-trends/dataset-metadata.json` |
| Credit Card Fraud Detection | `datasets/credit-card-fraud/dataset-metadata.json` |
| E-Commerce Behavior | `datasets/ecommerce-behavior/dataset-metadata.json` |
| GitHub Repo Metrics | `datasets/github-repo-metrics/dataset-metadata.json` |
| Job Postings | `datasets/job-postings/dataset-metadata.json` |
| Mental Health in Tech | `datasets/mental-health-tech/dataset-metadata.json` |
| ML/DS Interview Q&A | `datasets/ml-interview-qa/dataset-metadata.json` |
| Programming Benchmarks | `datasets/programming-benchmarks/dataset-metadata.json` |
| Spotify Tracks | `datasets/spotify-tracks/dataset-metadata.json` |
| Student Performance | `datasets/student-performance/dataset-metadata.json` |

## Default Provenance Text

Use this unless a dataset has a special source story:

### Sources

```text
Synthetic data generation scripts in this repository
Public domain schemas and domain conventions for educational simulation
```

### Collection Methodology

```text
Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data.
```

## Recommended Workflow

1. Edit the relevant `dataset-metadata.json`.
2. Keep every resource file listed in `resources`.
3. Add one `schema.fields[]` entry per column for every CSV resource.
4. Run `bash manage.sh validate`.
5. Publish with `kaggle datasets version -p <dataset-dir> -m "..."`.

## Repo Guardrail

`manage.sh validate` now enforces dataset:
- `resources[].description`
- `resources[].schema.fields[]`
- `authors`
- `coverage`
- `provenance.sources`
- `provenance.collection_methodology`

That means future datasets will fail validation if these sections are missing.
