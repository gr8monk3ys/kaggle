# Kaggle Portfolio - Claude Code Guide

## Project Structure

This is a monorepo containing Kaggle competition entries, educational notebooks, datasets, and community engagement strategy documents.

```
kaggle/
├── manage.sh                          # CLI wrapper for kaggle_portfolio (44 subcommands)
├── kaggle_portfolio/                  # Python package behind manage.sh (CLI, ops, quality, campaigns)
├── tests/                             # pytest suite for the kaggle_portfolio package
├── medal_ops/                         # Generated scorecards/plans/reports (gitignored except README)
├── pi-automation/                     # Dockerized Playwright/cron automation for Kaggle engagement
├── docs/
│   ├── reports/
│   │   ├── grandmaster-tracker.md     # Progress across 4 Grandmaster categories (synced via medal_ops)
│   │   └── competition-scout-report.md # Active competition analysis (regenerated via scout)
│   ├── discussions/
│   │   ├── engagement-strategy.md     # 12-week community plan
│   │   └── discussion-drafts.md       # Pre-written discussion posts
│   └── superpowers/                   # Design specs and implementation plans
│
├── projects/
│   ├── competitions/
│   │   ├── med-gemma-challenge/      # MedGemma 4B chest X-ray triage (LoRA)
│   │   ├── akkadian-translation/     # ByT5 seq2seq ancient language translation
│   │   ├── vesuvius-surface/         # 3D U-Net volumetric segmentation
│   │   ├── spaceship-titanic/        # Binary classification with feature engineering
│   │   ├── titanic-ultimate/         # Titanic: From Zero to Top 5%
│   │   ├── store-sales-forecasting/  # Time series forecasting with LightGBM
│   │   ├── nlp-disaster-tweets/      # BERT-based disaster tweet classification
│   │   ├── house-prices/             # House Prices: Complete EDA + Feature Engineering
│   │   └── digit-recognizer/         # Digit Recognizer: CNN from Scratch to 99%+
│   └── educational/
│       ├── feature-engineering/      # 50 techniques across 8 categories
│       ├── attention-guide/          # Bahdanau → Transformer walkthrough
│       ├── llm-finetuning/           # LoRA/QLoRA practical guide
│       ├── image-segmentation/       # U-Net → SegFormer masterclass
│       ├── timeseries-transformers/  # Transformer time series forecasting
│       ├── ensemble-stacking/        # Competition-winning ensemble methods
│       ├── rag-from-scratch/         # RAG from first principles
│       ├── graph-neural-networks/    # GNN practical guide
│       ├── financial-analysis/       # Financial time-series prediction
│       ├── fraud-detection/          # Explainable fraud detection
│       ├── eda-tutorial/             # EDA best practices
│       ├── competition-template/     # Standardized ML pipeline template
│       ├── shap-explainability/      # SHAP Model Explainability Masterclass
│       ├── optuna-guide/             # Optuna Hyperparameter Optimization Guide
│       └── nlp-text-classification/  # NLP Text Classification: TF-IDF to BERT
│
└── datasets/                          # 8 custom Kaggle datasets
    ├── ml-interview-questions/
    ├── ecommerce-behavior/
    ├── github-metrics/
    ├── ai-research-papers/
    ├── programming-benchmarks/
    ├── credit-card-fraud/             # Credit Card Fraud Detection (200K transactions)
    ├── job-postings/                  # Job Postings NLP & Salary Prediction (15K listings)
    └── student-performance/           # Student Academic Performance (in progress)
```

## Conventions

- **Notebooks**: Each subfolder contains a Jupyter notebook (`.ipynb`) as the primary artifact, designed for Kaggle kernel execution
- **Competition entries** include both EDA and submission notebooks
- **Datasets** are in CSV/Parquet format, structured for Kaggle Datasets publishing
- **Strategy docs** are markdown files at the repo root level

## Management Script

`manage.sh` is the primary CLI tool for interacting with Kaggle. It wraps the
`kaggle_portfolio` package (dispatch table in `kaggle_portfolio/manage_commands.py`,
44 subcommands — run `./manage.sh help` for the full list). Run it from the repo root.

```bash
./manage.sh push <dir>                # Push a specific notebook/dataset directory
./manage.sh push-nb                   # Push all notebooks
./manage.sh push-ds                   # Push all datasets
./manage.sh status                    # Check publication status

./manage.sh doctor                    # Preflight checks (tracker, env, credentials)
./manage.sh sync --dry-run            # Preview tracker metric sync from live Kaggle
./manage.sh scorecard                 # Medal progress scorecard
./manage.sh weekly-plan               # 7-day execution plan
./manage.sh quality --min-score 70 --scope all   # Notebook quality rubric
./manage.sh scout --update            # Regenerate competition-scout-report.md
```

## Key Context

- **Goal**: Kaggle Grandmaster across all 4 categories (Competitions, Notebooks, Datasets, Discussion)
- **Current status**: 70 notebooks live (3 bronze), 12 datasets published (1 silver, 1 bronze), 12 competitions entered — see `docs/reports/grandmaster-tracker.md` for live numbers (synced 2026-06-11)
- **Priority competitions**: AI Agent Security: Tool Attacks (best medal odds, 19 teams, Sep 1), Hull Tactical Market (Jun 25), Orbit Wars (Jun 23)
- **Discussion strategy**: 2-3 posts/week targeting 50+ bronze medals over 12 weeks

## Development Workflow

1. Develop notebooks locally or on Kaggle
2. Use `manage.sh` to push to Kaggle
3. Track progress in `grandmaster-tracker.md`
4. Scout new competitions via `competition-scout-report.md`
5. Engage community per `discussion-engagement-strategy.md`

## Dependencies

Each notebook declares its own dependencies. Common across projects:
- PyTorch, HuggingFace Transformers
- scikit-learn, pandas, numpy
- Plotly, matplotlib, seaborn
- Kaggle API (`kaggle` CLI)
