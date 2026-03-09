# Kaggle ML Portfolio

[![Medal Ops Health](https://github.com/gr8monk3ys/kaggle/actions/workflows/medal-ops-health.yml/badge.svg)](https://github.com/gr8monk3ys/kaggle/actions/workflows/medal-ops-health.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-lorenzoscaturchio-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/lorenzoscaturchio)

A systematic collection of Kaggle notebooks, competition entries, datasets, and community engagement resources targeting Kaggle Grandmaster status across all four categories.

Progress is tracked in [`grandmaster-tracker.md`](./grandmaster-tracker.md).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Competition Entries](#competition-entries)
- [Educational Notebooks](#educational-notebooks)
- [Datasets](#datasets)
- [Strategy and Tracking](#strategy--tracking)
- [CLI Reference (manage.sh)](#cli-reference-managesh)
- [Automation](#automation)
- [Authentication and Secrets](#authentication--secrets)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Python** | 3.9+ | Notebooks, `medal_ops.py`, `notebook_quality.py` |
| **Kaggle CLI** | latest | `pip install kaggle` -- push/pull notebooks and datasets |
| **Bash** | 3.2+ (macOS default OK) | `manage.sh` CLI |
| **GPU** (optional) | CUDA-capable or Kaggle T4/P100 | Required for LLM fine-tuning, image segmentation, and competition GPU notebooks |

Notebooks that require GPU acceleration are marked with `enable_gpu: true` in their `kernel-metadata.json`. These include: `llm-finetuning`, `image-segmentation`, `timeseries-transformers`, `med-gemma-challenge`, `akkadian-translation`, `nlp-text-classification`, and `nlp-disaster-tweets`.

---

## Getting Started

### 1. Clone and configure credentials

```bash
git clone https://github.com/gr8monk3ys/kaggle.git
cd kaggle

# Set up Kaggle API credentials (never commit kaggle.json)
mkdir -p ~/.kaggle
cp kaggle.json.example ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
# Edit ~/.kaggle/kaggle.json with your username and API key from https://www.kaggle.com/settings
```

### 2. Validate and push notebooks

```bash
chmod +x manage.sh

# Validate all metadata before pushing
./manage.sh validate

# Push a single notebook
./manage.sh push feature-engineering

# Push everything at once
./manage.sh push-all
```

### 3. Enter a competition

```bash
# Link a notebook to a competition
./manage.sh link-competition med-gemma-challenge medgemma-assistants-impact-challenge

# Browse active medal-eligible competitions
./manage.sh competitions
```

### 4. Track progress

```bash
./manage.sh scorecard     # Medal operations scorecard
./manage.sh weekly-plan   # Weekly execution plan
./manage.sh pace          # Velocity and ETA analysis
./manage.sh doctor        # Preflight environment checks
```

---

## Competition Entries

| Project | Competition | Teams | Approach | Framework | GPU |
|---------|-------------|-------|----------|-----------|-----|
| [med-gemma-challenge](./med-gemma-challenge) | Medical AI (MedGemma Impact) | 58 | MedGemma 4B chest X-ray triage with LoRA fine-tuning | PyTorch, HuggingFace, PEFT | Yes |
| [akkadian-translation](./akkadian-translation) | Ancient Language (Deep Past) | 1,321 | ByT5 seq2seq translation of Akkadian cuneiform | HuggingFace Transformers, ByT5 | Yes |
| [vesuvius-surface](./vesuvius-surface) | Scroll Detection (Vesuvius 2) | 759 | 3D U-Net volumetric segmentation | PyTorch, segmentation_models_pytorch | No |
| [spaceship-titanic](./spaceship-titanic) | Spaceship Titanic | -- | Binary classification with feature engineering | XGBoost, scikit-learn | No |
| [titanic-ultimate](./titanic-ultimate) | Titanic | -- | End-to-end guide: EDA to top 5% | scikit-learn, XGBoost, ensemble | No |
| [store-sales-forecasting](./store-sales-forecasting) | Store Sales Time Series | -- | Time series forecasting with LightGBM | LightGBM, pandas | No |
| [nlp-disaster-tweets](./nlp-disaster-tweets) | NLP Disaster Tweets | -- | BERT-based text classification | HuggingFace, BERT | Yes |
| [house-prices](./house-prices) | House Prices Regression | -- | EDA + feature engineering + stacked ensemble | XGBoost, LightGBM, scikit-learn | No |
| [digit-recognizer](./digit-recognizer) | Digit Recognizer (MNIST) | -- | CNN from scratch to 99%+ accuracy | PyTorch | No |

---

## Educational Notebooks

| Notebook | Topic | Key Techniques | Framework / Libraries | GPU |
|----------|-------|----------------|-----------------------|-----|
| [feature-engineering](./feature-engineering) | Feature Engineering Masterclass | 50 techniques across 8 categories | pandas, scikit-learn | No |
| [attention-guide](./attention-guide) | Attention Mechanisms | Bahdanau to Transformer, PyTorch implementations | PyTorch | No |
| [llm-finetuning](./llm-finetuning) | LLM Fine-Tuning | LoRA/QLoRA, 4-bit quantization, vLLM deployment | HuggingFace, PEFT, BitsAndBytes | Yes |
| [image-segmentation](./image-segmentation) | Segmentation Masterclass | U-Net to SegFormer, Dice/Focal/Tversky losses | PyTorch, segmentation_models_pytorch | Yes |
| [timeseries-transformers](./timeseries-transformers) | Time Series Forecasting | Temporal Fusion Transformer, Informer, Autoformer | PyTorch, HuggingFace | Yes |
| [ensemble-stacking](./ensemble-stacking) | Ensemble Methods | Competition-winning stacking, blending, model averaging | XGBoost, LightGBM, CatBoost | No |
| [rag-from-scratch](./rag-from-scratch) | RAG Systems | Building retrieval-augmented generation from scratch | LangChain, vector databases, embeddings | No |
| [graph-neural-networks](./graph-neural-networks) | Graph Neural Networks | GNN practical guide for structured data | PyTorch Geometric | No |
| [financial-analysis](./financial-analysis) | Financial Analysis | Stock market prediction, LSTM, technical analysis | PyTorch, pandas | No |
| [fraud-detection](./fraud-detection) | Fraud Detection | Explainable ML for credit card fraud, SMOTE | XGBoost, Random Forest, SHAP | No |
| [eda-tutorial](./eda-tutorial) | EDA Best Practices | End-to-end ML pipeline, house price prediction | scikit-learn, XGBoost, matplotlib | No |
| [competition-template](./competition-template) | Competition Template | Standardized ML pipeline for rapid competition entry | scikit-learn, XGBoost | No |
| [shap-explainability](./shap-explainability) | SHAP Explainability | Model interpretability masterclass | SHAP, XGBoost, scikit-learn | No |
| [optuna-guide](./optuna-guide) | Hyperparameter Optimization | Bayesian optimization with Optuna | Optuna, XGBoost, LightGBM | No |
| [nlp-text-classification](./nlp-text-classification) | NLP Text Classification | TF-IDF to BERT, full NLP pipeline | HuggingFace, BERT, scikit-learn | Yes |

---

## Datasets

All datasets live under [`datasets/`](./datasets). Each directory contains a `create_dataset.py` generator script, an `explore.ipynb` EDA notebook, and `dataset-metadata.json` for Kaggle publishing.

| Dataset | Directory | Rows | Description | Format |
|---------|-----------|------|-------------|--------|
| ML/DS Interview Q&A | [ml-interview-qa](./datasets/ml-interview-qa) | 502 | 500+ ML/DS interview questions with detailed answers across categories | CSV |
| E-Commerce Behavior | [ecommerce-behavior](./datasets/ecommerce-behavior) | 236K+ | Customers (10K), products (1K), reviews (25K), sessions (80K), transactions (120K) | 5 CSVs |
| GitHub Repo Metrics | [github-repo-metrics](./datasets/github-repo-metrics) | 5,500 | 5K+ repos with stars, forks, issues, languages, and activity metrics | CSV |
| AI Research Papers | [ai-research-trends](./datasets/ai-research-trends) | 3,200 | AI/ML papers 2018-2025 with citations, venues, methods for bibliometric analysis | CSV |
| Programming Benchmarks | [programming-benchmarks](./datasets/programming-benchmarks) | 2,200 | Language benchmark data: execution time, memory, throughput across tasks | CSV |
| Credit Card Fraud | [credit-card-fraud](./datasets/credit-card-fraud) | 200,000 | Synthetic credit card transactions with fraud labels for binary classification | CSV |
| Job Postings NLP | [job-postings](./datasets/job-postings) | 15,000 | Job listings with descriptions, skills, salary for NLP and salary prediction | CSV |
| Mental Health in Tech | [mental-health-tech](./datasets/mental-health-tech) | 5,000 | Tech industry mental health survey responses and workplace factors | CSV |
| Spotify Tracks | [spotify-tracks](./datasets/spotify-tracks) | 50,000 | Audio features, popularity, genres, and metadata for music analysis | CSV |
| Student Performance | [student-performance](./datasets/student-performance) | 10,000 | Academic performance with demographics, study habits, and grades | CSV |

---

## Strategy & Tracking

| Document | Purpose |
|----------|---------|
| [grandmaster-tracker.md](./grandmaster-tracker.md) | Progress tracking across all 4 Grandmaster categories |
| [competition-scout-report.md](./competition-scout-report.md) | Active competition analysis and medal probability assessment |
| [discussion-engagement-strategy.md](./discussion-engagement-strategy.md) | 12-week community engagement roadmap |
| [discussion-drafts.md](./discussion-drafts.md) | Pre-written discussion posts for community engagement |
| [manage.sh](./manage.sh) | CLI tool for Kaggle notebook/dataset management |

---

## CLI Reference (manage.sh)

`manage.sh` is the primary CLI for all Kaggle operations. It auto-discovers notebook and dataset directories by scanning for `kernel-metadata.json` and `dataset-metadata.json` files.

```bash
chmod +x manage.sh
./manage.sh <command> [options]
```

### Commands

| Command | Description | Requires Kaggle CLI |
|---------|-------------|---------------------|
| `status` | Show all notebooks/datasets and their Kaggle account status | Yes |
| `push-all` | Push all notebooks and datasets (validates first) | Yes |
| `push-nb` | Push all notebooks only | Yes |
| `push-ds` | Push all datasets only | Yes |
| `push <dir>` | Push a specific notebook or dataset directory | Yes |
| `validate` | Validate all `kernel-metadata.json` files (JSON syntax, required fields, file existence, credential scan) | No |
| `votes` | Show vote counts with bronze/silver/gold medal threshold dashboard | Yes |
| `competitions` | List active medal-eligible featured and research competitions | Yes |
| `link-competition <dir> <slug>` | Add `competition_sources` to a notebook and re-push | Yes |
| `scorecard` | Generate medal operations scorecard (writes to `medal_ops/reports/`) | No |
| `weekly-plan` | Generate weekly execution plan report | No |
| `pace` | Generate velocity and ETA pace analysis from historical snapshots | No |
| `sync` | Sync tracker metrics from live Kaggle CLI data or CSV exports | Depends |
| `sync-template` | Generate CSV templates and export helper script for offline sync | No |
| `doctor` | Run preflight checks (tracker, sync inputs, environment) | No |
| `quality` | Score notebook quality against rubric (writes to `medal_ops/reports/`) | No |
| `dataset-usability` | Score dataset usability and emit actionable report (writes to `medal_ops/reports/`) | No |
| `usability-tracker` | Run live daily tracker with `0.8` alert gate and `1.0` target queue | Depends |
| `campaign-pack` | Generate multi-channel promotion campaign pack + queue from latest usability report | No |
| `campaign-run` | Execute queue operations (`show`, `claim`, `complete`) and export runbook | No |
| `post-discussion [--dry-run|--init|--schedule-weeks N]` | Post next queued discussion draft or rebuild a rolling scheduled window | No |
| `draft-ops` | Show draft stage counts, flow health, and prioritized backlog | No |
| `draft-set <id> [--status/--priority/--deadline]` | Update one draft's status/priority/deadline and rebalance schedule window | No |
| `help` | Show usage message | No |

### Examples

```bash
# Check portfolio status on Kaggle
./manage.sh status

# Validate metadata before pushing (catches invalid JSON, missing fields, embedded secrets)
./manage.sh validate

# Push a single notebook
./manage.sh push llm-finetuning

# Push a specific dataset
./manage.sh push datasets/credit-card-fraud

# View vote counts and medal proximity dashboard
./manage.sh votes

# Generate reports
./manage.sh scorecard
./manage.sh weekly-plan
./manage.sh pace

# Run environment health checks
./manage.sh doctor

# Score all notebooks against quality rubric (min score 95)
./manage.sh quality --min-score 95 --fail-under-threshold

# Score all datasets for usability quality
./manage.sh dataset-usability --strict --fail-under 85

# Run live usability tracker with 0.8 gate + 1.0 target
./manage.sh usability-tracker --fail-on-live-alert

# Build a 14-day promotion campaign pack + queue
./manage.sh campaign-pack --days 14 --posts-per-day 2

# Claim and export first 7 queue items for execution
./manage.sh campaign-run --limit 7 --claim --print-copy

# Rebuild draft queue with a 4-week scheduled window (rest stays ready)
./manage.sh post-discussion --init --schedule-weeks 4

# Review draft backlog flow health
./manage.sh draft-ops

# Update one draft and rebalance schedule window
./manage.sh draft-set draft_012 --priority high --deadline 2026-03-08

# Offline sync with exported CSV files
./manage.sh sync --dry-run \
  --kernels-csv /path/to/kernels.csv \
  --datasets-csv /path/to/datasets.csv \
  --competitions-csv /path/to/competitions.csv
```

Reports generated by `scorecard`, `weekly-plan`, `pace`, `doctor`, and `quality` are written to `medal_ops/reports/`.

---

## Automation

A daily GitHub Actions workflow ([`.github/workflows/medal-ops-health.yml`](./.github/workflows/medal-ops-health.yml)) runs automated health checks:

- **Schedule**: Daily at 09:10 UTC
- **Checks performed**: `doctor --strict`, `quality --fail-under-threshold`, `dataset-usability --strict`, `dataset-usability --daily-tracker`, `discussion_scheduler.py --health-check`, `sync --dry-run`
- **On failure**: Opens (or updates) a GitHub issue with logs and run link
- **On recovery**: Automatically closes the incident issue
- **Quality gate**: Minimum notebook quality score of 95 (configurable)
- **Dataset gate**: Minimum dataset usability score of 85 (configurable)
- **Stale threshold**: 30 days max before tracker data is flagged (configurable)

### Run modes

| Mode | Description |
|------|-------------|
| `auto` (default) | Uses `live` if `KAGGLE_USERNAME` and `KAGGLE_KEY` secrets are configured, otherwise falls back to `offline-fixture` |
| `live` | Queries Kaggle API directly (requires repository secrets) |
| `offline-fixture` | Uses synthetic CSV fixtures so pipeline breakages are still caught without API access |

Manual dispatch supports custom `mode`, `max_stale_days`, `min_quality_score`, `min_dataset_usability_score`, `live_alert_under`, `live_target_rating`, `max_overdue_scheduled`, and `max_days_until_next_post` inputs.

---

## Authentication & Secrets

```bash
# Keep credentials outside the repository
mkdir -p ~/.kaggle
cp kaggle.json.example ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Never commit API credentials. `kaggle.json` is gitignored in this repo.

For GitHub Actions live checks, add `KAGGLE_USERNAME` and `KAGGLE_KEY` in repository **Settings > Secrets and variables > Actions**.

---

## Project Structure

```
kaggle/
├── .github/
│   └── workflows/
│       └── medal-ops-health.yml        # Daily CI health checks
├── manage.sh                           # CLI for all Kaggle operations
├── medal_ops.py                        # Scorecard, weekly-plan, pace, sync, doctor
├── notebook_quality.py                 # Notebook quality scoring engine
├── build_utils.py                      # Shared notebook build utilities
├── grandmaster-tracker.md              # Progress across all 4 GM categories
├── competition-scout-report.md         # Active competition analysis
├── discussion-engagement-strategy.md   # 12-week community engagement plan
├── discussion-drafts.md                # Pre-written discussion posts
├── DISCUSSION_POSTS_READY.md           # Ready-to-post discussion index
├── kaggle.json.example                 # Credential template (never commit real keys)
│
├── Competition Entries (9 competitions)
│   ├── med-gemma-challenge/            # MedGemma 4B chest X-ray triage (LoRA)
│   ├── akkadian-translation/           # ByT5 seq2seq ancient language translation
│   ├── vesuvius-surface/               # 3D U-Net volumetric segmentation
│   ├── spaceship-titanic/              # Binary classification with feature engineering
│   ├── titanic-ultimate/               # Titanic: From Zero to Top 5%
│   ├── store-sales-forecasting/        # Time series forecasting with LightGBM
│   ├── nlp-disaster-tweets/            # BERT-based disaster tweet classification
│   ├── house-prices/                   # House Prices: EDA + Feature Engineering
│   └── digit-recognizer/              # CNN from Scratch to 99%+
│
├── Educational Notebooks (15 topics)
│   ├── feature-engineering/            # 50 techniques across 8 categories
│   ├── attention-guide/                # Bahdanau to Transformer walkthrough
│   ├── llm-finetuning/                # LoRA/QLoRA practical guide
│   ├── image-segmentation/             # U-Net to SegFormer masterclass
│   ├── timeseries-transformers/        # Transformer time series forecasting
│   ├── ensemble-stacking/             # Competition-winning ensemble methods
│   ├── rag-from-scratch/              # RAG from first principles
│   ├── graph-neural-networks/         # GNN practical guide
│   ├── financial-analysis/            # Financial time-series prediction
│   ├── fraud-detection/               # Explainable fraud detection
│   ├── eda-tutorial/                  # EDA best practices
│   ├── competition-template/          # Standardized ML pipeline template
│   ├── shap-explainability/           # SHAP Model Explainability Masterclass
│   ├── optuna-guide/                  # Optuna Hyperparameter Optimization
│   └── nlp-text-classification/       # NLP: TF-IDF to BERT
│
├── datasets/                           # 10 custom Kaggle datasets
│   ├── ml-interview-qa/               # 500+ ML/DS interview Q&A
│   ├── ecommerce-behavior/            # Multi-table e-commerce (236K+ rows)
│   ├── github-repo-metrics/           # 5.5K GitHub repos
│   ├── ai-research-trends/            # 3.2K AI/ML papers
│   ├── programming-benchmarks/        # 2.2K language benchmarks
│   ├── credit-card-fraud/             # 200K synthetic transactions
│   ├── job-postings/                  # 15K job listings
│   ├── mental-health-tech/            # 5K tech survey responses
│   ├── spotify-tracks/                # 50K audio features
│   └── student-performance/           # 10K student records
│
├── pi-automation/                      # Raspberry Pi automation (Docker)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── scripts/                       # Automation scripts
│   └── tests/                         # Automation tests
│
├── medal_ops/                          # Generated reports (gitignored)
│   ├── reports/                       # Scorecard, pace, quality reports
│   └── history/                       # Historical snapshots for trend analysis
│
├── tests/                              # Test suite
│   ├── test_medal_ops.py
│   ├── test_notebook_quality.py
│   ├── test_metadata_validate.py
│   ├── test_build_utils.py
│   └── test_repo_guardrails.py
│
├── docs/                               # Additional documentation
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE                             # MIT
```

---

## Tech Stack

- **Deep Learning**: PyTorch, HuggingFace Transformers, segmentation_models_pytorch, PyTorch Geometric
- **LLMs**: MedGemma, ByT5, LoRA/QLoRA, BitsAndBytes quantization, PEFT, vLLM serving
- **Computer Vision**: U-Net, DeepLabV3+, SegFormer, CNNs, Test-Time Augmentation
- **NLP**: Attention mechanisms, seq2seq, BERT, TF-IDF, FastText, ELMo, spaCy, LangChain
- **Classical ML**: XGBoost, LightGBM, CatBoost, scikit-learn, ensemble stacking
- **Explainability**: SHAP, feature importance analysis
- **Optimization**: Optuna, Bayesian hyperparameter tuning
- **Visualization**: Plotly, matplotlib, seaborn
- **Data**: pandas, NumPy
- **Tools**: Kaggle CLI, GGUF export, Docker (pi-automation)
- **CI/CD**: GitHub Actions (daily health checks, auto-issue management)
- **Testing**: pytest (portfolio + automation tests covering medal ops, quality scoring, metadata validation, repo guardrails, dataset optimization, scheduling, and pipeline checks)

---

## Tests

Run the test suite with:

```bash
python -m pytest tests/ -v
```

The test suite covers:

| Module | What it tests |
|--------|---------------|
| `test_medal_ops.py` | Scorecard generation, sync, weekly-plan, pace analysis |
| `test_notebook_quality.py` | Quality scoring engine and threshold enforcement |
| `test_metadata_validate.py` | `kernel-metadata.json` validation (JSON syntax, required fields, credential detection) |
| `test_build_utils.py` | Shared notebook build utilities |
| `test_repo_guardrails.py` | Repository-level invariants (no secrets, structure checks) |

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full workflow. In brief:

1. Create a feature branch from the default branch.
2. Keep changes focused and small.
3. Run `./manage.sh validate` and `python -m pytest tests/ -v` before pushing.
4. Never commit credentials -- keep `kaggle.json` outside the repository.
5. Open a pull request with context and validation steps.

---

## License

This project is licensed under the [MIT License](./LICENSE).

Copyright (c) 2026 gr8monk3ys.
