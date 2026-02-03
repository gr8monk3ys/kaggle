# Kaggle Portfolio - Claude Code Guide

## Project Structure

This is a monorepo containing Kaggle competition entries, educational notebooks, datasets, and community engagement strategy documents.

```
kaggle/
├── manage.sh                          # CLI for Kaggle push/pull/status operations
├── grandmaster-tracker.md             # Progress across 4 Grandmaster categories
├── competition-scout-report.md        # Active competition analysis
├── discussion-engagement-strategy.md  # 12-week community plan
├── discussion-drafts.md               # Pre-written discussion posts
│
├── Competition Entries
│   ├── med-gemma-challenge/           # MedGemma 4B chest X-ray triage (LoRA)
│   ├── akkadian-translation/          # ByT5 seq2seq ancient language translation
│   └── vesuvius-surface/              # 3D U-Net volumetric segmentation
│
├── Educational Notebooks
│   ├── feature-engineering/           # 50 techniques across 8 categories
│   ├── attention-guide/               # Bahdanau → Transformer walkthrough
│   ├── llm-finetuning/               # LoRA/QLoRA practical guide
│   ├── image-segmentation/            # U-Net → SegFormer masterclass
│   ├── timeseries-transformers/       # Transformer time series forecasting
│   ├── ensemble-stacking/            # Competition-winning ensemble methods
│   ├── rag-from-scratch/             # RAG from first principles
│   ├── graph-neural-networks/        # GNN practical guide
│   ├── financial-analysis/           # Financial time-series prediction
│   ├── fraud-detection/              # Explainable fraud detection
│   ├── eda-tutorial/                 # EDA best practices
│   └── competition-template/         # Standardized ML pipeline template
│
└── datasets/                          # 5 custom Kaggle datasets
    ├── ml-interview-questions/
    ├── ecommerce-behavior/
    ├── github-metrics/
    ├── ai-research-papers/
    └── programming-benchmarks/
```

## Conventions

- **Notebooks**: Each subfolder contains a Jupyter notebook (`.ipynb`) as the primary artifact, designed for Kaggle kernel execution
- **Competition entries** include both EDA and submission notebooks
- **Datasets** are in CSV/Parquet format, structured for Kaggle Datasets publishing
- **Strategy docs** are markdown files at the repo root level

## Management Script

`manage.sh` is the primary CLI tool for interacting with Kaggle:

```bash
./manage.sh push-notebook <folder>    # Push notebook to Kaggle
./manage.sh push-dataset <folder>     # Push dataset to Kaggle
./manage.sh status                    # Check publication status
```

## Key Context

- **Goal**: Kaggle Grandmaster across all 4 categories (Competitions, Notebooks, Datasets, Discussion)
- **Current status**: 21 notebooks live, 4 datasets published, 1 competition entered
- **Priority competitions**: Med-Gemma (highest medal probability, 58 teams), Vesuvius, Akkadian
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
