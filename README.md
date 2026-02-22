# Kaggle ML Portfolio

A systematic collection of Kaggle notebooks, competition entries, datasets, and community engagement resources targeting Kaggle Grandmaster status across all four categories.

## Competition Entries

| Project | Competition | Approach |
|---------|-------------|----------|
| [med-gemma-challenge](./med-gemma-challenge) | Medical AI (58 teams) | MedGemma 4B chest X-ray triage with LoRA fine-tuning |
| [akkadian-translation](./akkadian-translation) | Ancient Language (1,321 teams) | ByT5 seq2seq translation |
| [vesuvius-surface](./vesuvius-surface) | Scroll Detection (759 teams) | 3D U-Net volumetric segmentation |

## Educational Notebooks

| Notebook | Topic | Key Techniques |
|----------|-------|----------------|
| [feature-engineering](./feature-engineering) | Feature Engineering Masterclass | 50 techniques across 8 categories |
| [attention-guide](./attention-guide) | Attention Mechanisms | Bahdanau to Transformer, PyTorch implementations |
| [llm-finetuning](./llm-finetuning) | LLM Fine-Tuning | LoRA/QLoRA, 4-bit quantization, vLLM deployment |
| [image-segmentation](./image-segmentation) | Segmentation Masterclass | U-Net to SegFormer, Dice/Focal/Tversky losses |
| [timeseries-transformers](./timeseries-transformers) | Time Series Forecasting | Transformer architectures for energy demand |
| [ensemble-stacking](./ensemble-stacking) | Ensemble Methods | Competition-winning stacking techniques |
| [rag-from-scratch](./rag-from-scratch) | RAG Systems | Building retrieval-augmented generation from scratch |
| [graph-neural-networks](./graph-neural-networks) | Graph Neural Networks | GNN practical guide for structured data |
| [financial-analysis](./financial-analysis) | Financial Analysis | Time-series prediction for financial data |
| [fraud-detection](./fraud-detection) | Fraud Detection | Explainable ML for credit card fraud |
| [eda-tutorial](./eda-tutorial) | EDA Best Practices | Visualization techniques and workflows |
| [competition-template](./competition-template) | Competition Template | Standardized ML pipeline for rapid entry |

## Datasets

Located in [`datasets/`](./datasets):
- ML Interview Q&A (500+ questions)
- E-commerce Behavior
- GitHub Metrics
- AI Research Papers
- Programming Benchmarks

## Strategy & Tracking

| Document | Purpose |
|----------|---------|
| [grandmaster-tracker.md](./grandmaster-tracker.md) | Progress tracking across all 4 Grandmaster categories |
| [competition-scout-report.md](./competition-scout-report.md) | Active competition analysis and medal probability assessment |
| [discussion-engagement-strategy.md](./discussion-engagement-strategy.md) | 12-week community engagement roadmap |
| [discussion-drafts.md](./discussion-drafts.md) | Pre-written discussion posts for community engagement |
| [manage.sh](./manage.sh) | CLI tool for Kaggle notebook/dataset management |

## Tech Stack

- **Deep Learning**: PyTorch, HuggingFace Transformers, segmentation_models_pytorch
- **LLMs**: MedGemma, ByT5, LoRA/QLoRA, BitsAndBytes quantization
- **Computer Vision**: U-Net, DeepLabV3+, SegFormer, TTA
- **NLP**: Attention mechanisms, seq2seq, embeddings (FastText, ELMo)
- **Tools**: Kaggle CLI, GGUF export, vLLM serving, Plotly

## Management

```bash
# Use the management script for Kaggle operations
chmod +x manage.sh
./manage.sh help
./manage.sh scorecard
./manage.sh weekly-plan
./manage.sh pace
./manage.sh sync
./manage.sh sync-template
./manage.sh doctor
./manage.sh quality
# Fallback when Kaggle CLI/network is unavailable:
./manage.sh sync --dry-run --kernels-csv /path/kernels.csv --datasets-csv /path/datasets.csv --competitions-csv /path/competitions.csv
```

`scorecard`, `weekly-plan`, `pace`, and `sync` generate reports under `medal_ops/reports/`.
`sync-template` scaffolds `kernels.csv`, `datasets.csv`, `competitions.csv`, and an export helper script.
`sync` can use live Kaggle CLI data or exported CSV files, and now fails fast if required vote columns are missing.
`doctor` runs preflight checks and writes `medal_ops/reports/latest-doctor.md`.
`quality` scores notebooks via rubric and writes `medal_ops/reports/latest-notebook-quality.md`.
`quality` also generates prioritized fixer checklists in `medal_ops/reports/latest-notebook-quality-fixes.md`.

## Automation

- Daily GitHub Actions workflow: `.github/workflows/medal-ops-health.yml`
- It runs `doctor --strict`, `quality --fail-under-threshold`, and `sync --dry-run` every day and opens an issue on failure.
- The workflow uses `--max-stale-days 30` to reduce noise from short tracker update gaps.
- The quality gate currently defaults to `--min-score 95`.
- Manual runs support `mode` (`auto`, `live`, `offline-fixture`), `max_stale_days`, and `min_quality_score` inputs.
- If `KAGGLE_USERNAME` and `KAGGLE_KEY` repository secrets are configured, it runs in live mode.
- Without secrets, it runs a strict offline-fixture mode so pipeline breakages are still caught.

## Authentication & Secrets

```bash
# Keep credentials outside the repository
mkdir -p ~/.kaggle
cp kaggle.json.example ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Never commit API credentials. `kaggle.json` is gitignored in this repo.
For GitHub Actions live checks, add `KAGGLE_USERNAME` and `KAGGLE_KEY` in repository Settings -> Secrets and variables -> Actions.

## License

MIT
