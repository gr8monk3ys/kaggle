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
./manage.sh --help
```

## License

MIT
