# AI/ML Research Papers Trends (3K+ Papers)

A comprehensive synthetic dataset of **3,200+ AI/ML research papers** spanning 2018--2025, designed for bibliometric analysis, trend detection, and citation prediction.

## Dataset Overview

This dataset simulates metadata from AI/ML research papers, capturing realistic trends such as:

- The rapid rise of **transformer architectures** (from ~8% in 2018 to ~52% in 2025)
- Increasing paper volume over time, reflecting the growth of the field
- **Power-law citation distributions** with venue and recency adjustments
- Category-venue affinities (e.g., CVPR papers are predominantly computer vision)
- Growing code availability in recent years

## Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `paper_id` | string | Unique identifier in arXiv-like format (e.g., `arxiv.202301.00042`) |
| `title` | string | Paper title generated from category-specific topic components |
| `abstract` | string | Synthetic abstract of 50--100 words summarizing the paper |
| `authors` | string | Semicolon-separated list of author names |
| `year` | int | Publication year (2018--2025) |
| `month` | int | Publication month (1--12) |
| `category` | string | Primary arXiv category: `cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`, `cs.NE`, `stat.ML`, `cs.RO`, `cs.IR` |
| `subcategory` | string | More specific topic within the category |
| `citation_count` | int | Number of citations (power-law distributed, adjusted by age, venue, survey status) |
| `venue` | string | Publication venue: `NeurIPS`, `ICML`, `ICLR`, `AAAI`, `CVPR`, `ACL`, `EMNLP`, `NAACL`, or `arXiv-only` |
| `num_authors` | int | Number of authors (normally distributed, mean ~4.2) |
| `has_code` | bool | Whether the paper released source code |
| `primary_method` | string | Main ML method: `transformer`, `cnn`, `rnn`, `gnn`, `diffusion`, `reinforcement_learning`, `bayesian`, `ensemble`, `other` |
| `dataset_used` | string | Primary dataset used in the paper |
| `is_survey` | bool | Whether the paper is a survey/review (~4% of papers) |

## Key Statistics

- **Total papers**: 3,200
- **Year range**: 2018--2025
- **Categories**: 8 primary arXiv categories
- **Venues**: 9 (8 top conferences + arXiv-only)
- **Methods**: 9 primary method types
- **Papers with code**: ~45% overall, increasing in recent years

## Use Cases

1. **Bibliometric analysis**: Study publication trends, author collaboration patterns, and venue distributions over time.
2. **Trend detection**: Analyze the rise and fall of research methods (e.g., the transformer takeover), identify emerging subcategories.
3. **Citation prediction**: Build models to predict citation counts from paper metadata (title, abstract, venue, method, etc.).
4. **NLP tasks**: Use titles and abstracts for text classification, topic modeling, or summarization tasks.
5. **Network analysis**: Construct co-authorship networks from the author data.
6. **Venue recommendation**: Predict suitable venues based on paper characteristics.

## How It Was Generated

The dataset was created using a Python script (`create_dataset.py`) that employs:

- **Weighted random sampling** for realistic distributions across categories, venues, methods, and years
- **Year-dependent method weights** reflecting real-world trends (transformer adoption, CNN decline)
- **Power-law citation distributions** with corrections for paper age, venue prestige, and survey status
- **Category-venue affinity** matrices (e.g., CVPR papers skew toward cs.CV)
- **Template-based abstract generation** with randomized components for diversity

All data is synthetic and does not represent real papers or authors.

## Sample Rows

| paper_id | title | year | category | venue | primary_method | citation_count |
|----------|-------|------|----------|-------|----------------|----------------|
| arxiv.202301.00042 | Rethinking Language Modeling with Transformers | 2023 | cs.CL | ACL | transformer | 45 |
| arxiv.201906.00128 | Object Detection via Contrastive Learning | 2019 | cs.CV | CVPR | cnn | 312 |
| arxiv.202207.00891 | Efficient Meta-Learning at Scale | 2022 | cs.LG | ICML | transformer | 67 |
| arxiv.202503.01234 | AI Safety using Large Language Models | 2025 | cs.AI | arXiv-only | transformer | 3 |

## License

This dataset is released under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license.

## Citation

If you use this dataset in your work, please cite:

```
@dataset{scaturchio2025airesearch,
  title={AI/ML Research Papers Trends (3K+ Papers)},
  author={Scaturchio, Lorenzo},
  year={2025},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/lorenzoscaturchio/ai-ml-research-papers-trends}
}
```
