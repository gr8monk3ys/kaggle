# Credit Card Fraud Detection (200K Transactions)

> 200K transactions, 0.5% fraud rate, 30 features

**License:** GPL-3.0  

**Kaggle:** [lorenzoscaturchio/credit-card-fraud-detection-synthetic](https://www.kaggle.com/datasets/lorenzoscaturchio/credit-card-fraud-detection-synthetic)  

## Description

200,000 synthetic credit card transactions with a realistic ~0.5% fraud rate (1,000 fraud cases), designed to mirror the statistical properties of real fraud datasets. Features include 28 PCA-transformed components (V1-V28), transaction Amount, Time, merchant_category, hour_of_day, and day_of_week. Fraud transactions exhibit shifted PCA distributions, higher rates in online and electronics merchants, and atypical amount patterns.

Built for: fraud detection (binary classification with severe class imbalance), threshold optimization (precision-recall tradeoff), anomaly detection, cost-sensitive learning, and SMOTE/oversampling experiments. The class imbalance mirrors real-world fraud detection challenges.

Notable features: V1-V28 PCA structure allows comparison with real-world fraud datasets; merchant_category enables feature engineering; hour_of_day captures temporal fraud patterns; Amount distribution differs significantly between fraud and legit. All data is synthetic and generated for educational and ML practice purposes.

## Tags

`business`, `classification`, `binary classification`, `regression`, `clustering`

## Authors

- **Lorenzo Scaturchio**: Independent ML engineer building synthetic, education-first datasets for reproducible benchmarking and prototyping.

## Coverage

- Temporal: 2024-01-01 to 2025-12-31
- Geospatial: Global (synthetic)

## DOI and Citations

- DOI: Not assigned
- Scaturchio, Lorenzo (2026). Credit Card Fraud Detection (200K Transactions). Kaggle Dataset. https://www.kaggle.com/datasets/lorenzoscaturchio/credit-card-fraud-detection-synthetic

## Provenance

- Source: Synthetic data generation scripts in this repository
- Source: Public domain schemas and domain conventions for educational simulation
- Collection methodology: Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data.

## credit_card_transactions.csv

**Rows:** 5,000  |  **Columns:** 36  |  **Size:** 59,363.2 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `transaction_id` | string | 0.0% | 5,000 | `TXN000000`, `TXN000001`, `TXN000002` |
| `Time` | integer | 0.0% | 4,942 | `21669`, `154552`, `108347` |
| `V1` | float | 0.0% | 4,997 | `-1.674946`, `0.457052`, `-0.869588` |
| `V2` | float | 0.0% | 4,998 | `0.4876`, `0.048544`, `1.381336` |
| `V3` | float | 0.0% | 4,997 | `1.175017`, `-0.013781`, `0.549428` |
| `V4` | float | 0.0% | 4,997 | `0.635452`, `0.622927`, `0.540193` |
| `V5` | float | 0.0% | 4,999 | `-0.521807`, `2.308919`, `-0.715576` |
| `V6` | float | 0.0% | 4,994 | `0.254763`, `-0.487454`, `-1.626004` |
| `V7` | float | 0.0% | 4,997 | `0.059483`, `-0.169976`, `-1.458558` |
| `V8` | float | 0.0% | 4,998 | `1.010286`, `-1.569339`, `1.64164` |
| `V9` | float | 0.0% | 4,991 | `-1.553294`, `0.179202`, `0.726559` |
| `V10` | float | 0.0% | 4,994 | `-1.505219`, `-0.692094`, `-0.764346` |
| `V11` | float | 0.0% | 4,998 | `1.432725`, `-1.237974`, `0.704293` |
| `V12` | float | 0.0% | 4,995 | `-0.75462`, `-2.084791`, `0.907533` |
| `V13` | float | 0.0% | 4,995 | `-0.249966`, `0.041788`, `-1.861304` |
| `V14` | float | 0.0% | 4,996 | `-0.355055`, `-0.261141`, `-0.724303` |
| `V15` | float | 0.0% | 4,999 | `0.257343`, `-0.682517`, `1.605375` |
| `V16` | float | 0.0% | 4,995 | `-1.748765`, `-0.187089`, `-0.928111` |
| `V17` | float | 0.0% | 4,995 | `-0.919773`, `-0.420935`, `-0.181425` |
| `V18` | float | 0.0% | 4,999 | `-1.196052`, `-1.383653`, `0.951308` |
| `V19` | float | 0.0% | 4,996 | `-0.141891`, `-0.727302`, `-0.797672` |
| `V20` | float | 0.0% | 4,999 | `1.214874`, `0.597476`, `-1.867054` |
| `V21` | float | 0.0% | 4,997 | `-0.606343`, `1.752873`, `1.744513` |
| `V22` | float | 0.0% | 4,998 | `-0.68923`, `-0.845305`, `-0.886995` |
| `V23` | float | 0.0% | 4,996 | `-0.473771`, `0.38852`, `0.03757` |
| `V24` | float | 0.0% | 5,000 | `0.203366`, `-0.753456`, `-0.604088` |
| `V25` | float | 0.0% | 4,994 | `1.241134`, `-0.596844`, `-1.291588` |
| `V26` | float | 0.0% | 4,997 | `-0.643846`, `-0.223539`, `-1.484245` |
| `V27` | float | 0.0% | 4,999 | `-0.217364`, `-0.769639`, `0.500329` |
| `V28` | float | 0.0% | 4,997 | `0.690976`, `-0.352037`, `1.085212` |
| `Amount` | float | 0.0% | 3,975 | `19.49`, `63.58`, `1.13` |
| `merchant_category` | string | 0.0% | 8 | `grocery`, `restaurant`, `online` |
| `hour_of_day` | integer | 0.0% | 24 | `6`, `22`, `17` |
| `day_of_week` | integer | 0.0% | 2 | `1`, `0` |
| `is_weekend` | integer | 0.0% | 1 | `0` |
| `Class` | integer | 0.0% | 2 | `0`, `1` |

## Suggested Use Cases

- Binary classification (fraud detection) with severe class imbalance
- Anomaly detection (Isolation Forest, Autoencoder)
- Threshold optimization (precision-recall tradeoff)
- Text classification (TF-IDF, BERT embeddings)
- Named entity recognition or topic modeling

---
*Generated by `dataset_optimizer.py` — dataset_optimizer.py*
