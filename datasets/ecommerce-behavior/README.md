---
title: "Synthetic E-Commerce Customer Behavior Dataset"
description: "Realistic synthetic e-commerce dataset with 120K transactions, 10K customers, 1K products, 80K sessions, and 25K reviews. Features seasonality, customer segments, churn signals, and multi-table relationships."
license: CC0-1.0
tags:
  - e-commerce
  - customer behavior
  - recommendation systems
  - churn prediction
  - customer segmentation
  - market basket analysis
---

# Synthetic E-Commerce Customer Behavior Dataset

![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)
![Transactions: 120K](https://img.shields.io/badge/Transactions-120K-blue.svg)
![Customers: 10K](https://img.shields.io/badge/Customers-10K-green.svg)
![Products: 1K](https://img.shields.io/badge/Products-1K-orange.svg)
![Tables: 5](https://img.shields.io/badge/Tables-5-purple.svg)

## Overview

A comprehensive synthetic e-commerce dataset designed for practicing **recommendation systems**, **customer segmentation**, **churn prediction**, and **market basket analysis**. The data contains realistic patterns including seasonality, customer lifecycle stages, and correlated features across five interlinked tables.

This dataset is ideal for data science portfolios, Kaggle notebooks, and learning multi-table feature engineering.

---

## Quick Start

```python
import pandas as pd

# On Kaggle
base = '/kaggle/input/ecommerce-behavior'

customers = pd.read_csv(f'{base}/customers.csv')
products = pd.read_csv(f'{base}/products.csv')
transactions = pd.read_csv(f'{base}/transactions.csv')
sessions = pd.read_csv(f'{base}/sessions.csv')
reviews = pd.read_csv(f'{base}/reviews.csv')

# Quick stats
for name, df in [('Customers', customers), ('Products', products),
                  ('Transactions', transactions), ('Sessions', sessions), ('Reviews', reviews)]:
    print(f'{name:15s}: {len(df):>8,} rows x {df.shape[1]} cols')

# Example: Join transactions with customer data
tx_with_customer = transactions.merge(customers, on='customer_id')
print(f"\nAvg transaction by segment:")
print(tx_with_customer.groupby('segment')['total_amount'].mean().round(2))

# Example: RFM Analysis
completed = transactions[transactions['status'] == 'completed']
rfm = completed.groupby('customer_id').agg(
    recency=('transaction_date', lambda x: (pd.to_datetime(x).max() - pd.Timestamp('2020-01-01')).days),
    frequency=('transaction_id', 'count'),
    monetary=('total_amount', 'sum')
)
print(f"\nRFM shape: {rfm.shape}")
```

---

## Dataset Description

### Tables

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `customers.csv` | 10,000 | 10 | Customer profiles with demographics, segments, and churn labels |
| `products.csv` | 1,000 | 11 | Product catalog with categories, pricing, and ratings |
| `transactions.csv` | 120,000 | 11 | Purchase history with amounts, discounts, and payment methods |
| `sessions.csv` | 80,000 | 10 | Browsing sessions with device, channel, duration, and conversion |
| `reviews.csv` | 25,000 | 8 | Product reviews with ratings and text |

### Schema Details

#### customers.csv
| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique identifier (C00000-C09999) |
| signup_date | date | Account creation date |
| age | int | Customer age (18-75) |
| gender | string | M, F, Non-binary, Prefer not to say |
| country | string | Two-letter country code |
| segment | string | Customer segment (5 types) |
| is_churned | int | 1 if customer has churned, 0 otherwise |
| lifetime_value | float | Total lifetime spend |
| email_opt_in | int | Email marketing consent |
| has_app | int | Has installed the mobile app |

#### products.csv
| Column | Type | Description |
|--------|------|-------------|
| product_id | string | Unique identifier (P0000-P0999) |
| product_name | string | Product name |
| category | string | Product category (15 categories) |
| brand | string | Brand name (20 brands) |
| price | float | Current price |
| avg_rating | float | Average customer rating (1-5) |
| num_ratings | int | Number of ratings received |
| stock_quantity | int | Current stock level |
| discount_pct | int | Active discount percentage |
| is_featured | int | Featured on homepage |
| weight_kg | float | Product weight |

#### transactions.csv
| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string | Unique identifier |
| customer_id | string | FK to customers |
| product_id | string | FK to products |
| transaction_date | datetime | Purchase timestamp |
| quantity | int | Items purchased |
| unit_price | float | Price after discount |
| total_amount | float | Total transaction value |
| discount_applied | int | Discount percentage applied |
| status | string | completed, refunded, cancelled, pending |
| payment_method | string | Payment type |
| shipping_cost | float | Shipping fee (free over $50) |

#### sessions.csv
| Column | Type | Description |
|--------|------|-------------|
| session_id | string | Unique identifier |
| customer_id | string | FK to customers |
| session_date | datetime | Session start time |
| device | string | desktop, mobile, tablet |
| channel | string | Traffic source |
| duration_seconds | int | Session length |
| pages_viewed | int | Pages visited |
| converted | int | Led to a purchase |
| bounced | int | Left within 30 seconds |
| cart_additions | int | Items added to cart |

#### reviews.csv
| Column | Type | Description |
|--------|------|-------------|
| review_id | string | Unique identifier |
| customer_id | string | FK to customers |
| product_id | string | FK to products |
| review_date | date | Review submission date |
| rating | int | Star rating (1-5) |
| review_text | string | Review content |
| helpful_votes | int | Helpfulness votes |
| verified_purchase | int | Verified buyer |

### Entity Relationship Diagram

```
customers ──┬──< transactions >──── products
            │                         │
            ├──< sessions             │
            │                         │
            └──< reviews >────────────┘
```

### Built-in Patterns

- **Customer segments**: Budget Shopper, Regular, Premium, VIP, Occasional Visitor
- **Seasonality**: Transaction volume peaks in Q4 (holiday season)
- **Churn signals**: Segment-specific churn probabilities
- **Price correlations**: Product prices follow category-specific log-normal distributions
- **Conversion funnels**: Session conversion rates vary by segment and device
- **Free shipping threshold**: Orders over $50 get free shipping

---

## Use Cases

| # | Use Case | Type | Tables Needed |
|---|----------|------|---------------|
| 1 | **Customer Segmentation** | Unsupervised (K-Means, DBSCAN) | customers, transactions |
| 2 | **Churn Prediction** | Binary Classification | customers, transactions, sessions |
| 3 | **Recommendation Systems** | Collaborative / Content-Based | transactions, products, reviews |
| 4 | **Market Basket Analysis** | Association Rules (Apriori) | transactions |
| 5 | **CLV Prediction** | Regression | customers, transactions |
| 6 | **Conversion Rate Optimization** | A/B Testing Simulation | sessions |
| 7 | **Demand Forecasting** | Time Series | transactions |
| 8 | **Sentiment Analysis** | NLP | reviews |
| 9 | **RFM Analysis** | Business Analytics | transactions |
| 10 | **Funnel Analysis** | Product Analytics | sessions |

### Related Kaggle Competitions

This dataset lets you practice techniques from:
- [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis) -- market basket + recommendations
- [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) -- multi-table feature engineering
- [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation) -- customer behavior modeling
- [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) -- recommendation systems

---

## File Structure

```
ecommerce-behavior/
  customers.csv           # 10K customer profiles
  products.csv            # 1K product catalog
  transactions.csv        # 120K purchase records
  sessions.csv            # 80K browsing sessions
  reviews.csv             # 25K product reviews
  create_dataset.py       # Generation script
  explore.ipynb           # Exploration notebook with EDA & churn model
  dataset-metadata.json   # Kaggle dataset metadata
  kernel-metadata.json    # Kaggle notebook metadata
```

---

## Citation

```
@dataset{ecommerce_behavior_2025,
  title={Synthetic E-Commerce Customer Behavior Dataset},
  author={Lorenzo Scaturchio},
  year={2025},
  url={https://www.kaggle.com/datasets/lorenzoscaturchio/ecommerce-behavior}
}
```

## License

CC0 1.0 Universal -- Public Domain.

---

**If you found this dataset useful, please upvote! It helps others in the community discover it.**
