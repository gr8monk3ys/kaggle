# Synthetic E-Commerce Customer Behavior Dataset

> 10K customers, 120K transactions, 25K reviews for ML projects

**License:** GPL-3.0  

**Kaggle:** [lorenzoscaturchio/ecommerce-behavior](https://www.kaggle.com/datasets/lorenzoscaturchio/ecommerce-behavior)  

## Description

Five relational tables covering the full e-commerce lifecycle: 10K customer profiles, 1K products across 8 categories, 120K transactions, 80K browsing sessions, and 25K product reviews. Tables join cleanly on customer_id and product_id — ready for multi-table ML pipelines without extra preprocessing.

Built for: churn prediction (is_churned label on customers), customer lifetime value regression, product recommendation systems, market basket analysis and association rules, RFM segmentation, conversion rate optimization (converted/bounced on sessions), and sentiment analysis on review text.

Notable design choices: transactions include refunded and cancelled statuses for realistic imbalance; sessions capture device, channel, and cart additions for funnel analysis; customers include Premium/Regular/Budget segments for stratified modeling. All data is synthetic.

## Tags

`business`, `classification`, `clustering`, `regression`, `beginner`

## Authors

- **Lorenzo Scaturchio**: Independent ML engineer building synthetic, education-first datasets for reproducible benchmarking and prototyping.

## Coverage

- Temporal: 2023-01-01 to 2025-12-31
- Geospatial: Global (synthetic)

## DOI and Citations

- DOI: Not assigned
- Scaturchio, Lorenzo (2026). Synthetic E-Commerce Customer Behavior Dataset. Kaggle Dataset. https://www.kaggle.com/datasets/lorenzoscaturchio/ecommerce-behavior

## Provenance

- Source: Synthetic data generation scripts in this repository
- Source: Public domain schemas and domain conventions for educational simulation
- Collection methodology: Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data.

## customers.csv

**Rows:** 5,000  |  **Columns:** 10  |  **Size:** 497.5 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `customer_id` | string | 0.0% | 5,000 | `C00000`, `C00001`, `C00002` |
| `signup_date` | string | 0.0% | 1,529 | `2020-02-13`, `2021-10-01`, `2022-06-26` |
| `age` | integer | 0.0% | 55 | `28`, `22`, `30` |
| `gender` | string | 0.0% | 4 | `F`, `M`, `Prefer not to say` |
| `country` | string | 0.0% | 10 | `US`, `UK`, `CA` |
| `segment` | string | 0.0% | 5 | `Regular`, `Budget Shopper`, `Premium` |
| `is_churned` | integer | 0.0% | 2 | `0`, `1` |
| `lifetime_value` | float | 0.0% | 4,939 | `1595.27`, `1160.61`, `3093.32` |
| `email_opt_in` | integer | 0.0% | 2 | `1`, `0` |
| `has_app` | integer | 0.0% | 2 | `1`, `0` |

## products.csv

**Rows:** 1,000  |  **Columns:** 11  |  **Size:** 75.0 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `product_id` | string | 0.0% | 1,000 | `P0000`, `P0001`, `P0002` |
| `product_name` | string | 0.0% | 1,000 | `PetLife Health #0`, `GlowUp Beauty #1`, `EcoLiving Office Supplies #2` |
| `category` | string | 0.0% | 15 | `Health`, `Sports`, `Beauty` |
| `brand` | string | 0.0% | 20 | `PetLife`, `GlowUp`, `EcoLiving` |
| `price` | float | 0.0% | 948 | `34.82`, `53.35`, `51.45` |
| `avg_rating` | float | 0.0% | 37 | `4.0`, `3.4`, `5.0` |
| `num_ratings` | integer | 0.0% | 437 | `137`, `494`, `433` |
| `stock_quantity` | integer | 0.0% | 625 | `126`, `51`, `928` |
| `discount_pct` | integer | 0.0% | 9 | `0`, `5`, `50` |
| `is_featured` | integer | 0.0% | 2 | `0`, `1` |
| `weight_kg` | float | 0.0% | 349 | `0.35`, `1.76`, `0.21` |

## reviews.csv

**Rows:** 5,000  |  **Columns:** 8  |  **Size:** 1,656.0 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `review_id` | string | 0.0% | 5,000 | `R22520`, `R11296`, `R15342` |
| `customer_id` | string | 0.0% | 3,943 | `C06622`, `C01152`, `C05432` |
| `product_id` | string | 0.0% | 988 | `P0077`, `P0726`, `P0167` |
| `review_date` | string | 0.0% | 149 | `2023-01-05`, `2023-01-01`, `2023-01-09` |
| `rating` | integer | 0.0% | 5 | `3`, `4`, `2` |
| `review_text` | string | 0.0% | 15 | `Amazing value for the price.`, `Love it! Exactly what I needed.`, `Great product! Highly recommend.` |
| `helpful_votes` | integer | 0.0% | 50 | `44`, `0`, `3` |
| `verified_purchase` | integer | 0.0% | 2 | `1`, `0` |

## sessions.csv

**Rows:** 5,000  |  **Columns:** 10  |  **Size:** 4,896.2 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `session_id` | string | 0.0% | 5,000 | `S049064`, `S015939`, `S003166` |
| `customer_id` | string | 0.0% | 3,695 | `C03202`, `C09526`, `C07034` |
| `session_date` | string | 0.0% | 4,996 | `2023-01-01 00:24:50`, `2023-01-01 00:42:04`, `2023-01-01 00:55:56` |
| `device` | string | 0.0% | 3 | `mobile`, `desktop`, `tablet` |
| `channel` | string | 0.0% | 6 | `organic`, `paid_search`, `direct` |
| `duration_seconds` | integer | 0.0% | 1,185 | `284`, `72`, `36` |
| `pages_viewed` | integer | 0.0% | 66 | `1`, `3`, `5` |
| `converted` | integer | 0.0% | 2 | `0`, `1` |
| `bounced` | integer | 0.0% | 2 | `0`, `1` |
| `cart_additions` | integer | 0.0% | 6 | `0`, `1`, `2` |

## transactions.csv

**Rows:** 5,000  |  **Columns:** 11  |  **Size:** 9,606.7 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `transaction_id` | string | 0.0% | 5,000 | `T077767`, `T021376`, `T117783` |
| `customer_id` | string | 0.0% | 3,720 | `C08119`, `C08883`, `C05855` |
| `product_id` | string | 0.0% | 986 | `P0932`, `P0889`, `P0065` |
| `transaction_date` | string | 0.0% | 4,995 | `2023-01-01 00:23:39`, `2023-01-01 00:26:51`, `2023-01-01 00:45:53` |
| `quantity` | integer | 0.0% | 7 | `1`, `2`, `3` |
| `unit_price` | float | 0.0% | 938 | `23.87`, `31.53`, `67.75` |
| `total_amount` | float | 0.0% | 1,538 | `23.87`, `31.53`, `67.75` |
| `discount_applied` | integer | 0.0% | 9 | `0`, `5`, `50` |
| `status` | string | 0.0% | 4 | `completed`, `cancelled`, `refunded` |
| `payment_method` | string | 0.0% | 6 | `credit_card`, `debit_card`, `paypal` |
| `shipping_cost` | float | 0.0% | 593 | `0.0`, `8.15`, `6.13` |

## Suggested Use Cases

- Text classification (TF-IDF, BERT embeddings)
- Named entity recognition or topic modeling

---
*Generated by `dataset_optimizer.py` — dataset_optimizer.py*
