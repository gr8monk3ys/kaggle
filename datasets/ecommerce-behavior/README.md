# E-Commerce Behavior: 5 Relational Tables (236K Rows)

> 10K customers, 120K transactions, 80K sessions, 25K reviews — all synthetic

**License:** GPL-3.0

**Kaggle:** [lorenzoscaturchio/ecommerce-behavior](https://www.kaggle.com/datasets/lorenzoscaturchio/ecommerce-behavior)

## Description

Five relational tables covering the full e-commerce lifecycle: 10,000 customer profiles, 1,000 products across 15 categories and 20 brands, 120,000 transactions, 80,000 browsing sessions, and 25,000 product reviews — 236,000 rows in total. All of it is synthetic, produced by the seeded generator (create_dataset.py) that ships alongside the CSVs, so it carries no personal data and anyone can regenerate or resize it.

The tables really do join. Every customer_id in transactions, sessions and reviews resolves to a row in customers, and every product_id in transactions and reviews resolves to a row in products — 100% referential integrity on all five foreign-key edges, checked in the companion notebook rather than asserted here. The price chain is consistent too: transactions.unit_price equals products.price x (1 - discount_applied/100) for all 120,000 transaction rows.

Built for churn prediction (is_churned on customers), lifetime-value regression, recommendation and market-basket work, RFM segmentation, conversion-rate optimization (converted/bounced on sessions), and multi-table feature engineering. The companion notebook shows what the join buys you: aggregating sessions and transactions onto the customer table lifts 5-fold churn ROC-AUC from 0.660 to 0.679 against a demographics-only baseline, and wins in all five folds.

Design notes: transactions carry completed / pending / refunded / cancelled statuses for realistic label imbalance; sessions record device, channel, duration, pages and cart additions for funnel analysis; customers span five value segments (VIP, Premium, Regular, Budget Shopper, Occasional Visitor) whose churn rates differ by design. Events span 2023-01-01 to 2024-12-30, with signup dates reaching back to 2020-01-01. Because it is generated rather than observed, it is meant for teaching, benchmarking and prototyping — not for drawing conclusions about real shoppers.

## Tags

`business`, `classification`, `clustering`, `regression`, `tabular`, `data analytics`

## Authors

- **Lorenzo Scaturchio**: Independent ML engineer building synthetic, education-first datasets for reproducible benchmarking and prototyping.

## Coverage

- Temporal: 2020-01-01 to 2024-12-31 (signup dates; transactions, sessions and reviews run 2023-01-01 to 2024-12-30)
- Geospatial: Global (synthetic)

## DOI and Citations

- DOI: Not assigned
- Scaturchio, Lorenzo (2026). E-Commerce Behavior: 5 Relational Tables (236K Rows). Kaggle Dataset. https://www.kaggle.com/datasets/lorenzoscaturchio/ecommerce-behavior

## Provenance

- Source: Synthetic data generation scripts in this repository
- Source: Public domain schemas and domain conventions for educational simulation
- Collection methodology: Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data.

## Schema at a glance

All five tables join on two keys. Every foreign-key value resolves: 100% of the
`customer_id` values in `transactions`, `sessions` and `reviews` exist in `customers`,
and 100% of the `product_id` values in `transactions` and `reviews` exist in `products`.
Coverage the other way is near-total but not complete — 99.4% of customers have at
least one transaction and 98.2% have at least one session, so left joins produce a
small, realistic set of behaviour-free customers to handle.

```
customers (10,000)  --customer_id-->  transactions (120,000)  <--product_id--  products (1,000)
       |                                                                            |
       +------------ customer_id ----->  sessions (80,000)                          |
       |                                                                            |
       +------------ customer_id ----->  reviews (25,000)  <----- product_id -------+
```

## customers.csv

**Rows:** 10,000  |  **Columns:** 10  |  **Size:** 497.5 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `customer_id` | string | 0.0% | 10,000 | `C00000`, `C00001`, `C00002` |
| `signup_date` | string | 0.0% | 1,714 | `2020-02-13`, `2021-10-01`, `2022-06-26` |
| `age` | integer | 0.0% | 58 | `28`, `22`, `30` |
| `gender` | string | 0.0% | 4 | `M`, `Prefer not to say`, `F` |
| `country` | string | 0.0% | 10 | `BR`, `FR`, `US` |
| `segment` | string | 0.0% | 5 | `Premium`, `Regular`, `VIP` |
| `is_churned` | integer | 0.0% | 2 | `0`, `1` |
| `lifetime_value` | float | 0.0% | 9,781 | `1595.27`, `1160.61`, `3093.32` |
| `email_opt_in` | integer | 0.0% | 2 | `0`, `1` |
| `has_app` | integer | 0.0% | 2 | `0`, `1` |

## products.csv

**Rows:** 1,000  |  **Columns:** 11  |  **Size:** 75.0 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `product_id` | string | 0.0% | 1,000 | `P0000`, `P0001`, `P0002` |
| `product_name` | string | 0.0% | 1,000 | `PetLife Health #0`, `GlowUp Beauty #1`, `EcoLiving Office Supplies #2` |
| `category` | string | 0.0% | 15 | `Health`, `Beauty`, `Office Supplies` |
| `brand` | string | 0.0% | 20 | `PetLife`, `GlowUp`, `EcoLiving` |
| `price` | float | 0.0% | 948 | `34.82`, `53.35`, `51.45` |
| `avg_rating` | float | 0.0% | 37 | `4.0`, `5.0`, `3.5` |
| `num_ratings` | integer | 0.0% | 437 | `137`, `494`, `433` |
| `stock_quantity` | integer | 0.0% | 625 | `126`, `51`, `928` |
| `discount_pct` | integer | 0.0% | 9 | `0`, `10`, `40` |
| `is_featured` | integer | 0.0% | 2 | `0`, `1` |
| `weight_kg` | float | 0.0% | 349 | `0.35`, `1.76`, `0.21` |

## transactions.csv

**Rows:** 120,000  |  **Columns:** 11  |  **Size:** 9,606.7 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `transaction_id` | string | 0.0% | 120,000 | `T077767`, `T021376`, `T117783` |
| `customer_id` | string | 0.0% | 9,941 | `C08119`, `C08883`, `C05855` |
| `product_id` | string | 0.0% | 1,000 | `P0932`, `P0889`, `P0065` |
| `transaction_date` | string | 0.0% | 119,867 | `2023-01-01 00:23:39`, `2023-01-01 00:26:51`, `2023-01-01 00:45:53` |
| `quantity` | integer | 0.0% | 19 | `1`, `2`, `4` |
| `unit_price` | float | 0.0% | 952 | `23.87`, `31.53`, `67.75` |
| `total_amount` | float | 0.0% | 5,631 | `23.87`, `31.53`, `67.75` |
| `discount_applied` | integer | 0.0% | 9 | `0`, `50`, `5` |
| `status` | string | 0.0% | 4 | `refunded`, `completed`, `cancelled` |
| `payment_method` | string | 0.0% | 6 | `paypal`, `credit_card`, `apple_pay` |
| `shipping_cost` | float | 0.0% | 602 | `8.15`, `6.13`, `0.0` |

## sessions.csv

**Rows:** 80,000  |  **Columns:** 10  |  **Size:** 4,896.2 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `session_id` | string | 0.0% | 80,000 | `S049064`, `S015939`, `S003166` |
| `customer_id` | string | 0.0% | 9,824 | `C03202`, `C09526`, `C07034` |
| `session_date` | string | 0.0% | 79,946 | `2023-01-01 00:24:50`, `2023-01-01 00:42:04`, `2023-01-01 00:55:56` |
| `device` | string | 0.0% | 3 | `desktop`, `mobile`, `tablet` |
| `channel` | string | 0.0% | 6 | `direct`, `organic`, `social` |
| `duration_seconds` | integer | 0.0% | 2,611 | `72`, `36`, `284` |
| `pages_viewed` | integer | 0.0% | 108 | `1`, `5`, `10` |
| `converted` | integer | 0.0% | 2 | `1`, `0` |
| `bounced` | integer | 0.0% | 2 | `0`, `1` |
| `cart_additions` | integer | 0.0% | 6 | `5`, `1`, `0` |

## reviews.csv

**Rows:** 25,000  |  **Columns:** 8  |  **Size:** 1,656.0 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `review_id` | string | 0.0% | 25,000 | `R22520`, `R11296`, `R15342` |
| `customer_id` | string | 0.0% | 9,174 | `C06622`, `C01152`, `C05432` |
| `product_id` | string | 0.0% | 1,000 | `P0077`, `P0726`, `P0167` |
| `review_date` | string | 0.0% | 730 | `2023-01-01`, `2023-01-02`, `2023-01-03` |
| `rating` | integer | 0.0% | 5 | `4`, `5`, `3` |
| `review_text` | string | 0.0% | 15 | `Great product! Highly recommend.`, `Love it! Exactly what I needed.`, `Average quality, meets expectations.` |
| `helpful_votes` | integer | 0.0% | 50 | `0`, `3`, `42` |
| `verified_purchase` | integer | 0.0% | 2 | `0`, `1` |
## Suggested Use Cases

- Churn prediction on `customers.is_churned`, using behavioural aggregates joined from `sessions` and `transactions` (the companion notebook measures the lift the join gives you)
- Customer lifetime value regression, or auditing `lifetime_value` against realised spend from `transactions`
- Market basket analysis and association rules over `transactions` joined to `products`
- RFM segmentation from transaction recency, frequency and monetary value
- Conversion funnel modelling on `sessions.converted` / `sessions.bounced` by device and channel
- Practising multi-table feature engineering and join hygiene before moving to a real relational dataset

## Companion notebook

`explore.ipynb` loads all five tables, verifies every foreign key, builds a joined
customer-360 feature table, and quantifies what the join is worth for churn modelling.

---
*Row counts, null rates and unique counts above are measured over the complete files.*
