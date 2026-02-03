#!/usr/bin/env python3
"""
Synthetic E-Commerce Customer Behavior Dataset Generator
=========================================================
Generates realistic e-commerce data with 100K+ transactions, 10K customers,
1K products, sessions, and reviews.  Includes seasonality, customer segments,
churn signals, and realistic correlations.

Usage:
    python create_dataset.py   # writes customers.csv, products.csv,
                               #   transactions.csv, sessions.csv, reviews.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CUSTOMERS = 10_000
N_PRODUCTS = 1_000
N_TRANSACTIONS = 120_000
N_SESSIONS = 80_000
N_REVIEWS = 25_000

DATE_START = pd.Timestamp("2023-01-01")
DATE_END = pd.Timestamp("2024-12-31")

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Customer segments
# ---------------------------------------------------------------------------
SEGMENTS = {
    "Budget Shopper":       {"frac": 0.30, "avg_spend": 25,  "freq": 0.5, "churn_prob": 0.25},
    "Regular":              {"frac": 0.35, "avg_spend": 55,  "freq": 1.0, "churn_prob": 0.15},
    "Premium":              {"frac": 0.20, "avg_spend": 120, "freq": 1.5, "churn_prob": 0.08},
    "VIP":                  {"frac": 0.10, "avg_spend": 250, "freq": 2.5, "churn_prob": 0.03},
    "Occasional Visitor":   {"frac": 0.05, "avg_spend": 40,  "freq": 0.2, "churn_prob": 0.40},
}

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Books", "Sports",
    "Beauty", "Toys", "Food & Grocery", "Office Supplies", "Automotive",
    "Health", "Pet Supplies", "Jewelry", "Music", "Software",
]

BRANDS = [
    "TechPro", "StyleMax", "HomeEssentials", "ReadMore", "FitGear",
    "GlowUp", "PlayTime", "FreshMarket", "WorkSmart", "AutoParts",
    "WellBeing", "PetLife", "ShineOn", "SoundWave", "DigiTools",
    "EcoLiving", "UrbanStyle", "ClassicCo", "NovaTech", "PureBrand",
]

GENDERS = ["M", "F", "Non-binary", "Prefer not to say"]
GENDER_WEIGHTS = [0.45, 0.45, 0.05, 0.05]

COUNTRIES = ["US", "UK", "CA", "DE", "FR", "AU", "IN", "BR", "JP", "MX"]
COUNTRY_WEIGHTS = [0.40, 0.12, 0.10, 0.08, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03]

DEVICES = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS = [0.35, 0.50, 0.15]

CHANNELS = ["organic", "paid_search", "social", "email", "direct", "referral"]
CHANNEL_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]

PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay", "bank_transfer"]
PAYMENT_WEIGHTS = [0.35, 0.20, 0.18, 0.12, 0.10, 0.05]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_dates(start, end, n, rng):
    """Generate n random dates between start and end (inclusive)."""
    ts_start = start.value // 10**9
    ts_end = end.value // 10**9
    timestamps = rng.integers(ts_start, ts_end, size=n)
    return pd.to_datetime(timestamps, unit="s")


def seasonal_weight(date):
    """Return a seasonal multiplier (higher in Q4 / holidays)."""
    month = date.month
    weights = {1: 0.7, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.85, 6: 0.9,
               7: 0.95, 8: 0.9, 9: 0.85, 10: 1.0, 11: 1.3, 12: 1.5}
    return weights.get(month, 1.0)


# ---------------------------------------------------------------------------
# Generate CUSTOMERS
# ---------------------------------------------------------------------------
def generate_customers():
    """Create a customers DataFrame."""
    # Assign segments
    seg_names = list(SEGMENTS.keys())
    seg_fracs = [SEGMENTS[s]["frac"] for s in seg_names]
    segments = rng.choice(seg_names, size=N_CUSTOMERS, p=seg_fracs)

    # Signup dates (earlier for VIP/Premium)
    signup_dates = []
    for seg in segments:
        if seg in ("VIP", "Premium"):
            d = random_dates(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-06-30"), 1, rng)[0]
        elif seg == "Occasional Visitor":
            d = random_dates(pd.Timestamp("2023-06-01"), DATE_END, 1, rng)[0]
        else:
            d = random_dates(pd.Timestamp("2021-01-01"), pd.Timestamp("2024-06-30"), 1, rng)[0]
        signup_dates.append(d)

    ages = rng.normal(35, 12, N_CUSTOMERS).clip(18, 75).astype(int)
    genders = rng.choice(GENDERS, size=N_CUSTOMERS, p=GENDER_WEIGHTS)
    countries = rng.choice(COUNTRIES, size=N_CUSTOMERS, p=COUNTRY_WEIGHTS)

    # Churn flag
    churn_probs = np.array([SEGMENTS[s]["churn_prob"] for s in segments])
    is_churned = rng.random(N_CUSTOMERS) < churn_probs

    # Lifetime value (correlated with segment)
    ltv = np.array([
        max(0, rng.normal(SEGMENTS[s]["avg_spend"] * 20, SEGMENTS[s]["avg_spend"] * 5))
        for s in segments
    ]).round(2)

    df = pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(5)}" for i in range(N_CUSTOMERS)],
        "signup_date": signup_dates,
        "age": ages,
        "gender": genders,
        "country": countries,
        "segment": segments,
        "is_churned": is_churned.astype(int),
        "lifetime_value": ltv,
        "email_opt_in": (rng.random(N_CUSTOMERS) > 0.3).astype(int),
        "has_app": (rng.random(N_CUSTOMERS) > 0.45).astype(int),
    })
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Generate PRODUCTS
# ---------------------------------------------------------------------------
def generate_products():
    """Create a products DataFrame."""
    categories = rng.choice(PRODUCT_CATEGORIES, size=N_PRODUCTS)
    brands = rng.choice(BRANDS, size=N_PRODUCTS)

    # Price depends on category
    cat_base_price = {
        "Electronics": 200, "Clothing": 45, "Home & Garden": 60, "Books": 15,
        "Sports": 50, "Beauty": 30, "Toys": 25, "Food & Grocery": 12,
        "Office Supplies": 20, "Automotive": 80, "Health": 35,
        "Pet Supplies": 25, "Jewelry": 90, "Music": 18, "Software": 50,
    }

    prices = []
    for cat in categories:
        base = cat_base_price[cat]
        price = max(1.99, rng.lognormal(np.log(base), 0.5))
        prices.append(round(price, 2))

    # Rating correlated with price (slightly)
    ratings = np.clip(rng.normal(3.8, 0.8, N_PRODUCTS) + (np.array(prices) > 50) * 0.2, 1.0, 5.0).round(1)
    n_ratings = rng.integers(0, 500, N_PRODUCTS)

    # Stock & discount
    stock = rng.integers(0, 1000, N_PRODUCTS)
    discount_pct = np.where(rng.random(N_PRODUCTS) > 0.7,
                             rng.choice([5, 10, 15, 20, 25, 30, 40, 50], N_PRODUCTS), 0)

    df = pd.DataFrame({
        "product_id": [f"P{str(i).zfill(4)}" for i in range(N_PRODUCTS)],
        "product_name": [f"{brands[i]} {categories[i]} #{i}" for i in range(N_PRODUCTS)],
        "category": categories,
        "brand": brands,
        "price": prices,
        "avg_rating": ratings,
        "num_ratings": n_ratings,
        "stock_quantity": stock,
        "discount_pct": discount_pct,
        "is_featured": (rng.random(N_PRODUCTS) > 0.9).astype(int),
        "weight_kg": np.clip(rng.lognormal(0, 1, N_PRODUCTS), 0.05, 50).round(2),
    })
    return df


# ---------------------------------------------------------------------------
# Generate TRANSACTIONS
# ---------------------------------------------------------------------------
def generate_transactions(customers_df, products_df):
    """Create a transactions DataFrame with seasonality and segment-based patterns."""
    cust_ids = customers_df["customer_id"].values
    prod_ids = products_df["product_id"].values
    prod_prices = products_df.set_index("product_id")["price"].to_dict()
    prod_discounts = products_df.set_index("product_id")["discount_pct"].to_dict()
    cust_segments = customers_df.set_index("customer_id")["segment"].to_dict()

    # Weighted customer sampling (higher-freq segments buy more)
    freq_weights = np.array([SEGMENTS[cust_segments[c]]["freq"] for c in cust_ids])
    freq_weights /= freq_weights.sum()

    txn_customers = rng.choice(cust_ids, size=N_TRANSACTIONS, p=freq_weights)
    txn_products = rng.choice(prod_ids, size=N_TRANSACTIONS)
    txn_dates = random_dates(DATE_START, DATE_END, N_TRANSACTIONS, rng)

    # Apply seasonal multiplier to quantities
    quantities = []
    for d in txn_dates:
        sw = seasonal_weight(d)
        q = max(1, int(rng.exponential(1.5) * sw))
        quantities.append(q)
    quantities = np.array(quantities)

    # Compute amounts
    base_prices = np.array([prod_prices[p] for p in txn_products])
    discounts = np.array([prod_discounts[p] for p in txn_products])
    unit_prices = base_prices * (1 - discounts / 100)
    amounts = (unit_prices * quantities).round(2)

    statuses = rng.choice(
        ["completed", "completed", "completed", "completed",
         "refunded", "cancelled", "pending"],
        size=N_TRANSACTIONS,
    )
    payments = rng.choice(PAYMENT_METHODS, size=N_TRANSACTIONS, p=PAYMENT_WEIGHTS)

    df = pd.DataFrame({
        "transaction_id": [f"T{str(i).zfill(6)}" for i in range(N_TRANSACTIONS)],
        "customer_id": txn_customers,
        "product_id": txn_products,
        "transaction_date": txn_dates,
        "quantity": quantities,
        "unit_price": unit_prices.round(2),
        "total_amount": amounts,
        "discount_applied": discounts,
        "status": statuses,
        "payment_method": payments,
        "shipping_cost": np.where(amounts > 50, 0.0, rng.uniform(3.99, 9.99, N_TRANSACTIONS)).round(2),
    })
    df = df.sort_values("transaction_date").reset_index(drop=True)
    df["transaction_date"] = df["transaction_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


# ---------------------------------------------------------------------------
# Generate SESSIONS
# ---------------------------------------------------------------------------
def generate_sessions(customers_df):
    """Create a sessions DataFrame with realistic browsing patterns."""
    cust_ids = customers_df["customer_id"].values
    cust_segments = customers_df.set_index("customer_id")["segment"].to_dict()

    freq_weights = np.array([SEGMENTS[cust_segments[c]]["freq"] for c in cust_ids])
    freq_weights /= freq_weights.sum()

    session_customers = rng.choice(cust_ids, size=N_SESSIONS, p=freq_weights)
    session_dates = random_dates(DATE_START, DATE_END, N_SESSIONS, rng)
    devices = rng.choice(DEVICES, size=N_SESSIONS, p=DEVICE_WEIGHTS)
    channels = rng.choice(CHANNELS, size=N_SESSIONS, p=CHANNEL_WEIGHTS)

    # Duration depends on device and segment
    durations = []
    pages_viewed = []
    for i in range(N_SESSIONS):
        seg = cust_segments[session_customers[i]]
        base_dur = {"Budget Shopper": 180, "Regular": 300, "Premium": 420,
                    "VIP": 600, "Occasional Visitor": 120}[seg]
        device_mult = {"desktop": 1.2, "mobile": 0.8, "tablet": 1.0}[devices[i]]
        dur = max(10, int(rng.exponential(base_dur * device_mult)))
        durations.append(dur)
        pages_viewed.append(max(1, int(dur / rng.uniform(30, 90))))

    # Conversion (did the session lead to a purchase?)
    conversion_probs = np.array([
        {"Budget Shopper": 0.05, "Regular": 0.08, "Premium": 0.12,
         "VIP": 0.18, "Occasional Visitor": 0.03}[cust_segments[c]]
        for c in session_customers
    ])
    converted = (rng.random(N_SESSIONS) < conversion_probs).astype(int)

    # Bounce = very short session
    bounced = (np.array(durations) < 30).astype(int)

    df = pd.DataFrame({
        "session_id": [f"S{str(i).zfill(6)}" for i in range(N_SESSIONS)],
        "customer_id": session_customers,
        "session_date": session_dates,
        "device": devices,
        "channel": channels,
        "duration_seconds": durations,
        "pages_viewed": pages_viewed,
        "converted": converted,
        "bounced": bounced,
        "cart_additions": np.where(converted, rng.integers(1, 6, N_SESSIONS),
                                    rng.choice([0, 0, 0, 1, 2], N_SESSIONS)),
    })
    df = df.sort_values("session_date").reset_index(drop=True)
    df["session_date"] = df["session_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


# ---------------------------------------------------------------------------
# Generate REVIEWS
# ---------------------------------------------------------------------------
def generate_reviews(customers_df, products_df):
    """Create a reviews DataFrame with realistic rating distributions."""
    cust_ids = customers_df["customer_id"].values
    prod_ids = products_df["product_id"].values
    prod_ratings = products_df.set_index("product_id")["avg_rating"].to_dict()

    review_customers = rng.choice(cust_ids, size=N_REVIEWS)
    review_products = rng.choice(prod_ids, size=N_REVIEWS)
    review_dates = random_dates(DATE_START, DATE_END, N_REVIEWS, rng)

    # Rating centered around product's avg rating
    ratings = []
    for pid in review_products:
        avg = prod_ratings[pid]
        r = int(np.clip(rng.normal(avg, 0.8), 1, 5))
        ratings.append(r)

    # Review text templates based on rating
    positive = [
        "Great product! Highly recommend.",
        "Excellent quality, fast shipping.",
        "Love it! Exactly what I needed.",
        "Perfect. Will buy again.",
        "Amazing value for the price.",
    ]
    neutral = [
        "It is okay, nothing special.",
        "Decent product for the price.",
        "Average quality, meets expectations.",
        "Fine for basic use.",
        "Not bad, not great.",
    ]
    negative = [
        "Disappointed with the quality.",
        "Would not recommend. Poor value.",
        "Not as described. Returning it.",
        "Broke after a week of use.",
        "Terrible experience. Avoid.",
    ]

    review_texts = []
    for r in ratings:
        if r >= 4:
            review_texts.append(rng.choice(positive))
        elif r == 3:
            review_texts.append(rng.choice(neutral))
        else:
            review_texts.append(rng.choice(negative))

    helpful_votes = rng.integers(0, 50, N_REVIEWS)
    verified = (rng.random(N_REVIEWS) > 0.2).astype(int)

    df = pd.DataFrame({
        "review_id": [f"R{str(i).zfill(5)}" for i in range(N_REVIEWS)],
        "customer_id": review_customers,
        "product_id": review_products,
        "review_date": review_dates,
        "rating": ratings,
        "review_text": review_texts,
        "helpful_votes": helpful_votes,
        "verified_purchase": verified,
    })
    df = df.sort_values("review_date").reset_index(drop=True)
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating customers...")
    customers = generate_customers()
    customers.to_csv(OUTPUT_DIR / "customers.csv", index=False)
    print(f"  -> {len(customers)} customers")

    print("Generating products...")
    products = generate_products()
    products.to_csv(OUTPUT_DIR / "products.csv", index=False)
    print(f"  -> {len(products)} products")

    print("Generating transactions...")
    transactions = generate_transactions(customers, products)
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False)
    print(f"  -> {len(transactions)} transactions")

    print("Generating sessions...")
    sessions = generate_sessions(customers)
    sessions.to_csv(OUTPUT_DIR / "sessions.csv", index=False)
    print(f"  -> {len(sessions)} sessions")

    print("Generating reviews...")
    reviews = generate_reviews(customers, products)
    reviews.to_csv(OUTPUT_DIR / "reviews.csv", index=False)
    print(f"  -> {len(reviews)} reviews")

    print("\nAll files saved to:", OUTPUT_DIR)
    print("Summary:")
    print(f"  customers.csv  : {len(customers):>8,} rows x {customers.shape[1]} cols")
    print(f"  products.csv   : {len(products):>8,} rows x {products.shape[1]} cols")
    print(f"  transactions.csv: {len(transactions):>8,} rows x {transactions.shape[1]} cols")
    print(f"  sessions.csv   : {len(sessions):>8,} rows x {sessions.shape[1]} cols")
    print(f"  reviews.csv    : {len(reviews):>8,} rows x {reviews.shape[1]} cols")


if __name__ == "__main__":
    main()
