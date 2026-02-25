#!/usr/bin/env python3
"""
Synthetic Credit Card Fraud Detection Dataset
==============================================
Generates 200K credit card transactions with ~0.5% fraud rate.
Designed to mimic the statistical properties of real fraud datasets
(PCA features, class imbalance, amount distributions).

Usage:
    python create_dataset.py  # writes credit_card_transactions.csv
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
N_LEGIT = 199_000
N_FRAUD = 1_000
N_TOTAL = N_LEGIT + N_FRAUD
TIME_WINDOW = 172_800  # 48 hours in seconds

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# PCA feature simulation
# ---------------------------------------------------------------------------
# For legit: standard normal (near-zero means)
# For fraud: shift certain components to create realistic PCA separation
FRAUD_SHIFTS = {
    "V1":  -3.0,
    "V3":  -3.0,
    "V4":   2.5,
    "V7":  -2.0,
    "V10": -2.5,
    "V11":  2.0,
    "V12": -3.0,
    "V14": -3.5,
    "V16": -1.5,
    "V17": -2.0,
    "V18": -1.5,
}

# ---------------------------------------------------------------------------
# Merchant categories
# ---------------------------------------------------------------------------
MERCHANT_CATEGORIES = [
    "grocery", "electronics", "gas_station", "restaurant",
    "online", "travel", "entertainment", "healthcare",
]

# Relative fraud risk per category (multipliers; normalized when sampling)
CATEGORY_FRAUD_RISK = {
    "grocery":       0.3,
    "electronics":   2.5,
    "gas_station":   1.5,
    "restaurant":    0.5,
    "online":        3.0,
    "travel":        2.0,
    "entertainment": 0.8,
    "healthcare":    0.6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_pca_features(n, shifts=None):
    """
    Generate V1-V28 PCA-like features for n transactions.

    For legitimate transactions pass shifts=None (all zero shifts).
    For fraud, pass FRAUD_SHIFTS dict to offset selected components.
    """
    # Slight covariance structure so features are not perfectly independent
    # (mimics real PCA output where components are decorrelated but the
    #  generative process still has structure)
    cov = np.eye(28)
    for i in range(27):
        cov[i, i + 1] = cov[i + 1, i] = 0.05  # mild off-diagonal

    means = np.zeros(28)
    if shifts:
        for key, val in shifts.items():
            idx = int(key[1:]) - 1  # "V1" -> 0, "V14" -> 13, etc.
            means[idx] = val

    data = rng.multivariate_normal(means, cov, size=n)
    cols = {f"V{i + 1}": data[:, i] for i in range(28)}
    return cols


def make_amounts(n, is_fraud=False):
    """
    Transaction amount in USD.

    Legit: lognormal(mean=3.5, sigma=1.2), capped at 25 000.
    Fraud: mixture -- 70% small amounts (<100 USD) + 30% large amounts,
           to reflect both micro-testing behaviour and large-ticket fraud.
    """
    if not is_fraud:
        amounts = rng.lognormal(mean=3.5, sigma=1.2, size=n)
        amounts = np.clip(amounts, 0.50, 25_000.0)
    else:
        n_small = int(n * 0.70)
        n_large = n - n_small
        small = rng.lognormal(mean=2.0, sigma=0.9, size=n_small)   # peak ~$7, up to ~$100
        small = np.clip(small, 0.50, 100.0)
        large = rng.lognormal(mean=5.5, sigma=0.8, size=n_large)   # peak ~$245, up to ~$5k
        large = np.clip(large, 100.0, 5_000.0)
        amounts = np.concatenate([small, large])
        rng.shuffle(amounts)
    return amounts.round(2)


def make_times(n, fraud=False):
    """
    Seconds elapsed since first transaction (uniform over 48 h window).
    Fraud is weighted toward night hours (23:00-04:00).
    """
    if not fraud:
        return rng.integers(0, TIME_WINDOW, size=n)

    # Night window: 23:00-04:00 each day = 5 h * 2 days = 36 000 s of night
    # Build a probability-weighted time by oversampling night hours.
    # Simple approach: 60% of fraud in night slots, 40% in daytime.
    n_night = int(n * 0.60)
    n_day = n - n_night

    # Night seconds within a single day: 23*3600 to 24*3600, plus 0 to 4*3600
    night_slots_day1 = [
        rng.integers(23 * 3600, 24 * 3600, size=n_night // 2),
        rng.integers(0, 4 * 3600, size=n_night - n_night // 2),
    ]
    night_day1 = np.concatenate(night_slots_day1)
    # Second day shifts by 86400 but we cap at TIME_WINDOW
    night_day2 = night_day1 + rng.choice([0, 86_400], size=len(night_day1))
    night_times = np.minimum(night_day2, TIME_WINDOW - 1)

    day_times = rng.integers(4 * 3600, 23 * 3600, size=n_day)

    times = np.concatenate([night_times, day_times])
    rng.shuffle(times)
    return times.astype(int)


def make_merchant_categories(n, fraud=False):
    """
    Assign merchant categories.

    For fraud, weight toward higher-risk categories.
    For legit, weight inversely to fraud risk (i.e. uniform-ish).
    """
    cats = MERCHANT_CATEGORIES
    if fraud:
        weights = np.array([CATEGORY_FRAUD_RISK[c] for c in cats], dtype=float)
    else:
        # Legitimate distribution: roughly uniform with slight grocery/restaurant bias
        weights = np.array([2.0, 1.0, 1.5, 2.0, 1.5, 0.8, 1.2, 0.8])
    weights /= weights.sum()
    return rng.choice(cats, size=n, p=weights)


# ---------------------------------------------------------------------------
# Generate legitimate transactions
# ---------------------------------------------------------------------------
def generate_legit():
    print(f"  Generating {N_LEGIT:,} legitimate transactions...")
    pca = make_pca_features(N_LEGIT, shifts=None)
    amounts = make_amounts(N_LEGIT, is_fraud=False)
    times = make_times(N_LEGIT, fraud=False)
    merchants = make_merchant_categories(N_LEGIT, fraud=False)

    df = pd.DataFrame(pca)
    df["Amount"] = amounts
    df["Time"] = times
    df["merchant_category"] = merchants
    df["Class"] = 0
    return df


# ---------------------------------------------------------------------------
# Generate fraud transactions
# ---------------------------------------------------------------------------
def generate_fraud():
    print(f"  Generating {N_FRAUD:,} fraud transactions...")
    pca = make_pca_features(N_FRAUD, shifts=FRAUD_SHIFTS)
    amounts = make_amounts(N_FRAUD, is_fraud=True)
    times = make_times(N_FRAUD, fraud=True)
    merchants = make_merchant_categories(N_FRAUD, fraud=True)

    df = pd.DataFrame(pca)
    df["Amount"] = amounts
    df["Time"] = times
    df["merchant_category"] = merchants
    df["Class"] = 1
    return df


# ---------------------------------------------------------------------------
# Derive temporal features
# ---------------------------------------------------------------------------
def add_temporal_features(df):
    """Derive hour_of_day, day_of_week, is_weekend from Time column."""
    seconds_in_day = 86_400
    df["hour_of_day"] = (df["Time"] % seconds_in_day) // 3600
    df["day_of_week"] = (df["Time"] // seconds_in_day) % 7   # 0=Monday ... 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating synthetic credit card fraud dataset...")

    legit = generate_legit()
    fraud = generate_fraud()

    combined = pd.concat([legit, fraud], ignore_index=True)

    # Shuffle rows
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Add transaction IDs
    combined.insert(0, "transaction_id", [f"TXN{str(i).zfill(6)}" for i in range(N_TOTAL)])

    # Derive temporal features
    combined = add_temporal_features(combined)

    # Reorder columns: transaction_id, Time, V1-V28, Amount, merchant_category,
    #                  hour_of_day, day_of_week, is_weekend, Class
    v_cols = [f"V{i}" for i in range(1, 29)]
    col_order = (
        ["transaction_id", "Time"]
        + v_cols
        + ["Amount", "merchant_category", "hour_of_day", "day_of_week", "is_weekend", "Class"]
    )
    combined = combined[col_order]

    # Round V features to 6 decimal places (mimics real dataset format)
    for col in v_cols:
        combined[col] = combined[col].round(6)

    # Save
    out_path = OUTPUT_DIR / "credit_card_transactions.csv"
    combined.to_csv(out_path, index=False)

    # Summary stats
    n_fraud_actual = combined["Class"].sum()
    fraud_rate = n_fraud_actual / len(combined) * 100

    print(f"\nDataset saved to: {out_path}")
    print(f"Total rows     : {len(combined):,}")
    print(f"Total columns  : {combined.shape[1]}")
    print(f"Fraud count    : {n_fraud_actual:,}  ({fraud_rate:.2f}%)")
    print(f"Legit count    : {len(combined) - n_fraud_actual:,}")

    print("\nAmount statistics by class:")
    print(
        combined.groupby("Class")["Amount"]
        .describe()
        .round(2)
        .rename(index={0: "Legit (0)", 1: "Fraud (1)"})
    )

    print("\nMerchant category distribution:")
    cat_stats = combined.groupby("merchant_category")["Class"].agg(
        total="count", fraud_count="sum"
    )
    cat_stats["fraud_rate_%"] = (cat_stats["fraud_count"] / cat_stats["total"] * 100).round(2)
    print(cat_stats.sort_values("fraud_rate_%", ascending=False))

    print("\nFraud rate by hour (top 5 highest):")
    hour_fraud = combined.groupby("hour_of_day")["Class"].mean().sort_values(ascending=False)
    print(hour_fraud.head())


if __name__ == "__main__":
    main()
