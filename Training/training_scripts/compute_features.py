"""
compute_features.py — Computes the 20-feature vector for spending trend detection
from raw Firefly III transaction data (PostgreSQL).

Can be used for:
  1. Building training data from historical transactions
  2. Batch inference (nightly feature computation for all users)
  3. On-demand inference (single user refresh)

Usage:
  python compute_features.py configs/compute_features.yaml
  python compute_features.py configs/compute_features.yaml --user-id <UUID>
  python compute_features.py configs/compute_features.yaml --mode inference

All 20 features match the documentation specification exactly.
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# The 20 features as defined in the documentation (Section 4.2.2)
FEATURE_COLUMNS = [
    "current_spend",        # 1.  Total spend in category this month
    "rolling_mean_1m",      # 2.  Mean spend, preceding 1 month
    "rolling_mean_3m",      # 3.  Mean spend, preceding 3 months
    "rolling_mean_6m",      # 4.  Mean spend, preceding 6 months
    "rolling_std_3m",       # 5.  Std deviation, preceding 3 months
    "rolling_std_6m",       # 6.  Std deviation, preceding 6 months
    "deviation_ratio",      # 7.  current_spend / rolling_mean_3m (capped at 10)
    "share_of_wallet",      # 8.  Category spend / total monthly spend
    "hist_share_of_wallet", # 9.  Historical avg share-of-wallet (6 months)
    "txn_count",            # 10. Transaction count this month
    "hist_txn_count_mean",  # 11. Mean transaction count (3 months)
    "avg_txn_size",         # 12. current_spend / txn_count
    "hist_avg_txn_size",    # 13. Historical mean transaction size
    "days_since_last_txn",  # 14. Days since last transaction in category
    "month_of_year",        # 15. Calendar month (1-12, seasonality)
    "spending_velocity",    # 16. Cumulative spend at day 15 / rolling_mean_3m
    "weekend_txn_ratio",    # 17. Weekend txns / total txns
    "total_monthly_spend",  # 18. Total spend across all categories
    "elevated_cat_count",   # 19. Categories above 1.2x their mean
    "budget_utilization",   # 20. Spend / budget (0 if no budget set)
]


# ============================================================
# DATABASE CONNECTION
# ============================================================
def get_db_connection(config: dict):
    """Create a PostgreSQL connection to Firefly III database."""
    import psycopg2

    db_config = config.get("database", {})
    conn = psycopg2.connect(
        host=db_config.get("host", "localhost"),
        port=db_config.get("port", 5432),
        dbname=db_config.get("dbname", "firefly"),
        user=db_config.get("user", "firefly"),
        password=db_config.get("password", ""),
    )
    return conn


def load_transactions_from_db(config: dict, user_id: str = None) -> pd.DataFrame:
    """Load transactions from Firefly III PostgreSQL database."""
    conn = get_db_connection(config)

    # Query transactions with category information
    query = """
    SELECT
        t.id AS transaction_id,
        tj.user_id,
        t.description,
        t.amount,
        t.date AS transaction_date,
        c.name AS category_name,
        tj.category_id,
        tj.budget_id
    FROM transactions t
    JOIN transaction_journals tj ON t.transaction_journal_id = tj.id
    LEFT JOIN categories c ON tj.category_id = c.id
    WHERE t.amount > 0
      AND t.date >= NOW() - INTERVAL '7 months'
    """

    if user_id:
        query += f" AND tj.user_id = '{user_id}'"

    query += " ORDER BY tj.user_id, c.name, t.date"

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"[INFO] Loaded {len(df)} transactions from database")
    return df


def load_transactions_from_file(config: dict) -> pd.DataFrame:
    """Load transactions from a CSV/Parquet file (for offline/testing)."""
    data_path = config["data_path"]
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported format: {data_path}")

    # Ensure required columns
    required = ["user_id", "category_name", "amount", "transaction_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    print(f"[INFO] Loaded {len(df)} transactions from {data_path}")
    return df


def load_budgets(config: dict, user_id: str = None) -> dict:
    """Load budget information from Firefly III. Returns {(user_id, category): budget_amount}."""
    budgets = {}
    try:
        conn = get_db_connection(config)
        query = """
        SELECT bl.budget_id, b.user_id, b.name AS budget_name,
               bl.amount AS budget_amount
        FROM budget_limits bl
        JOIN budgets b ON bl.budget_id = b.id
        WHERE bl.start_date <= NOW() AND bl.end_date >= NOW()
        """
        if user_id:
            query += f" AND b.user_id = '{user_id}'"

        df = pd.read_sql(query, conn)
        conn.close()

        for _, row in df.iterrows():
            budgets[(str(row["user_id"]), row["budget_name"])] = float(row["budget_amount"])
    except Exception as e:
        print(f"[WARN] Could not load budgets: {e}")

    return budgets


# ============================================================
# FEATURE COMPUTATION
# ============================================================
def compute_monthly_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to monthly level per user x category."""
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["year_month"] = df["transaction_date"].dt.to_period("M")
    df["day_of_week"] = df["transaction_date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Monthly aggregation per user x category
    monthly = df.groupby(["user_id", "category_name", "year_month"]).agg(
        current_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        last_txn_date=("transaction_date", "max"),
        weekend_txns=("day_of_week", lambda x: sum(d >= 5 for d in x)),
    ).reset_index()

    # Total monthly spend per user
    user_monthly_total = df.groupby(["user_id", "year_month"])["amount"].sum()
    user_monthly_total = user_monthly_total.reset_index().rename(
        columns={"amount": "total_monthly_spend"}
    )
    monthly = monthly.merge(user_monthly_total, on=["user_id", "year_month"], how="left")

    # Mid-month cumulative spend (spend up to day 15)
    mid_month = df[df["transaction_date"].dt.day <= 15].groupby(
        ["user_id", "category_name", "year_month"]
    )["amount"].sum().reset_index().rename(columns={"amount": "mid_month_spend"})
    monthly = monthly.merge(
        mid_month, on=["user_id", "category_name", "year_month"], how="left"
    )
    monthly["mid_month_spend"] = monthly["mid_month_spend"].fillna(0)

    monthly = monthly.sort_values(["user_id", "category_name", "year_month"])
    return monthly


def compute_features_for_group(
    group: pd.DataFrame,
    budgets: dict,
    elevated_counts: dict,
    reference_date: pd.Period = None,
) -> list:
    """Compute the 20-feature vector for a user x category time series.

    Args:
        group: DataFrame with monthly aggregates for one user x category,
               sorted chronologically.
        budgets: Dict of (user_id, category) -> budget_amount.
        elevated_counts: Dict of (user_id, period) -> count of elevated categories.
        reference_date: If set, only compute features for this period (inference mode).

    Returns:
        List of feature dicts, one per eligible month.
    """
    records = []
    group = group.sort_values("year_month").reset_index(drop=True)

    spends = group["current_spend"].values
    txn_counts = group["txn_count"].values
    periods = group["year_month"].values
    total_monthly = group["total_monthly_spend"].values
    mid_month_spends = group["mid_month_spend"].values
    weekend_txns = group["weekend_txns"].values
    last_txn_dates = group["last_txn_date"].values

    user_id = group["user_id"].iloc[0]
    category = group["category_name"].iloc[0]

    for i in range(len(group)):
        period = periods[i]

        # Skip if reference_date specified and this isn't it
        if reference_date is not None and period != reference_date:
            continue

        # Need at least 1 month of history for meaningful features
        if i < 1:
            continue

        current = float(spends[i])

        # Rolling means
        roll_1m = float(spends[i - 1])
        roll_3m = float(np.mean(spends[max(0, i - 3):i]))
        roll_6m = float(np.mean(spends[max(0, i - 6):i]))

        # Rolling standard deviations
        std_3m = float(np.std(spends[max(0, i - 3):i])) if i >= 2 else 0.0
        std_6m = float(np.std(spends[max(0, i - 6):i])) if i >= 2 else 0.0

        # Deviation ratio (capped at 10)
        dev_ratio = min(current / max(roll_3m, 1e-8), 10.0)

        # Share of wallet
        total = float(total_monthly[i])
        sow = current / max(total, 1e-8)

        # Historical share of wallet (6 month average)
        hist_sows = [
            float(spends[j]) / max(float(total_monthly[j]), 1e-8)
            for j in range(max(0, i - 6), i)
        ]
        hist_sow = float(np.mean(hist_sows)) if hist_sows else sow

        # Transaction count features
        txn_count = int(txn_counts[i])
        hist_txn_mean = float(np.mean(txn_counts[max(0, i - 3):i]))
        avg_txn_size = current / max(txn_count, 1)
        hist_avg_txns = [
            float(spends[j]) / max(int(txn_counts[j]), 1)
            for j in range(max(0, i - 3), i)
        ]
        hist_avg_txn = float(np.mean(hist_avg_txns)) if hist_avg_txns else avg_txn_size

        # Days since last transaction
        if i > 0 and pd.notna(last_txn_dates[i - 1]):
            try:
                last_date = pd.Timestamp(last_txn_dates[i])
                period_start = period.to_timestamp()
                days_since = max(0, (period_start - pd.Timestamp(last_txn_dates[i - 1])).days)
            except Exception:
                days_since = 30
        else:
            days_since = 30

        # Month of year
        month_of_year = period.month

        # Spending velocity (cumulative spend at day 15 / rolling_mean_3m)
        spending_velocity = float(mid_month_spends[i]) / max(roll_3m, 1e-8)

        # Weekend transaction ratio
        weekend_ratio = float(weekend_txns[i]) / max(txn_count, 1)

        # Elevated category count
        elevated = elevated_counts.get((str(user_id), str(period)), 0)

        # Budget utilization
        budget = budgets.get((str(user_id), category), 0)
        budget_util = current / budget if budget > 0 else 0.0

        record = {
            "user_id": user_id,
            "category": category,
            "period": str(period),
            "current_spend": round(current, 2),
            "rolling_mean_1m": round(roll_1m, 2),
            "rolling_mean_3m": round(roll_3m, 2),
            "rolling_mean_6m": round(roll_6m, 2),
            "rolling_std_3m": round(std_3m, 2),
            "rolling_std_6m": round(std_6m, 2),
            "deviation_ratio": round(dev_ratio, 4),
            "share_of_wallet": round(sow, 4),
            "hist_share_of_wallet": round(hist_sow, 4),
            "txn_count": txn_count,
            "hist_txn_count_mean": round(hist_txn_mean, 2),
            "avg_txn_size": round(avg_txn_size, 2),
            "hist_avg_txn_size": round(hist_avg_txn, 2),
            "days_since_last_txn": days_since,
            "month_of_year": month_of_year,
            "spending_velocity": round(spending_velocity, 4),
            "weekend_txn_ratio": round(weekend_ratio, 4),
            "total_monthly_spend": round(total, 2),
            "elevated_cat_count": elevated,
            "budget_utilization": round(budget_util, 4),
        }
        records.append(record)

    return records


def compute_elevated_categories(monthly: pd.DataFrame) -> dict:
    """Compute how many categories are above 1.2x their 3-month mean per user per period.

    Returns dict of (user_id, period) -> count.
    """
    elevated = defaultdict(int)

    for (uid, cat), group in monthly.groupby(["user_id", "category_name"]):
        group = group.sort_values("year_month").reset_index(drop=True)
        spends = group["current_spend"].values
        periods = group["year_month"].values

        for i in range(1, len(group)):
            roll_3m = np.mean(spends[max(0, i - 3):i])
            if spends[i] > roll_3m * 1.2:
                key = (str(uid), str(periods[i]))
                elevated[key] += 1

    return dict(elevated)


def compute_all_features(
    transactions: pd.DataFrame,
    config: dict,
    user_id: str = None,
    mode: str = "training",
) -> pd.DataFrame:
    """Compute features for all eligible user-category pairs.

    Args:
        transactions: Raw transaction DataFrame.
        config: Configuration dict.
        user_id: If set, only compute for this user.
        mode: "training" adds next_month_spend target; "inference" does not.

    Returns:
        DataFrame with 20 features + metadata columns.
    """
    print(f"[INFO] Computing features (mode={mode})")

    # Filter by user if specified
    if user_id:
        transactions = transactions[transactions["user_id"] == str(user_id)]

    # Monthly aggregation
    monthly = compute_monthly_aggregates(transactions)

    # Candidate selection filters (from documentation Section 7.2)
    min_months = config.get("min_history_months", 3)
    min_txn_per_month = config.get("min_txn_per_month", 2)

    # Filter: active categories with >= min_txn_per_month
    active = monthly.groupby(["user_id", "category_name"]).agg(
        n_months=("year_month", "count"),
        avg_txn=("txn_count", "mean"),
    ).reset_index()
    active = active[(active["n_months"] >= min_months) & (active["avg_txn"] >= min_txn_per_month)]
    eligible_pairs = set(zip(active["user_id"], active["category_name"]))
    monthly = monthly[
        monthly.apply(lambda r: (r["user_id"], r["category_name"]) in eligible_pairs, axis=1)
    ]

    print(f"[INFO] {len(eligible_pairs)} eligible user-category pairs "
          f"(>={min_months} months, >={min_txn_per_month} txns/month)")

    # Load budgets
    budgets = {}
    if config.get("load_budgets", False):
        budgets = load_budgets(config, user_id)

    # Compute elevated category counts
    elevated_counts = compute_elevated_categories(monthly)

    # Compute features per user x category
    all_records = []
    for (uid, cat), group in monthly.groupby(["user_id", "category_name"]):
        records = compute_features_for_group(group, budgets, elevated_counts)
        all_records.extend(records)

    features_df = pd.DataFrame(all_records)

    if features_df.empty:
        print("[WARN] No features computed (insufficient data)")
        return features_df

    # Add target column for training mode
    if mode == "training":
        # Target: next month's spend for the same user x category
        features_df = features_df.sort_values(["user_id", "category", "period"])
        features_df["next_month_spend"] = features_df.groupby(
            ["user_id", "category"]
        )["current_spend"].shift(-1)

        # Drop rows without target (last month per group)
        before = len(features_df)
        features_df = features_df.dropna(subset=["next_month_spend"])
        print(f"[INFO] Dropped {before - len(features_df)} rows without target")

    # Outlier flagging: spending > 10x category median
    outlier_threshold = config.get("outlier_threshold", 10)
    category_medians = features_df.groupby("category")["current_spend"].median()
    features_df["_is_outlier"] = features_df.apply(
        lambda r: r["current_spend"] > category_medians.get(r["category"], 0) * outlier_threshold,
        axis=1,
    )
    n_outliers = features_df["_is_outlier"].sum()
    if n_outliers > 0:
        print(f"[WARN] {n_outliers} outlier rows flagged (>{outlier_threshold}x category median)")

    # Validate all 20 features are present
    missing_features = [f for f in FEATURE_COLUMNS if f not in features_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in output: {missing_features}")

    print(f"[INFO] Computed {len(features_df)} feature vectors "
          f"({features_df['user_id'].nunique()} users, "
          f"{features_df['category'].nunique()} categories)")

    return features_df


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compute trend detection features")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--user-id", help="Compute for specific user only")
    parser.add_argument("--mode", choices=["training", "inference"], default="training",
                        help="training adds target column; inference does not")
    parser.add_argument("--output", help="Override output path")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load transactions
    source = config.get("data_source", "file")
    if source == "database":
        transactions = load_transactions_from_db(config, args.user_id)
    else:
        transactions = load_transactions_from_file(config)

    # Compute features
    features_df = compute_all_features(
        transactions, config, user_id=args.user_id, mode=args.mode
    )

    if features_df.empty:
        print("[WARN] No features to save")
        return

    # Save output
    output_path = args.output or config.get("output_path", "/data/trend_features.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".parquet"):
        features_df.to_parquet(output_path, index=False)
    else:
        features_df.to_csv(output_path, index=False)

    print(f"[DONE] Saved {len(features_df)} feature vectors to {output_path}")


if __name__ == "__main__":
    main()
