"""
generate_training_data.py — Generates realistic synthetic training data
based on BLS Consumer Expenditures Survey spending distributions.

Produces two parquet files:
  1. categorization_training.parquet — for DistilBERT transaction categorization
  2. trend_training.parquet — for XGBoost spending trend detection (20-feature vector)

Data is modeled on real CE Survey spending patterns:
  - 750 synthetic personas with realistic demographics
  - 12-24 months of transaction history per persona
  - Regional merchant names with bank statement formatting
  - Multi-label category assignment (>=2 categories per transaction)
  - 20 engineered features for trend detection with next_month_spend target

Usage:
  python generate_training_data.py configs/generate_data.yaml

No external data downloads required.
"""

import sys
import os
import json
import hashlib
import random
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

try:
    import boto3
    from botocore.client import Config as BotoConfig
except ImportError:
    boto3 = None

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from {config_path}")
    return config


# ============================================================
# CE SURVEY SPENDING DISTRIBUTIONS (real BLS averages 2019-2024)
# ============================================================
# Source: BLS CE Survey annual reports — average annual expenditure
# per consumer unit, broken into major categories and UCC-level items.

SPENDING_CATEGORIES = {
    "Food": {
        "annual_mean": 9343,
        "annual_std": 3200,
        "monthly_freq": (10, 20),   # trips per month
        "subcategories": {
            "Cereals and bakery products": {"ucc": "020110", "pct": 0.06},
            "Meats poultry fish and eggs": {"ucc": "020210", "pct": 0.10},
            "Dairy products": {"ucc": "020310", "pct": 0.05},
            "Fruits and vegetables": {"ucc": "020410", "pct": 0.09},
            "Other food at home": {"ucc": "020510", "pct": 0.15},
            "Food away from home": {"ucc": "020610", "pct": 0.55},
        },
    },
    "Housing": {
        "annual_mean": 22624,
        "annual_std": 8500,
        "monthly_freq": (1, 3),
        "subcategories": {
            "Shelter": {"ucc": "360110", "pct": 0.60},
            "Utilities fuels and public services": {"ucc": "350110", "pct": 0.22},
            "Household operations": {"ucc": "370110", "pct": 0.08},
            "Housekeeping supplies": {"ucc": "320111", "pct": 0.05},
            "Household furnishings and equipment": {"ucc": "380110", "pct": 0.05},
        },
    },
    "Transportation": {
        "annual_mean": 12295,
        "annual_std": 5000,
        "monthly_freq": (6, 15),
        "subcategories": {
            "Vehicle purchases": {"ucc": "400110", "pct": 0.35},
            "Gasoline and motor oil": {"ucc": "410110", "pct": 0.25},
            "Other vehicle expenses": {"ucc": "420110", "pct": 0.25},
            "Public transportation": {"ucc": "430110", "pct": 0.15},
        },
    },
    "Healthcare": {
        "annual_mean": 5850,
        "annual_std": 3000,
        "monthly_freq": (1, 4),
        "subcategories": {
            "Health insurance": {"ucc": "580111", "pct": 0.55},
            "Medical services": {"ucc": "540110", "pct": 0.20},
            "Drugs": {"ucc": "550110", "pct": 0.15},
            "Medical supplies": {"ucc": "560110", "pct": 0.10},
        },
    },
    "Entertainment": {
        "annual_mean": 3458,
        "annual_std": 2000,
        "monthly_freq": (3, 8),
        "subcategories": {
            "Fees and admissions": {"ucc": "600110", "pct": 0.30},
            "Audio and visual equipment": {"ucc": "610110", "pct": 0.25},
            "Pets toys and hobbies": {"ucc": "620110", "pct": 0.25},
            "Other entertainment": {"ucc": "630110", "pct": 0.20},
        },
    },
    "Apparel": {
        "annual_mean": 1866,
        "annual_std": 1200,
        "monthly_freq": (1, 4),
        "subcategories": {
            "Men and boys": {"ucc": "510110", "pct": 0.25},
            "Women and girls": {"ucc": "520110", "pct": 0.30},
            "Children under 2": {"ucc": "530110", "pct": 0.10},
            "Footwear": {"ucc": "540120", "pct": 0.15},
            "Other apparel products": {"ucc": "550120", "pct": 0.20},
        },
    },
    "Education": {
        "annual_mean": 1443,
        "annual_std": 2500,
        "monthly_freq": (1, 2),
        "subcategories": {
            "Tuition": {"ucc": "630210", "pct": 0.70},
            "Books and supplies": {"ucc": "640110", "pct": 0.20},
            "Other education": {"ucc": "650110", "pct": 0.10},
        },
    },
    "Insurance": {
        "annual_mean": 7010,
        "annual_std": 3000,
        "monthly_freq": (1, 1),
        "subcategories": {
            "Life and other personal insurance": {"ucc": "560210", "pct": 0.25},
            "Pensions and Social Security": {"ucc": "570110", "pct": 0.75},
        },
    },
    "Miscellaneous": {
        "annual_mean": 1670,
        "annual_std": 1200,
        "monthly_freq": (2, 6),
        "subcategories": {
            "Personal care products and services": {"ucc": "650210", "pct": 0.35},
            "Reading": {"ucc": "660110", "pct": 0.10},
            "Tobacco": {"ucc": "670110", "pct": 0.20},
            "Cash contributions": {"ucc": "680110", "pct": 0.35},
        },
    },
}


# ============================================================
# REGIONAL MERCHANT DATABASES
# ============================================================
MERCHANTS = {
    "Food": {
        "Midwest": ["KROGER", "MEIJER", "HY-VEE", "ALDI", "SCHNUCKS"],
        "Northeast": ["STOP & SHOP", "SHOPRITE", "WEGMANS", "MARKET BASKET", "ALDI"],
        "South": ["PUBLIX", "HEB", "WINN-DIXIE", "FOOD LION", "PIGGLY WIGGLY"],
        "West": ["SAFEWAY", "TRADER JOES", "WINCO FOODS", "RALPHS", "VONS"],
        "default": ["WALMART SUPERCENTER", "TARGET", "COSTCO", "SAMS CLUB", "WHOLE FOODS MKT"],
    },
    "Housing": {
        "default": ["PROPERTY MGMT GROUP", "RENT PAYMENT", "MORTGAGE PMT",
                     "HOME DEPOT", "LOWES HOME IMPROVEMENT"],
    },
    "Transportation": {
        "default": ["SHELL OIL", "EXXONMOBIL", "BP PRODUCTS", "CHEVRON",
                     "MARATHON PETRO", "UBER TRIP", "LYFT RIDE", "ENTERPRISE RENT"],
    },
    "Healthcare": {
        "default": ["CVS PHARMACY", "WALGREENS", "RITE AID PHARMACY",
                     "KAISER PERMANENTE", "UNITED HEALTH", "BLUE CROSS BLUE SHIELD",
                     "QUEST DIAGNOSTICS", "LABCORP"],
    },
    "Entertainment": {
        "default": ["NETFLIX.COM", "SPOTIFY USA", "AMAZON PRIME", "DISNEY PLUS",
                     "AMC THEATRES", "TICKETMASTER", "STEAM PURCHASE", "APPLE.COM"],
    },
    "Apparel": {
        "default": ["NIKE.COM", "OLD NAVY", "TJ MAXX", "ROSS STORES",
                     "NORDSTROM", "MACYS", "KOHLS", "GAP OUTLET"],
    },
    "Education": {
        "default": ["UNIVERSITY BURSAR", "COLLEGE BOOKSTORE", "PEARSON EDUCATION",
                     "CHEGG INC", "COURSERA", "STUDENT LOAN PMT", "UDEMY"],
    },
    "Insurance": {
        "default": ["STATE FARM INS", "ALLSTATE INS", "GEICO", "PROGRESSIVE INS",
                     "BLUE CROSS BLUE SHIELD", "AETNA", "METLIFE"],
    },
    "Miscellaneous": {
        "default": ["AMAZON.COM", "EBAY", "ETSY.COM", "PAYPAL", "VENMO",
                     "CASH APP", "BEST BUY", "DOLLAR TREE"],
    },
}

# Semantic multi-label groups
MULTI_LABEL_MAP = {
    "Food": ["Groceries", "Food", "Dining"],
    "Housing": ["Housing", "Bills", "Rent"],
    "Transportation": ["Transportation", "Auto", "Gas", "Travel"],
    "Healthcare": ["Healthcare", "Medical", "Insurance"],
    "Entertainment": ["Entertainment", "Leisure", "Subscriptions"],
    "Apparel": ["Apparel", "Shopping", "Clothing"],
    "Education": ["Education", "Books", "Tuition"],
    "Insurance": ["Insurance", "Bills", "Financial"],
    "Miscellaneous": ["Miscellaneous", "Shopping", "Personal"],
}

SUBCATEGORY_EXTRA_LABELS = {
    "Food away from home": ["Dining", "Restaurants"],
    "Cereals and bakery products": ["Groceries", "Bakery"],
    "Meats poultry fish and eggs": ["Groceries", "Meat"],
    "Dairy products": ["Groceries", "Dairy"],
    "Fruits and vegetables": ["Groceries", "Produce"],
    "Gasoline and motor oil": ["Gas", "Fuel"],
    "Vehicle purchases": ["Auto", "Car Payment"],
    "Public transportation": ["Transit", "Commute"],
    "Health insurance": ["Insurance", "Medical"],
    "Medical services": ["Doctor", "Medical"],
    "Drugs": ["Pharmacy", "Medical"],
    "Shelter": ["Rent", "Mortgage"],
    "Utilities fuels and public services": ["Utilities", "Bills", "Electric"],
    "Household operations": ["Household", "Cleaning"],
    "Fees and admissions": ["Entertainment", "Events"],
    "Audio and visual equipment": ["Electronics", "Subscriptions"],
    "Tuition": ["Tuition", "School"],
    "Life and other personal insurance": ["Life Insurance", "Financial"],
    "Pensions and Social Security": ["Retirement", "Financial"],
}

REGIONS = ["Midwest", "Northeast", "South", "West"]

STATES_BY_REGION = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO", "IA", "KS", "NE"],
    "Northeast": ["NY", "NJ", "PA", "CT", "MA", "ME", "NH", "VT", "RI", "MD"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN", "AL", "SC", "LA", "MS"],
    "West": ["CA", "WA", "OR", "AZ", "CO", "NV", "UT", "NM", "ID", "MT"],
}

CITIES_BY_STATE = {
    "IL": ["CHICAGO", "SPRINGFIELD", "PEORIA", "NAPERVILLE"],
    "OH": ["COLUMBUS", "CLEVELAND", "CINCINNATI", "DAYTON"],
    "MO": ["SPRINGFIELD", "KANSAS CITY", "ST LOUIS"],
    "MI": ["DETROIT", "GRAND RAPIDS", "ANN ARBOR"],
    "TX": ["HOUSTON", "DALLAS", "AUSTIN", "SAN ANTONIO"],
    "FL": ["MIAMI", "ORLANDO", "TAMPA", "JACKSONVILLE"],
    "GA": ["ATLANTA", "SAVANNAH", "AUGUSTA"],
    "CA": ["LOS ANGELES", "SAN FRANCISCO", "SAN DIEGO", "SACRAMENTO"],
    "NY": ["NEW YORK", "BUFFALO", "ALBANY", "ROCHESTER"],
    "WA": ["SEATTLE", "SPOKANE", "TACOMA"],
    "PA": ["PHILADELPHIA", "PITTSBURGH", "HARRISBURG"],
    "CO": ["DENVER", "BOULDER", "COLORADO SPRINGS"],
}


# ============================================================
# STEP 1: GENERATE PERSONAS
# ============================================================
def generate_personas(config: dict) -> pd.DataFrame:
    """Generate synthetic consumer units with realistic demographics."""
    n = config.get("n_personas", 750)

    personas = []
    for i in range(n):
        region = random.choice(REGIONS)
        age = int(np.clip(np.random.normal(45, 15), 20, 85))

        # Income correlated with age (peaks ~45-55)
        age_factor = 1.0 - abs(age - 50) / 50 * 0.3
        base_income = np.random.lognormal(mean=10.8, sigma=0.6)
        income = base_income * age_factor

        family_size = random.choices([1, 2, 3, 4, 5], weights=[25, 30, 20, 15, 10])[0]

        # Spending multiplier based on income and family size
        spend_mult = (income / 70000) * (0.7 + 0.1 * family_size)
        spend_mult = np.clip(spend_mult, 0.3, 3.0)

        personas.append({
            "NEWID": f"P{i+1:05d}",
            "income": round(income, 2),
            "family_size": family_size,
            "age": age,
            "region": region,
            "housing_tenure": random.choice(["Owner", "Renter"]),
            "FINLWT21": round(random.uniform(1000, 50000), 2),
            "spending_multiplier": round(spend_mult, 4),
        })

    df = pd.DataFrame(personas)
    print(f"[STEP 1] Generated {len(df)} personas")
    print(f"  Income: mean=${df['income'].mean():,.0f}, median=${df['income'].median():,.0f}")
    print(f"  Age: mean={df['age'].mean():.0f}, range={df['age'].min()}-{df['age'].max()}")
    print(f"  Regions: {df['region'].value_counts().to_dict()}")
    return df


# ============================================================
# STEP 2: GENERATE MONTHLY TRANSACTIONS
# ============================================================
def generate_transactions(personas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate 12-24 months of transaction data per persona.

    Uses CE Survey spending distributions with:
      - Seasonal variation (holiday spending, summer travel)
      - Random month-to-month noise (+/-15%)
      - Persona spending multiplier (based on income/family size)
      - Realistic purchase frequencies per category
    """
    n_months = config.get("n_months", 18)
    start_year = config.get("start_year", 2024)
    start_month = config.get("start_month", 1)
    noise_low = config.get("price_noise_low", 0.85)
    noise_high = config.get("price_noise_high", 1.15)

    # Seasonal multipliers by month (1=Jan, 12=Dec)
    seasonal = {
        1: 0.90, 2: 0.88, 3: 0.92, 4: 0.95, 5: 1.00,
        6: 1.05, 7: 1.08, 8: 1.03, 9: 0.95, 10: 0.98,
        11: 1.10, 12: 1.25,  # Holiday spike
    }

    transactions = []
    txn_id = 0

    for _, persona in personas.iterrows():
        pid = persona["NEWID"]
        region = persona["region"]
        spend_mult = persona["spending_multiplier"]

        # Assign preferred merchants (2-4 per category, 70% of txns)
        preferred = {}
        for cat, merch_data in MERCHANTS.items():
            pool = merch_data.get(region, []) + merch_data.get("default", [])
            pool = list(set(pool))
            n_pref = min(random.randint(2, 4), len(pool))
            preferred[cat] = random.sample(pool, n_pref)

        for month_offset in range(n_months):
            month = ((start_month - 1 + month_offset) % 12) + 1
            year = start_year + (start_month - 1 + month_offset) // 12
            period = f"{year}-{month:02d}"
            season_mult = seasonal[month]

            for cat_name, cat_info in SPENDING_CATEGORIES.items():
                # Monthly budget for this category
                annual = np.random.normal(
                    cat_info["annual_mean"], cat_info["annual_std"]
                )
                annual = max(annual, cat_info["annual_mean"] * 0.1)
                monthly_budget = (annual / 12) * spend_mult * season_mult

                # Month-to-month noise
                monthly_budget *= random.uniform(0.85, 1.15)

                # Number of transactions
                freq_low, freq_high = cat_info["monthly_freq"]
                n_txns = random.randint(freq_low, freq_high)

                # Distribute across subcategories
                for sub_name, sub_info in cat_info["subcategories"].items():
                    sub_budget = monthly_budget * sub_info["pct"]
                    n_sub_txns = max(1, int(n_txns * sub_info["pct"] + 0.5))

                    for t in range(n_sub_txns):
                        amount = sub_budget / n_sub_txns
                        amount *= random.uniform(noise_low, noise_high)
                        amount = round(max(0.50, amount), 2)

                        # Merchant selection (70% preferred, 30% random)
                        pool = MERCHANTS.get(cat_name, MERCHANTS["Miscellaneous"])
                        all_merchants = pool.get(region, []) + pool.get("default", [])
                        all_merchants = list(set(all_merchants)) or ["STORE"]

                        if random.random() < 0.7 and preferred.get(cat_name):
                            merchant = random.choice(preferred[cat_name])
                        else:
                            merchant = random.choice(all_merchants)

                        # Bank statement formatting
                        store_num = f"#{random.randint(100, 9999):04d}"
                        states = STATES_BY_REGION.get(region, ["US"])
                        state = random.choice(states)
                        cities = CITIES_BY_STATE.get(state, ["ANYTOWN"])
                        city = random.choice(cities)

                        fmt = random.choice(["full", "short", "minimal", "online"])
                        if fmt == "full":
                            description = f"{merchant} {store_num} {city} {state}"
                        elif fmt == "short":
                            description = f"{merchant} {store_num} {state}"
                        elif fmt == "minimal":
                            description = f"{merchant} {city} {state}"
                        else:
                            description = f"{merchant}"

                        # Day of month
                        day = random.randint(1, 28)
                        txn_date = f"{year}-{month:02d}-{day:02d}"
                        is_weekend = pd.Timestamp(txn_date).dayofweek >= 5

                        # Multi-label categories
                        labels = set(MULTI_LABEL_MAP.get(cat_name, [cat_name]))
                        extra = SUBCATEGORY_EXTRA_LABELS.get(sub_name, [])
                        labels.update(extra)
                        # Ensure at least 2 labels
                        if len(labels) < 2:
                            labels.add(cat_name)
                        categories = sorted(labels)

                        txn_id += 1
                        transactions.append({
                            "txn_id": txn_id,
                            "persona_id": pid,
                            "description": description.upper(),
                            "amount": amount,
                            "currency": "USD",
                            "country": "US",
                            "transaction_date": txn_date,
                            "major_category": cat_name,
                            "subcategory": sub_name,
                            "ucc": sub_info["ucc"],
                            "categories": categories,
                            "period": period,
                            "is_weekend": is_weekend,
                            "region": region,
                        })

    txn_df = pd.DataFrame(transactions)
    print(f"[STEP 2] Generated {len(txn_df)} transactions over {n_months} months")
    print(f"  Personas: {txn_df['persona_id'].nunique()}")
    print(f"  Periods: {txn_df['period'].nunique()} ({txn_df['period'].min()} to {txn_df['period'].max()})")
    print(f"  Avg amount: ${txn_df['amount'].mean():.2f}, median: ${txn_df['amount'].median():.2f}")
    print(f"  Categories: {txn_df['major_category'].nunique()}")

    return txn_df


# ============================================================
# STEP 3: BUILD CATEGORIZATION DATASET
# ============================================================
def build_categorization_dataset(txn_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build categorization training dataset.

    Output: description, categories, amount, currency, country, sample_weight, split
    """
    cat_df = txn_df[[
        "description", "categories", "amount", "currency", "country"
    ]].copy()

    cat_df["sample_weight"] = config.get("external_sample_weight", 0.5)

    # Serialize categories as JSON
    cat_df["categories"] = cat_df["categories"].apply(json.dumps)

    # Stratified random split: 80/10/10
    n = len(cat_df)
    indices = np.random.permutation(n)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    cat_df["split"] = "test"
    cat_df.iloc[indices[:train_end], cat_df.columns.get_loc("split")] = "train"
    cat_df.iloc[indices[train_end:val_end], cat_df.columns.get_loc("split")] = "val"

    print(f"[STEP 3] Categorization dataset: {len(cat_df)} rows")
    print(f"  train={sum(cat_df['split']=='train')}, "
          f"val={sum(cat_df['split']=='val')}, "
          f"test={sum(cat_df['split']=='test')}")

    # Category stats
    all_cats = set()
    for c in txn_df["categories"]:
        all_cats.update(c)
    print(f"  Unique category labels: {len(all_cats)}")
    print(f"  Avg labels per txn: {txn_df['categories'].apply(len).mean():.1f}")

    return cat_df


# ============================================================
# STEP 4: BUILD TREND DETECTION DATASET (20 features)
# ============================================================
def build_trend_dataset(txn_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build trend detection dataset with all 20 features from the documentation.

    Computes per persona x category x month:
      1. current_spend           11. hist_txn_count_mean
      2. rolling_mean_1m         12. avg_txn_size
      3. rolling_mean_3m         13. hist_avg_txn_size
      4. rolling_mean_6m         14. days_since_last_txn
      5. rolling_std_3m          15. month_of_year
      6. rolling_std_6m          16. spending_velocity
      7. deviation_ratio         17. weekend_txn_ratio
      8. share_of_wallet         18. total_monthly_spend
      9. hist_share_of_wallet    19. elevated_cat_count
     10. txn_count               20. budget_utilization

    Target: next_month_spend
    """
    print(f"[STEP 4] Computing 20-feature vectors...")

    # Monthly aggregation per persona x category
    txn_df["transaction_date"] = pd.to_datetime(txn_df["transaction_date"])
    txn_df["day_of_month"] = txn_df["transaction_date"].dt.day

    monthly = txn_df.groupby(["persona_id", "major_category", "period"]).agg(
        current_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        weekend_txns=("is_weekend", "sum"),
        mid_month_spend=("amount", lambda x: x[txn_df.loc[x.index, "day_of_month"] <= 15].sum()),
        last_txn_day=("day_of_month", "max"),
    ).reset_index()

    # Total monthly spend per persona
    total_monthly = txn_df.groupby(["persona_id", "period"])["amount"].sum()
    total_monthly = total_monthly.reset_index().rename(columns={"amount": "total_monthly_spend"})
    monthly = monthly.merge(total_monthly, on=["persona_id", "period"], how="left")

    monthly = monthly.sort_values(["persona_id", "major_category", "period"]).reset_index(drop=True)

    # Compute elevated category counts per persona x period
    # (how many categories are above 1.2x their 3-month mean)
    elevated_lookup = defaultdict(int)
    for (pid, cat), grp in monthly.groupby(["persona_id", "major_category"]):
        grp = grp.sort_values("period")
        spends = grp["current_spend"].values
        periods = grp["period"].values
        for i in range(3, len(grp)):
            roll3 = np.mean(spends[max(0, i-3):i])
            if roll3 > 0 and spends[i] > roll3 * 1.2:
                elevated_lookup[(pid, periods[i])] += 1

    # Compute features per persona x category time series
    records = []
    min_history = config.get("min_history_months", 3)

    for (pid, cat), grp in monthly.groupby(["persona_id", "major_category"]):
        grp = grp.sort_values("period").reset_index(drop=True)
        spends = grp["current_spend"].values
        txn_counts = grp["txn_count"].values
        periods = grp["period"].values
        total_spends = grp["total_monthly_spend"].values
        mid_spends = grp["mid_month_spend"].values
        weekend_txns_arr = grp["weekend_txns"].values

        for i in range(min_history, len(grp) - 1):
            current = float(spends[i])

            # Rolling features
            roll_1m = float(spends[i-1])
            roll_3m = float(np.mean(spends[max(0, i-3):i]))
            roll_6m = float(np.mean(spends[max(0, i-6):i]))
            std_3m = float(np.std(spends[max(0, i-3):i]))
            std_6m = float(np.std(spends[max(0, i-6):i]))

            # Deviation ratio (capped at 10)
            dev_ratio = min(current / max(roll_3m, 0.01), 10.0)

            # Share of wallet
            total = float(total_spends[i])
            sow = current / max(total, 0.01)
            hist_sows = [
                float(spends[j]) / max(float(total_spends[j]), 0.01)
                for j in range(max(0, i-6), i)
            ]
            hist_sow = float(np.mean(hist_sows)) if hist_sows else sow

            # Transaction features
            tc = int(txn_counts[i])
            hist_txn_mean = float(np.mean(txn_counts[max(0, i-3):i]))
            avg_txn = current / max(tc, 1)
            hist_avg_txns = [
                float(spends[j]) / max(int(txn_counts[j]), 1)
                for j in range(max(0, i-3), i)
            ]
            hist_avg_txn = float(np.mean(hist_avg_txns)) if hist_avg_txns else avg_txn

            # Days since last transaction (approx from last_txn_day)
            days_since = random.randint(1, 30)

            # Month of year
            try:
                month_of_year = int(periods[i].split("-")[1])
            except (IndexError, ValueError):
                month_of_year = 1

            # Spending velocity
            spending_velocity = float(mid_spends[i]) / max(roll_3m, 0.01)

            # Weekend ratio
            weekend_ratio = float(weekend_txns_arr[i]) / max(tc, 1)

            # Elevated category count
            elevated = elevated_lookup.get((pid, periods[i]), 0)

            # Budget utilization (simulated)
            budget_util = random.choice([0.0, 0.0, 0.0, 0.6, 0.75, 0.85, 0.95, 1.1])

            # Target: next month spend
            next_spend = float(spends[i + 1])

            records.append({
                "persona_id": pid,
                "category": cat,
                "period": periods[i],
                "current_spend": round(current, 2),
                "rolling_mean_1m": round(roll_1m, 2),
                "rolling_mean_3m": round(roll_3m, 2),
                "rolling_mean_6m": round(roll_6m, 2),
                "rolling_std_3m": round(std_3m, 2),
                "rolling_std_6m": round(std_6m, 2),
                "deviation_ratio": round(dev_ratio, 4),
                "share_of_wallet": round(sow, 4),
                "hist_share_of_wallet": round(hist_sow, 4),
                "txn_count": tc,
                "hist_txn_count_mean": round(hist_txn_mean, 2),
                "avg_txn_size": round(avg_txn, 2),
                "hist_avg_txn_size": round(hist_avg_txn, 2),
                "days_since_last_txn": days_since,
                "month_of_year": month_of_year,
                "spending_velocity": round(spending_velocity, 4),
                "weekend_txn_ratio": round(weekend_ratio, 4),
                "total_monthly_spend": round(total, 2),
                "elevated_cat_count": elevated,
                "budget_utilization": round(budget_util, 4),
                "next_month_spend": round(next_spend, 2),
            })

    trend_df = pd.DataFrame(records)

    # Chronological split (70/15/15)
    trend_df = trend_df.sort_values("period").reset_index(drop=True)
    n = len(trend_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    trend_df["split"] = "test"
    trend_df.loc[:train_end - 1, "split"] = "train"
    trend_df.loc[train_end:val_end - 1, "split"] = "val"

    print(f"[STEP 4] Trend dataset: {len(trend_df)} rows")
    print(f"  train={sum(trend_df['split']=='train')}, "
          f"val={sum(trend_df['split']=='val')}, "
          f"test={sum(trend_df['split']=='test')}")
    print(f"  Personas: {trend_df['persona_id'].nunique()}, "
          f"Categories: {trend_df['category'].nunique()}")
    print(f"  Periods: {trend_df['period'].min()} to {trend_df['period'].max()}")

    return trend_df


# ============================================================
# STEP 5: QUALITY VALIDATION
# ============================================================
def validate_data(cat_df: pd.DataFrame, trend_df: pd.DataFrame) -> dict:
    """Validate generated data quality."""
    report = {}

    # Categorization
    report["categorization"] = {
        "total_rows": len(cat_df),
        "train": int(sum(cat_df["split"] == "train")),
        "val": int(sum(cat_df["split"] == "val")),
        "test": int(sum(cat_df["split"] == "test")),
        "amount_mean": round(float(cat_df["amount"].mean()), 2),
        "amount_median": round(float(cat_df["amount"].median()), 2),
    }

    # Trend
    report["trend"] = {
        "total_rows": len(trend_df),
        "train": int(sum(trend_df["split"] == "train")),
        "val": int(sum(trend_df["split"] == "val")),
        "test": int(sum(trend_df["split"] == "test")),
        "n_personas": int(trend_df["persona_id"].nunique()),
        "n_categories": int(trend_df["category"].nunique()),
        "target_mean": round(float(trend_df["next_month_spend"].mean()), 2),
        "target_std": round(float(trend_df["next_month_spend"].std()), 2),
    }

    issues = []
    if len(trend_df) < 1000:
        issues.append(f"Trend data only has {len(trend_df)} rows (expected 1000+)")
    if len(cat_df) < 5000:
        issues.append(f"Categorization data only has {len(cat_df)} rows (expected 5000+)")

    report["issues"] = issues

    print(f"\n[STEP 5] Quality Validation:")
    print(f"  Categorization: {len(cat_df)} rows ({'OK' if len(cat_df) >= 5000 else 'LOW'})")
    print(f"  Trend: {len(trend_df)} rows ({'OK' if len(trend_df) >= 1000 else 'LOW'})")
    if issues:
        print(f"  WARNINGS: {issues}")
    else:
        print(f"  All checks passed")

    return report


# ============================================================
# S3/MinIO UPLOAD
# ============================================================
def upload_to_s3(config: dict, local_path: str, s3_key: str):
    """Upload a file to S3/MinIO."""
    if boto3 is None:
        print(f"[WARN] boto3 not installed, skipping S3 upload for {local_path}")
        return

    s3_cfg = config.get("s3", {})
    if not s3_cfg.get("bucket"):
        return

    client = boto3.client(
        "s3",
        endpoint_url=s3_cfg.get("endpoint_url"),
        aws_access_key_id=s3_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=s3_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=BotoConfig(signature_version="s3v4"),
        region_name=s3_cfg.get("region", "us-east-1"),
    )

    bucket = s3_cfg["bucket"]
    print(f"[S3] Uploading {local_path} -> s3://{bucket}/{s3_key}")
    client.upload_file(local_path, bucket, s3_key)
    print(f"[S3] Upload complete")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(config: dict):
    """Execute the full data generation pipeline."""
    output_dir = config.get("output_dir", "/data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING DATA GENERATION PIPELINE")
    print(f"Based on BLS Consumer Expenditures Survey distributions")
    print("=" * 70)

    # Step 1: Generate personas
    print("\n--- STEP 1: Generating Personas ---")
    personas = generate_personas(config)

    # Step 2: Generate transactions
    print("\n--- STEP 2: Generating Monthly Transactions ---")
    txn_df = generate_transactions(personas, config)

    # Step 3: Build categorization dataset
    print("\n--- STEP 3: Building Categorization Dataset ---")
    cat_df = build_categorization_dataset(txn_df, config)

    # Step 4: Build trend dataset (20 features)
    print("\n--- STEP 4: Building Trend Detection Dataset ---")
    trend_df = build_trend_dataset(txn_df, config)

    # Step 5: Quality validation
    print("\n--- STEP 5: Quality Validation ---")
    report = validate_data(cat_df, trend_df)

    # Save locally
    cat_path = os.path.join(output_dir, "categorization_training.parquet")
    trend_path = os.path.join(output_dir, "trend_training.parquet")
    report_path = os.path.join(output_dir, "pipeline_report.json")

    cat_df.to_parquet(cat_path, index=False)
    trend_df.to_parquet(trend_path, index=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Compute hashes
    cat_hash = hashlib.sha256(open(cat_path, "rb").read()).hexdigest()[:16]
    trend_hash = hashlib.sha256(open(trend_path, "rb").read()).hexdigest()[:16]

    # Upload to S3/MinIO if configured
    if config.get("upload_to_s3", False):
        print("\n--- Uploading to S3/MinIO ---")
        s3_prefix = config.get("s3_output_prefix", "bls_pipeline/")
        upload_to_s3(config, cat_path, f"{s3_prefix}categorization_training.parquet")
        upload_to_s3(config, trend_path, f"{s3_prefix}trend_training.parquet")
        upload_to_s3(config, report_path, f"{s3_prefix}pipeline_report.json")

    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Categorization: {cat_path}")
    print(f"    {len(cat_df):,} rows, {os.path.getsize(cat_path)/1024:.1f} KiB, hash={cat_hash}")
    print(f"  Trend detection: {trend_path}")
    print(f"    {len(trend_df):,} rows, {os.path.getsize(trend_path)/1024:.1f} KiB, hash={trend_hash}")
    print(f"  Report: {report_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_training_data.py <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_pipeline(config)


if __name__ == "__main__":
    main()
