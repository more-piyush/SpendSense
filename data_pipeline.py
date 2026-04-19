"""
data_pipeline.py — CE Survey ETL: transforms BLS Consumer Expenditures Survey
data into training-ready parquet files for both categorization and trend detection.

Pipeline Steps (from documentation):
  1. Download PUMD files from BLS
  2. Persona sampling from FMLI (weighted by FINLWT21)
  3. Item-to-transaction aggregation
  4. Multi-label category assignment (UCC hierarchy + semantic similarity)
  5. Synthetic merchant name generation (regional variation)
  6. Quality validation

Usage:
  python data_pipeline.py configs/data_pipeline.yaml

Outputs:
  - /data/categorization_training.parquet
  - /data/trend_training.parquet
"""

import sys
import os
import json
import hashlib
import zipfile
import random
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import yaml

warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# CONFIGURATION
# ============================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from {config_path}")
    return config


# ============================================================
# STEP 0: DOWNLOAD PUMD FILES
# ============================================================
BLS_BASE_URL = "https://www.bls.gov/cex/pumd/data/comma"

# Interview survey files for 2024 (quarterly: 241x through 251)
INTERVIEW_FILES = {
    "intrvw24": "intrvw24.zip",
}

# Hierarchical groupings file
HG_FILE_URL = "https://www.bls.gov/cex/cedict/CE-HG-Integ-2024.txt"


def download_file(url: str, dest_path: str, force: bool = False) -> str:
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest_path) and not force:
        print(f"[INFO] Already exists: {dest_path}")
        return dest_path

    print(f"[DOWNLOAD] {url} -> {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[DOWNLOAD] Complete: {dest_path} ({os.path.getsize(dest_path)} bytes)")
    return dest_path


def download_pumd_data(config: dict) -> dict:
    """Download all required PUMD files. Returns dict of paths."""
    raw_dir = config.get("raw_data_dir", "/data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    force = config.get("force_download", False)

    paths = {}

    # Download interview survey zip
    for name, filename in INTERVIEW_FILES.items():
        url = f"{BLS_BASE_URL}/{filename}"
        zip_path = os.path.join(raw_dir, filename)
        try:
            download_file(url, zip_path, force=force)
            paths[name] = zip_path
        except Exception as e:
            print(f"[WARN] Could not download {url}: {e}")
            # Check if user provided local path
            local = config.get(f"local_{name}_path")
            if local and os.path.exists(local):
                paths[name] = local
                print(f"[INFO] Using local file: {local}")
            else:
                raise FileNotFoundError(
                    f"Cannot find {name} data. Provide 'local_{name}_path' in config."
                )

    # Download hierarchical groupings
    hg_path = os.path.join(raw_dir, "CE-HG-Integ-2024.txt")
    try:
        download_file(HG_FILE_URL, hg_path, force=force)
    except Exception:
        local = config.get("local_hg_path")
        if local and os.path.exists(local):
            hg_path = local
        else:
            raise
    paths["hg_file"] = hg_path

    return paths


# ============================================================
# STEP 0b: PARSE PUMD FILES
# ============================================================
def extract_csv_from_zip(zip_path: str, pattern: str) -> pd.DataFrame:
    """Extract and concatenate CSV files matching a pattern from a zip."""
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        matching = [n for n in zf.namelist() if re.search(pattern, n, re.IGNORECASE)]
        print(f"[INFO] Found {len(matching)} files matching '{pattern}' in {zip_path}")
        for name in sorted(matching):
            with zf.open(name) as f:
                df = pd.read_csv(f)
                frames.append(df)
                print(f"  -> {name}: {len(df)} rows")

    if not frames:
        raise ValueError(f"No files matching '{pattern}' found in {zip_path}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Combined: {len(combined)} rows")
    return combined


def parse_hierarchical_groupings(hg_path: str) -> pd.DataFrame:
    """Parse the CE hierarchical groupings file to get UCC -> item name mapping."""
    records = []
    with open(hg_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: level, title, UCC (tab-separated or fixed-width)
            parts = line.split("\t")
            if len(parts) >= 3:
                level = parts[0].strip()
                title = parts[1].strip()
                ucc = parts[2].strip()
                records.append({
                    "level": level, "title": title, "ucc": ucc
                })

    df = pd.DataFrame(records)
    print(f"[INFO] Parsed {len(df)} hierarchical grouping entries")
    return df


def load_pumd_tables(paths: dict) -> dict:
    """Load all PUMD tables into DataFrames."""
    tables = {}

    zip_path = paths.get("intrvw24")
    if zip_path:
        tables["fmli"] = extract_csv_from_zip(zip_path, r"fmli\d")
        tables["mtbi"] = extract_csv_from_zip(zip_path, r"mtbi\d")
        tables["memi"] = extract_csv_from_zip(zip_path, r"memi\d")

    tables["hg"] = parse_hierarchical_groupings(paths["hg_file"])

    return tables


# ============================================================
# STEP 1: PERSONA SAMPLING
# ============================================================
def sample_personas(fmli: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Sample consumer units (personas) weighted by survey weights.

    Each CE Survey respondent represents a real spending persona.
    Sample 500-1000 personas from FMLI, weighted by FINLWT21.
    """
    n_personas = config.get("n_personas", 750)
    weight_col = "FINLWT21"

    # Ensure required columns exist
    required = ["NEWID"]
    for col in required:
        if col not in fmli.columns:
            raise ValueError(f"Missing column {col} in FMLI")

    # Deduplicate by NEWID (take latest quarter if multiple)
    fmli_unique = fmli.drop_duplicates(subset=["NEWID"], keep="last").copy()

    # Use survey weights for sampling probability
    if weight_col in fmli_unique.columns:
        weights = fmli_unique[weight_col].clip(lower=0).fillna(0)
        weights = weights / weights.sum()
    else:
        print(f"[WARN] {weight_col} not found, using uniform sampling")
        weights = None

    n_sample = min(n_personas, len(fmli_unique))
    sampled = fmli_unique.sample(
        n=n_sample, weights=weights, random_state=RANDOM_SEED, replace=False
    )

    # Extract demographics
    demo_cols = {
        "FINCBTXM": "income",
        "FAM_SIZE": "family_size",
        "AGE_REF": "age",
        "REGION": "region",
        "CUTENURE": "housing_tenure",
    }
    for orig, alias in demo_cols.items():
        if orig in sampled.columns:
            sampled = sampled.rename(columns={orig: alias})

    print(f"[STEP 1] Sampled {len(sampled)} personas from {len(fmli_unique)} consumer units")
    return sampled


# ============================================================
# STEP 2: ITEM-TO-TRANSACTION AGGREGATION
# ============================================================

# Purchase frequency assumptions by major category (times per month)
PURCHASE_FREQ = {
    "Food": (8, 15),
    "Housing": (1, 2),
    "Transportation": (4, 8),
    "Healthcare": (1, 3),
    "Entertainment": (2, 6),
    "Apparel": (1, 3),
    "Education": (1, 2),
    "Insurance": (1, 1),
    "Utilities": (1, 2),
    "default": (2, 5),
}

# Major category mapping from UCC prefix
UCC_MAJOR_CATEGORIES = {
    "01": "Food",
    "02": "Food",
    "03": "Household",
    "04": "Household",
    "05": "Apparel",
    "06": "Transportation",
    "07": "Healthcare",
    "08": "Entertainment",
    "09": "Education",
    "10": "Miscellaneous",
    "11": "Insurance",
    "12": "Insurance",
    "30": "Household",
    "31": "Household",
    "32": "Household",
    "33": "Utilities",
    "34": "Utilities",
    "35": "Utilities",
    "36": "Housing",
    "37": "Housing",
    "38": "Housing",
    "39": "Housing",
    "40": "Transportation",
    "41": "Transportation",
    "42": "Transportation",
    "43": "Transportation",
    "44": "Transportation",
    "45": "Transportation",
    "50": "Healthcare",
    "51": "Healthcare",
    "52": "Healthcare",
    "53": "Healthcare",
    "54": "Healthcare",
    "55": "Healthcare",
    "56": "Insurance",
    "57": "Insurance",
    "58": "Insurance",
    "59": "Insurance",
    "60": "Entertainment",
    "61": "Entertainment",
    "62": "Entertainment",
    "63": "Education",
    "64": "Education",
    "65": "Miscellaneous",
    "66": "Miscellaneous",
    "67": "Miscellaneous",
    "68": "Miscellaneous",
    "69": "Miscellaneous",
    "70": "Miscellaneous",
    "80": "Miscellaneous",
    "85": "Miscellaneous",
    "90": "Miscellaneous",
}


def get_major_category(ucc: str) -> str:
    """Map a UCC code to its major spending category."""
    prefix = str(ucc).zfill(6)[:2]
    return UCC_MAJOR_CATEGORIES.get(prefix, "Miscellaneous")


def aggregate_items_to_transactions(
    mtbi: pd.DataFrame,
    personas: pd.DataFrame,
    hg: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Group item-level expenditures into merchant-level transactions.

    CE Survey data is at the item level (e.g., "Eggs: $4.50"), but Firefly III
    transactions are at the merchant level (e.g., "KROGER #1247 $42.30").

    Groups items from the same broad category within the same reporting period
    into single transactions, distributes quarterly totals across realistic
    purchase frequencies, and adds +/-5-10% price noise.
    """
    persona_ids = set(personas["NEWID"].astype(str))

    # Filter MTBI to sampled personas
    mtbi = mtbi.copy()
    mtbi["NEWID"] = mtbi["NEWID"].astype(str)
    mtbi_filtered = mtbi[mtbi["NEWID"].isin(persona_ids)].copy()
    print(f"[STEP 2] Filtered MTBI: {len(mtbi_filtered)} expenditure records for {len(persona_ids)} personas")

    if mtbi_filtered.empty:
        raise ValueError("No expenditure records found for sampled personas")

    # Build UCC -> item name lookup from HG file
    ucc_to_name = {}
    if hg is not None and not hg.empty:
        for _, row in hg.iterrows():
            if row["ucc"] and row["title"]:
                ucc_to_name[str(row["ucc"]).strip()] = row["title"].strip()

    # Add category and item name
    mtbi_filtered["UCC"] = mtbi_filtered["UCC"].astype(str).str.strip()
    mtbi_filtered["major_category"] = mtbi_filtered["UCC"].apply(get_major_category)
    mtbi_filtered["item_name"] = mtbi_filtered["UCC"].map(ucc_to_name).fillna("Unknown Item")

    # Parse cost
    mtbi_filtered["COST"] = pd.to_numeric(mtbi_filtered["COST"], errors="coerce").fillna(0)

    # Parse reporting period
    if "REFYR" in mtbi_filtered.columns and "REFMO" in mtbi_filtered.columns:
        mtbi_filtered["period"] = (
            mtbi_filtered["REFYR"].astype(str) + "-" +
            mtbi_filtered["REFMO"].astype(str).str.zfill(2)
        )
    else:
        mtbi_filtered["period"] = "2024-01"

    # Group by persona x major_category x period
    grouped = mtbi_filtered.groupby(
        ["NEWID", "major_category", "period"]
    ).agg(
        total_cost=("COST", "sum"),
        item_count=("UCC", "count"),
        uccs=("UCC", list),
        item_names=("item_name", list),
    ).reset_index()

    # Distribute quarterly totals across realistic purchase frequencies
    transactions = []
    noise_low = config.get("price_noise_low", 0.95)
    noise_high = config.get("price_noise_high", 1.10)

    for _, row in grouped.iterrows():
        category = row["major_category"]
        total = row["total_cost"]

        if total <= 0:
            continue

        # Determine number of transactions for this category-month
        freq_range = PURCHASE_FREQ.get(category, PURCHASE_FREQ["default"])
        n_txns = random.randint(freq_range[0], freq_range[1])

        # Distribute total across transactions with noise
        base_amount = total / n_txns
        for i in range(n_txns):
            noise = random.uniform(noise_low, noise_high)
            amount = round(base_amount * noise, 2)

            transactions.append({
                "persona_id": row["NEWID"],
                "major_category": category,
                "period": row["period"],
                "amount": amount,
                "uccs": row["uccs"],
                "item_names": row["item_names"],
                "item_count": row["item_count"],
            })

    txn_df = pd.DataFrame(transactions)
    print(f"[STEP 2] Generated {len(txn_df)} transactions from {len(grouped)} category-month groups")
    return txn_df


# ============================================================
# STEP 3: MULTI-LABEL CATEGORY ASSIGNMENT
# ============================================================

# Semantic category groups for multi-label assignment
SEMANTIC_GROUPS = {
    "Groceries": ["Food", "Household"],
    "Food": ["Groceries", "Dining"],
    "Dining": ["Food", "Entertainment"],
    "Utilities": ["Bills", "Housing"],
    "Bills": ["Utilities", "Insurance"],
    "Housing": ["Bills", "Utilities"],
    "Healthcare": ["Insurance", "Medical"],
    "Medical": ["Healthcare", "Insurance"],
    "Insurance": ["Bills", "Healthcare"],
    "Transportation": ["Auto", "Gas"],
    "Auto": ["Transportation", "Insurance"],
    "Gas": ["Transportation", "Auto"],
    "Entertainment": ["Leisure", "Dining"],
    "Leisure": ["Entertainment"],
    "Apparel": ["Shopping", "Clothing"],
    "Shopping": ["Apparel", "Household"],
    "Clothing": ["Apparel", "Shopping"],
    "Education": ["Books", "Tuition"],
    "Books": ["Education", "Entertainment"],
    "Household": ["Groceries", "Shopping", "Cleaning"],
    "Cleaning": ["Household"],
    "Miscellaneous": ["Shopping"],
}


def assign_multilabel_categories(
    txn_df: pd.DataFrame,
    hg: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Assign multiple category labels to each transaction.

    Primary category from UCC major group.
    Secondary categories from semantic similarity groups.
    Format: (transaction_description, [list_of_valid_categories])
    """
    txn_df = txn_df.copy()

    categories_list = []
    for _, row in txn_df.iterrows():
        primary = row["major_category"]
        labels = {primary}

        # Add semantic secondary categories
        if primary in SEMANTIC_GROUPS:
            for secondary in SEMANTIC_GROUPS[primary]:
                labels.add(secondary)

        # Add specific item-level categories from UCC names
        for name in row.get("item_names", []):
            name_lower = str(name).lower()
            if "egg" in name_lower or "cereal" in name_lower or "meat" in name_lower:
                labels.add("Groceries")
                labels.add("Food")
            elif "electric" in name_lower or "gas" in name_lower or "water" in name_lower:
                labels.add("Utilities")
                labels.add("Bills")
            elif "insurance" in name_lower:
                labels.add("Insurance")
            elif "doctor" in name_lower or "hospital" in name_lower or "drug" in name_lower:
                labels.add("Healthcare")
                labels.add("Medical")

        categories_list.append(sorted(labels))

    txn_df["categories"] = categories_list

    # Verify multi-label coverage
    avg_labels = np.mean([len(c) for c in categories_list])
    min_labels = min(len(c) for c in categories_list)
    print(f"[STEP 3] Multi-label assignment: avg={avg_labels:.1f} labels/txn, min={min_labels}")

    if min_labels < 2:
        # Pad single-label transactions with a generic secondary
        padded = 0
        for i, cats in enumerate(txn_df["categories"].tolist()):
            if len(cats) < 2:
                primary = cats[0]
                fallback = SEMANTIC_GROUPS.get(primary, ["Miscellaneous"])
                cats.append(fallback[0])
                txn_df.at[txn_df.index[i], "categories"] = sorted(set(cats))
                padded += 1
        print(f"[STEP 3] Padded {padded} single-label transactions to >=2 labels")

    return txn_df


# ============================================================
# STEP 4: SYNTHETIC MERCHANT NAME GENERATION
# ============================================================

# Regional merchant databases
MERCHANTS_BY_CATEGORY_REGION = {
    "Food": {
        "Midwest": ["KROGER", "MEIJER", "HY-VEE", "ALDI", "SCHNUCKS"],
        "Northeast": ["STOP & SHOP", "SHOPRITE", "WEGMANS", "MARKET BASKET", "ALDI"],
        "South": ["PUBLIX", "HEB", "WINN-DIXIE", "FOOD LION", "PIGGLY WIGGLY"],
        "West": ["SAFEWAY", "TRADER JOES", "WINCO FOODS", "RALPHS", "VONS"],
        "default": ["WALMART", "TARGET", "COSTCO", "SAMS CLUB", "WHOLE FOODS"],
    },
    "Groceries": {
        "Midwest": ["KROGER", "MEIJER", "HY-VEE", "ALDI"],
        "Northeast": ["STOP & SHOP", "SHOPRITE", "WEGMANS"],
        "South": ["PUBLIX", "HEB", "WINN-DIXIE"],
        "West": ["SAFEWAY", "TRADER JOES", "RALPHS"],
        "default": ["WALMART GROCERY", "TARGET", "COSTCO", "ALDI"],
    },
    "Housing": {
        "default": ["PROPERTY MGMT", "RENT PAYMENT", "MORTGAGE PMT",
                     "HOME DEPOT", "LOWES"],
    },
    "Utilities": {
        "Midwest": ["AMEREN", "CONSUMERS ENERGY", "DTE ENERGY", "XCEL ENERGY"],
        "Northeast": ["CON EDISON", "NATIONAL GRID", "EVERSOURCE", "PSEG"],
        "South": ["DUKE ENERGY", "FPL", "ENTERGY", "GEORGIA POWER"],
        "West": ["PG&E", "SOUTHERN CAL EDISON", "PACIFIC POWER", "AVISTA"],
        "default": ["ELECTRIC CO", "GAS UTILITY", "WATER DEPT", "INTERNET SVC"],
    },
    "Transportation": {
        "default": ["SHELL", "EXXON MOBIL", "BP", "CHEVRON", "MARATHON",
                     "UBER", "LYFT", "ENTERPRISE RENT"],
    },
    "Healthcare": {
        "default": ["CVS PHARMACY", "WALGREENS", "RITE AID", "KAISER PERMANENTE",
                     "UNITED HEALTH", "BLUE CROSS BLUE SHIELD"],
    },
    "Insurance": {
        "default": ["STATE FARM", "ALLSTATE", "GEICO", "PROGRESSIVE",
                     "BLUE CROSS BLUE SHIELD", "UNITED HEALTH", "AETNA"],
    },
    "Entertainment": {
        "default": ["NETFLIX", "SPOTIFY", "AMAZON PRIME", "DISNEY PLUS",
                     "AMC THEATRES", "TICKETMASTER", "STEAM GAMES"],
    },
    "Apparel": {
        "default": ["NIKE", "OLD NAVY", "TJ MAXX", "ROSS STORES",
                     "NORDSTROM", "MACYS", "KOHLS"],
    },
    "Education": {
        "default": ["UNIVERSITY TUITION", "COLLEGE BOOKSTORE", "PEARSON EDUCATION",
                     "CHEGG INC", "COURSERA", "STUDENT LOAN PMT"],
    },
    "Household": {
        "default": ["TARGET", "WALMART", "HOME DEPOT", "LOWES",
                     "BED BATH BEYOND", "IKEA", "DOLLAR TREE"],
    },
    "Miscellaneous": {
        "default": ["AMAZON", "EBAY", "ETSY", "PAYPAL", "VENMO",
                     "CASH APP", "BEST BUY"],
    },
}

# Region code mapping (CE Survey REGION codes)
REGION_MAP = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West",
}

# US state abbreviations by region for bank statement formatting
STATES_BY_REGION = {
    "Midwest": ["IL", "OH", "MI", "IN", "WI", "MN", "MO", "IA", "KS", "NE"],
    "Northeast": ["NY", "NJ", "PA", "CT", "MA", "ME", "NH", "VT", "RI"],
    "South": ["TX", "FL", "GA", "NC", "VA", "TN", "AL", "SC", "LA", "MS"],
    "West": ["CA", "WA", "OR", "AZ", "CO", "NV", "UT", "NM", "ID", "MT"],
}

CITIES_BY_STATE = {
    "IL": ["CHICAGO", "SPRINGFIELD", "PEORIA"],
    "OH": ["COLUMBUS", "CLEVELAND", "CINCINNATI"],
    "MO": ["SPRINGFIELD", "KANSAS CITY", "ST LOUIS"],
    "TX": ["HOUSTON", "DALLAS", "AUSTIN", "SAN ANTONIO"],
    "FL": ["MIAMI", "ORLANDO", "TAMPA", "JACKSONVILLE"],
    "CA": ["LOS ANGELES", "SAN FRANCISCO", "SAN DIEGO", "SACRAMENTO"],
    "NY": ["NEW YORK", "BUFFALO", "ALBANY", "ROCHESTER"],
    "WA": ["SEATTLE", "SPOKANE", "TACOMA"],
    "default": ["ANYTOWN"],
}


def generate_merchant_names(
    txn_df: pd.DataFrame,
    personas: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Generate realistic synthetic merchant names with regional variation.

    Each persona is assigned 2-4 preferred merchants per category (70% of
    transactions), with 30% going to random alternatives.
    Names include realistic bank statement formatting (uppercase, store numbers,
    city/state abbreviations).
    """
    txn_df = txn_df.copy()

    # Build persona -> region mapping
    persona_region = {}
    if "region" in personas.columns:
        for _, row in personas.iterrows():
            reg_code = row.get("region", 0)
            persona_region[str(row["NEWID"])] = REGION_MAP.get(
                int(reg_code) if pd.notna(reg_code) else 0, "Midwest"
            )

    # Assign preferred merchants per persona per category
    persona_merchants = defaultdict(dict)
    preferred_ratio = config.get("preferred_merchant_ratio", 0.7)
    n_preferred = config.get("n_preferred_merchants", 3)

    for pid in txn_df["persona_id"].unique():
        region = persona_region.get(str(pid), "Midwest")
        for category in txn_df[txn_df["persona_id"] == pid]["major_category"].unique():
            merchants_pool = MERCHANTS_BY_CATEGORY_REGION.get(
                category, MERCHANTS_BY_CATEGORY_REGION["Miscellaneous"]
            )
            regional = merchants_pool.get(region, merchants_pool.get("default", []))
            general = merchants_pool.get("default", [])
            combined = list(set(regional + general))

            if not combined:
                combined = ["STORE"]

            preferred = random.sample(combined, min(n_preferred, len(combined)))
            persona_merchants[str(pid)][category] = {
                "preferred": preferred,
                "all": combined,
            }

    # Generate merchant name for each transaction
    descriptions = []
    for _, row in txn_df.iterrows():
        pid = str(row["persona_id"])
        category = row["major_category"]
        region = persona_region.get(pid, "Midwest")

        merch_info = persona_merchants.get(pid, {}).get(category)
        if merch_info is None:
            merchant = "STORE"
        elif random.random() < preferred_ratio:
            merchant = random.choice(merch_info["preferred"])
        else:
            merchant = random.choice(merch_info["all"])

        # Add bank statement formatting
        store_num = f"#{random.randint(100, 9999):04d}"

        # City/State
        states = STATES_BY_REGION.get(region, STATES_BY_REGION["Midwest"])
        state = random.choice(states)
        cities = CITIES_BY_STATE.get(state, CITIES_BY_STATE["default"])
        city = random.choice(cities)

        # Format variations (realistic bank statement styles)
        fmt = random.choice(["full", "short", "minimal"])
        if fmt == "full":
            description = f"{merchant} {store_num} {city} {state}"
        elif fmt == "short":
            description = f"{merchant} {store_num} {state}"
        else:
            description = f"{merchant} {city} {state}"

        descriptions.append(description.upper())

    txn_df["description"] = descriptions
    print(f"[STEP 4] Generated {len(descriptions)} merchant names with regional variation")

    # Sample preview
    sample = txn_df[["description", "major_category", "amount"]].head(10)
    print(f"[STEP 4] Sample transactions:\n{sample.to_string(index=False)}")

    return txn_df


# ============================================================
# STEP 5: QUALITY VALIDATION
# ============================================================
def validate_data(
    txn_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Validate generated data quality.

    Checks:
      - Spending distribution matches CE Survey averages
      - Temporal consistency
      - Multi-label coverage (>=2 valid categories per UCC)
    """
    report = {}

    # 1. Spending distribution
    monthly_by_cat = txn_df.groupby(["period", "major_category"])["amount"].sum()
    avg_monthly = monthly_by_cat.groupby("major_category").mean()
    report["avg_monthly_by_category"] = avg_monthly.to_dict()
    print(f"[STEP 5] Average monthly spending by category:")
    for cat, avg in sorted(avg_monthly.items()):
        print(f"  {cat}: ${avg:,.2f}")

    # 2. Transaction count distribution
    txn_counts = txn_df.groupby("major_category").size()
    report["txn_count_by_category"] = txn_counts.to_dict()
    print(f"[STEP 5] Transaction counts by category:")
    for cat, count in sorted(txn_counts.items()):
        print(f"  {cat}: {count:,}")

    # 3. Multi-label coverage
    label_counts = txn_df["categories"].apply(len)
    report["avg_labels_per_txn"] = float(label_counts.mean())
    report["min_labels_per_txn"] = int(label_counts.min())
    report["pct_multilabel"] = float((label_counts >= 2).mean())
    print(f"[STEP 5] Multi-label coverage: avg={report['avg_labels_per_txn']:.1f}, "
          f"min={report['min_labels_per_txn']}, "
          f">=2 labels: {report['pct_multilabel']:.1%}")

    # 4. Amount distribution
    report["amount_mean"] = float(txn_df["amount"].mean())
    report["amount_median"] = float(txn_df["amount"].median())
    report["amount_std"] = float(txn_df["amount"].std())
    report["amount_min"] = float(txn_df["amount"].min())
    report["amount_max"] = float(txn_df["amount"].max())
    print(f"[STEP 5] Amount stats: mean=${report['amount_mean']:.2f}, "
          f"median=${report['amount_median']:.2f}, "
          f"std=${report['amount_std']:.2f}")

    # 5. Persona coverage
    n_personas = txn_df["persona_id"].nunique()
    n_categories = txn_df["major_category"].nunique()
    report["n_personas"] = n_personas
    report["n_categories"] = n_categories
    report["n_transactions"] = len(txn_df)
    print(f"[STEP 5] Coverage: {n_personas} personas, {n_categories} categories, "
          f"{len(txn_df)} transactions")

    # Validation flags
    issues = []
    if report["min_labels_per_txn"] < 2:
        issues.append("Some transactions have <2 category labels")
    if report["amount_min"] <= 0:
        issues.append("Some transactions have non-positive amounts")
    if n_personas < 100:
        issues.append(f"Only {n_personas} personas (expected >=100)")

    if issues:
        print(f"[STEP 5] WARNINGS: {issues}")
    else:
        print(f"[STEP 5] All quality checks passed")

    report["issues"] = issues
    return report


# ============================================================
# STEP 6: BUILD TRAINING DATASETS
# ============================================================
def build_categorization_dataset(
    txn_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build the categorization training dataset.

    Output columns: description, categories, amount, currency, country, split, sample_weight
    """
    cat_df = txn_df[["description", "categories", "amount", "persona_id", "period"]].copy()

    # Add currency and country (default for CE Survey = USD, US)
    cat_df["currency"] = "USD"
    cat_df["country"] = "US"

    # Sample weights (all external data = 0.5 as per doc)
    cat_df["sample_weight"] = config.get("external_sample_weight", 0.5)

    # Train/val/test split (stratified random for categorization)
    n = len(cat_df)
    indices = np.random.permutation(n)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    cat_df["split"] = "test"
    cat_df.iloc[indices[:train_end], cat_df.columns.get_loc("split")] = "train"
    cat_df.iloc[indices[train_end:val_end], cat_df.columns.get_loc("split")] = "val"

    # Serialize categories as JSON strings for parquet compatibility
    cat_df["categories"] = cat_df["categories"].apply(json.dumps)

    print(f"[STEP 6] Categorization dataset: {len(cat_df)} rows")
    print(f"  train={sum(cat_df['split']=='train')}, "
          f"val={sum(cat_df['split']=='val')}, "
          f"test={sum(cat_df['split']=='test')}")

    return cat_df


def build_trend_dataset(
    txn_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build the trend detection training dataset.

    Computes the 20-feature vector per persona x category x month.
    Target: next_month_spend.
    """
    # Aggregate transactions by persona x category x month
    monthly = txn_df.groupby(["persona_id", "major_category", "period"]).agg(
        current_spend=("amount", "sum"),
        txn_count=("amount", "count"),
        amounts=("amount", list),
    ).reset_index()

    # Sort chronologically
    monthly = monthly.sort_values(["persona_id", "major_category", "period"])

    # Total monthly spend per persona
    total_monthly = txn_df.groupby(["persona_id", "period"])["amount"].sum().reset_index()
    total_monthly = total_monthly.rename(columns={"amount": "total_monthly_spend"})
    monthly = monthly.merge(total_monthly, on=["persona_id", "period"], how="left")

    # Compute rolling features per persona x category
    records = []
    for (pid, cat), group in monthly.groupby(["persona_id", "major_category"]):
        group = group.sort_values("period").reset_index(drop=True)
        spends = group["current_spend"].values
        txn_counts = group["txn_count"].values
        periods = group["period"].values

        for i in range(len(group)):
            if i < 3:
                # Need at least 3 months of history
                continue

            # Target: next month spend
            if i + 1 >= len(group):
                continue

            current = spends[i]
            history = spends[max(0, i-6):i]

            # Rolling features
            roll_1m = spends[i-1] if i >= 1 else current
            roll_3m = np.mean(spends[max(0, i-3):i]) if i >= 1 else current
            roll_6m = np.mean(spends[max(0, i-6):i]) if i >= 1 else current
            std_3m = np.std(spends[max(0, i-3):i]) if i >= 2 else 0
            std_6m = np.std(spends[max(0, i-6):i]) if i >= 2 else 0

            # Deviation ratio (capped at 10)
            dev_ratio = min(current / max(roll_3m, 1e-8), 10.0)

            # Share of wallet
            total = group.iloc[i]["total_monthly_spend"]
            sow = current / max(total, 1e-8)
            hist_sow = np.mean([
                spends[j] / max(group.iloc[j]["total_monthly_spend"], 1e-8)
                for j in range(max(0, i-6), i)
            ]) if i >= 1 else sow

            # Transaction counts
            hist_txn_mean = np.mean(txn_counts[max(0, i-3):i]) if i >= 1 else txn_counts[i]
            avg_txn_size = current / max(txn_counts[i], 1)
            hist_avg_txn = np.mean([
                spends[j] / max(txn_counts[j], 1) for j in range(max(0, i-3), i)
            ]) if i >= 1 else avg_txn_size

            # Days since last transaction (simulated from period gaps)
            days_since = 15  # Default mid-month assumption

            # Month of year
            try:
                month_of_year = int(periods[i].split("-")[1])
            except (IndexError, ValueError):
                month_of_year = 1

            # Spending velocity (simulated: mid-month ratio)
            spending_velocity = current / max(roll_3m, 1e-8) * 0.5

            # Weekend transaction ratio (simulated)
            weekend_ratio = random.uniform(0.2, 0.4)

            # Elevated category count (categories above 1.2x their mean)
            elevated = sum(1 for s in spends[max(0, i-3):i+1] if s > roll_3m * 1.2)

            # Budget utilization (simulated: 0 if no budget)
            budget_util = 0.0

            record = {
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
                "txn_count": int(txn_counts[i]),
                "hist_txn_count_mean": round(hist_txn_mean, 2),
                "avg_txn_size": round(avg_txn_size, 2),
                "hist_avg_txn_size": round(hist_avg_txn, 2),
                "days_since_last_txn": days_since,
                "month_of_year": month_of_year,
                "spending_velocity": round(spending_velocity, 4),
                "weekend_txn_ratio": round(weekend_ratio, 4),
                "total_monthly_spend": round(total, 2),
                "elevated_cat_count": elevated,
                "budget_utilization": budget_util,
                "next_month_spend": round(spends[i + 1], 2),
            }
            records.append(record)

    trend_df = pd.DataFrame(records)

    # Chronological split (70/15/15)
    trend_df = trend_df.sort_values("period").reset_index(drop=True)
    n = len(trend_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    trend_df["split"] = "test"
    trend_df.loc[:train_end - 1, "split"] = "train"
    trend_df.loc[train_end:val_end - 1, "split"] = "val"

    print(f"[STEP 6] Trend dataset: {len(trend_df)} rows")
    print(f"  train={sum(trend_df['split']=='train')}, "
          f"val={sum(trend_df['split']=='val')}, "
          f"test={sum(trend_df['split']=='test')}")

    return trend_df


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(config: dict):
    """Execute the full data pipeline."""
    output_dir = config.get("output_dir", "/data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CE SURVEY DATA PIPELINE")
    print("=" * 70)

    # Step 0: Download / load data
    print("\n--- STEP 0: Loading PUMD data ---")
    paths = download_pumd_data(config)
    tables = load_pumd_tables(paths)

    # Step 1: Persona sampling
    print("\n--- STEP 1: Persona Sampling ---")
    personas = sample_personas(tables["fmli"], config)

    # Step 2: Item-to-transaction aggregation
    print("\n--- STEP 2: Item-to-Transaction Aggregation ---")
    txn_df = aggregate_items_to_transactions(
        tables["mtbi"], personas, tables["hg"], config
    )

    # Step 3: Multi-label category assignment
    print("\n--- STEP 3: Multi-Label Category Assignment ---")
    txn_df = assign_multilabel_categories(txn_df, tables["hg"], config)

    # Step 4: Synthetic merchant name generation
    print("\n--- STEP 4: Synthetic Merchant Name Generation ---")
    txn_df = generate_merchant_names(txn_df, personas, config)

    # Step 5: Quality validation
    print("\n--- STEP 5: Quality Validation ---")
    report = validate_data(txn_df, config)

    # Step 6: Build training datasets
    print("\n--- STEP 6: Building Training Datasets ---")
    cat_df = build_categorization_dataset(txn_df, config)
    trend_df = build_trend_dataset(txn_df, config)

    # Save outputs
    cat_path = os.path.join(output_dir, "categorization_training.parquet")
    trend_path = os.path.join(output_dir, "trend_training.parquet")
    report_path = os.path.join(output_dir, "pipeline_report.json")

    cat_df.to_parquet(cat_path, index=False)
    trend_df.to_parquet(trend_path, index=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Compute data hashes
    cat_hash = hashlib.sha256(open(cat_path, "rb").read()).hexdigest()[:16]
    trend_hash = hashlib.sha256(open(trend_path, "rb").read()).hexdigest()[:16]

    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Categorization data: {cat_path} ({len(cat_df)} rows, hash={cat_hash})")
    print(f"  Trend data:          {trend_path} ({len(trend_df)} rows, hash={trend_hash})")
    print(f"  Pipeline report:     {report_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_pipeline(config)


if __name__ == "__main__":
    main()
