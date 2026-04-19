#!/usr/bin/env python3
import io, os, json, hashlib, random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from minio import Minio

client = Minio("localhost:9000", access_key="firefly-access-key",
               secret_key="firefly-secret-key", secure=False)
random.seed(42)
np.random.seed(42)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MERCHANT_PATTERNS = {
    "Food at home":   ["KROGER","WALMART SUPERCENTER","WHOLE FOODS","TRADER JOES","ALDI","SAFEWAY"],
    "Food away":      ["MCDONALDS","STARBUCKS","CHIPOTLE","SUBWAY","DOORDASH","PANERA BREAD"],
    "Housing":        ["RENT PAYMENT","HOME DEPOT","LOWES","IKEA","WAYFAIR"],
    "Utilities":      ["DUKE ENERGY","AMEREN ELECTRIC","AT&T","COMCAST","VERIZON"],
    "Clothing":       ["MACYS","TARGET","OLD NAVY","H&M","TJ MAXX","KOHLS"],
    "Transportation": ["SHELL OIL","BP GAS","CHEVRON","GEICO AUTO","UBER"],
    "Healthcare":     ["BLUE CROSS","WALGREENS","CVS PHARMACY","QUEST DIAGNOSTICS"],
    "Insurance":      ["STATE FARM","ALLSTATE","NORTHWESTERN MUTUAL"],
    "Education":      ["TUITION PAYMENT","COURSERA","AMAZON TEXTBOOKS"],
    "Entertainment":  ["NETFLIX","SPOTIFY","AMC THEATRES","AMAZON PRIME","HULU"],
    "Personal care":  ["GREAT CLIPS","ULTA BEAUTY","SEPHORA"],
    "Other":          ["AMAZON MKTPL","WALMART","TARGET","COSTCO"],
}
UCC_TO_CAT = {
    "01":"Food at home","02":"Food at home","03":"Food at home",
    "07":"Food away","08":"Food away",
    "21":"Housing","22":"Housing","23":"Housing","24":"Housing",
    "25":"Utilities","26":"Utilities","27":"Utilities",
    "36":"Clothing","37":"Clothing",
    "45":"Transportation","47":"Transportation","48":"Transportation",
    "53":"Healthcare","54":"Healthcare","55":"Healthcare","56":"Healthcare",
    "57":"Insurance","58":"Education",
    "60":"Entertainment","61":"Entertainment","62":"Entertainment",
    "63":"Personal care",
}
SUFFIXES = ["", " #1247", " #892", " #3301", " #421"]
CITIES = ["SPRINGFIELD IL","COLUMBUS OH","CHARLOTTE NC","AUSTIN TX","PHOENIX AZ"]

def ucc_to_category(ucc_str):
    prefix = str(ucc_str).strip().zfill(6)[:2]
    return UCC_TO_CAT.get(prefix, "Other")

def make_merchant(category):
    merchants = MERCHANT_PATTERNS.get(category, ["GENERIC STORE"])
    name = random.choice(merchants)
    sfx = random.choice(SUFFIXES)
    city = (" " + random.choice(CITIES)) if random.random() > 0.5 else ""
    return f"{name}{sfx}{city}".strip()

print("Loading BLS data from MinIO (10% sample per year)...")
objects = list(client.list_objects("processed-data",
               prefix="bls_ce_survey/processed/", recursive=True))
parquet_objects = [o for o in objects if o.object_name.endswith('.parquet')]

all_frames = []
for obj in parquet_objects:
    resp = client.get_object("processed-data", obj.object_name)
    df = pd.read_parquet(io.BytesIO(resp.read()))
    df = df.sample(frac=0.10, random_state=42)
    all_frames.append(df)
    print(f"  {obj.object_name.split('/')[-1]}: {len(df):,} rows")

bls_df = pd.concat(all_frames, ignore_index=True)
del all_frames
print(f"Total: {len(bls_df):,} rows")

# Auto-detect columns
print(f"\nAll columns: {list(bls_df.columns)}")
# Find cost column - try multiple names
cost_col = None
for candidate in ['COST_', 'COST', 'EXPN', 'AMOUNT']:
    matches = [c for c in bls_df.columns if candidate in c.upper()]
    if matches:
        cost_col = matches[0]
        break

# Find UCC/category column
ucc_col = None
for candidate in ['UCCSEQ', 'UCC', 'SEQNO']:
    matches = [c for c in bls_df.columns if candidate in c.upper()]
    if matches:
        ucc_col = matches[0]
        break

# Find item name column
item_col = next((c for c in bls_df.columns if 'ITEM' in c.upper() or 'EXPNAME' in c.upper()), None)

print(f"Using: cost_col={cost_col}, ucc_col={ucc_col}, item_col={item_col}")

# Convert cost column to numeric, coercing 'D' and other non-numeric values to NaN
if cost_col:
    bls_df[cost_col] = pd.to_numeric(bls_df[cost_col], errors='coerce')
    valid_costs = bls_df[cost_col].dropna()
    print(f"Valid numeric costs: {len(valid_costs):,} out of {len(bls_df):,} rows")
    print(f"Sample cost values: {valid_costs.head(5).tolist()}")
    # Filter to only valid costs
    bls_df = bls_df[bls_df[cost_col].notna() & (bls_df[cost_col] > 0) & (bls_df[cost_col] <= 50000)].copy()
    print(f"After filtering: {len(bls_df):,} rows with valid costs")

print("\nGenerating synthetic transactions...")
transactions = []
base_date = datetime(2020, 1, 1)

for _, row in bls_df.iterrows():
    try:
        # Get cost (already filtered to valid numeric values)
        cost = float(row[cost_col]) if cost_col and pd.notna(row[cost_col]) else 0
        if cost <= 0 or cost > 50000:
            continue

        # Get UCC - UCCSEQ is a sequential number, use EXPNAME for category
        expname = str(row.get('EXPNAME', '')).strip() if 'EXPNAME' in bls_df.columns else ''
        item_name = str(row.get(item_col, 'Unknown')) if item_col else expname

        # Map to category using UCC prefix or expname
        ucc_val = str(row[ucc_col]) if ucc_col else "999999"
        # Try expname-based category first
        expname_lower = expname.lower()
        if any(w in expname_lower for w in ['food','grocer','cereal','meat','dairy','fruit','veg','egg']):
            category = "Food at home"
        elif any(w in expname_lower for w in ['restaurant','dining','fast food','lunch','dinner']):
            category = "Food away"
        elif any(w in expname_lower for w in ['rent','mortgage','furnit','household']):
            category = "Housing"
        elif any(w in expname_lower for w in ['electric','gas','water','phone','internet','cable']):
            category = "Utilities"
        elif any(w in expname_lower for w in ['cloth','apparel','shoe','wear']):
            category = "Clothing"
        elif any(w in expname_lower for w in ['car','vehicle','gas','fuel','transport','bus','train']):
            category = "Transportation"
        elif any(w in expname_lower for w in ['health','medical','doctor','drug','pharma','dental']):
            category = "Healthcare"
        elif any(w in expname_lower for w in ['insur']):
            category = "Insurance"
        elif any(w in expname_lower for w in ['school','tuition','edu','book']):
            category = "Education"
        elif any(w in expname_lower for w in ['entertain','film','music','sport','hobby']):
            category = "Entertainment"
        else:
            category = ucc_to_category(ucc_val)

        merchant = make_merchant(category)
        days = random.randint(0, 365*4)
        date = base_date + timedelta(days=days)
        amount = cost * random.uniform(0.9, 1.1)

        transactions.append({
            "transaction_description": merchant,
            "amount": round(amount, 2),
            "date": date.strftime("%Y-%m-%d"),
            "category": category,
            "ucc_seq": ucc_val,
            "item_name": item_name,
            "source_year": str(row.get('SOURCE_YEAR', 'Unknown')),
            "synthetic": True,
        })
    except Exception:
        continue

print(f"Generated {len(transactions):,} synthetic transactions")
if not transactions:
    print("ERROR: No transactions generated!")
    import sys; sys.exit(1)

synthetic_df = pd.DataFrame(transactions)
print(f"Category distribution:\n{synthetic_df['category'].value_counts().to_string()}")

buf = io.BytesIO()
synthetic_df.to_parquet(buf, index=False)
buf.seek(0); data = buf.read()
data_hash = hashlib.sha256(data).hexdigest()[:16]
key = f"synthetic_transactions/v1_{timestamp}_{data_hash}.parquet"
client.put_object("processed-data", key, io.BytesIO(data), len(data))

meta = {
    "pipeline": "synthetic_data_generation", "version": "1.0.0",
    "timestamp": timestamp, "source": "BLS CE Survey PUMD 2020-2024 (10% sample)",
    "n_transactions": len(synthetic_df), "data_hash": data_hash, "object_key": key,
    "categories": synthetic_df['category'].value_counts().to_dict(),
}
mj = json.dumps(meta, indent=2).encode()
client.put_object("processed-data", f"synthetic_transactions/metadata_{timestamp}.json",
                  io.BytesIO(mj), len(mj))
print(f"\nSaved {len(synthetic_df):,} transactions → processed-data/{key}")
print("SYNTHETIC DATA GENERATION COMPLETE!")
