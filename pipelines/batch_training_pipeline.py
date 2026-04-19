#!/usr/bin/env python3
import io, json, hashlib, ast
from datetime import datetime
import pandas as pd
import numpy as np
from minio import Minio

client = Minio("localhost:9000", access_key="firefly-access-key",
               secret_key="firefly-secret-key", secure=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
version = datetime.now().strftime("%Y%m%d")

print("="*55)
print("BATCH TRAINING PIPELINE")
print(f"Version: {version} | Timestamp: {timestamp}")
print("="*55)

print("\nStep 1: Loading training data from MinIO...")
resp = client.get_object("training-data", "bls_pipeline/categorization_training.parquet")
df = pd.read_parquet(io.BytesIO(resp.read()))
print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")

def get_primary(cats):
    try:
        if isinstance(cats, list): return cats[0]
        parsed = ast.literal_eval(str(cats))
        return parsed[0] if parsed else 'Other'
    except: return 'Other'

df['primary_category'] = df['categories'].apply(get_primary)
print(f"  Categories: {sorted(df['primary_category'].unique().tolist())}")

print("\nStep 2: Candidate selection...")
n_before = len(df)

# Remove nulls
df = df.dropna(subset=['description', 'amount'])
df = df[df['amount'] > 0]

# Remove low-count categories
cat_counts = df['primary_category'].value_counts()
valid_cats = cat_counts[cat_counts >= 50].index.tolist()
df = df[df['primary_category'].isin(valid_cats)].copy()

# Remove p99 outliers per category
keep_idx = []
for cat in df['primary_category'].unique():
    mask = df['primary_category'] == cat
    cat_df = df[mask]
    p99 = cat_df['amount'].quantile(0.99)
    keep_idx.extend(cat_df[cat_df['amount'] <= p99].index.tolist())
df = df.loc[keep_idx].copy()

print(f"  After candidate selection: {len(df):,} rows (removed {n_before-len(df)})")
print(f"  Category distribution:")
for cat, count in df['primary_category'].value_counts().items():
    print(f"    {cat:<20} {count:>6,}")

print("\nStep 3: Chronological split (leakage prevention)...")
df['date'] = pd.to_datetime(df['period'])
df = df.sort_values('date').reset_index(drop=True)
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()
assert train_df['date'].max() <= val_df['date'].min(), "LEAKAGE DETECTED!"
assert val_df['date'].max() <= test_df['date'].min(), "LEAKAGE DETECTED!"
print(f"  Train: {len(train_df):,} rows ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
print(f"  Val:   {len(val_df):,} rows ({val_df['date'].min().date()} → {val_df['date'].max().date()})")
print(f"  Test:  {len(test_df):,} rows ({test_df['date'].min().date()} → {test_df['date'].max().date()})")
print(f"  Leakage check: PASSED ✓")

print("\nStep 4: Saving versioned datasets to MinIO...")
def save_split(split_df, split_name):
    buf = io.BytesIO()
    split_df.to_parquet(buf, index=False)
    buf.seek(0); data = buf.read()
    data_hash = hashlib.sha256(data).hexdigest()[:16]
    key = f"versioned/v{version}/{split_name}_{data_hash}.parquet"
    client.put_object("training-data", key, io.BytesIO(data), len(data))
    print(f"  Saved {split_name}: {len(split_df):,} rows → training-data/{key}")
    return key, data_hash

train_key, train_hash = save_split(train_df, "train")
val_key,   val_hash   = save_split(val_df,   "val")
test_key,  test_hash  = save_split(test_df,  "test")

manifest = {
    "dataset_version": version,
    "created_at": timestamp,
    "pipeline_version": "1.0.0",
    "source": "bls_pipeline/categorization_training.parquet",
    "total_examples": len(df),
    "candidate_selection": {
        "min_examples_per_category": 50,
        "outlier_threshold": "p99 per category",
        "null_removal": True,
    },
    "leakage_prevention": {
        "split_method": "chronological",
        "random_shuffle": False,
        "ratios": "70/15/15"
    },
    "splits": {
        "train": {"key": train_key, "hash": train_hash, "n_rows": len(train_df)},
        "val":   {"key": val_key,   "hash": val_hash,   "n_rows": len(val_df)},
        "test":  {"key": test_key,  "hash": test_hash,  "n_rows": len(test_df)},
    },
    "categories": df['primary_category'].value_counts().to_dict(),
}
mj = json.dumps(manifest, indent=2).encode()
manifest_key = f"versioned/v{version}/manifest_{timestamp}.json"
client.put_object("training-data", manifest_key, io.BytesIO(mj), len(mj))
print(f"  Saved manifest → training-data/{manifest_key}")

print(f"\n{'='*55}")
print(f"BATCH PIPELINE COMPLETE")
print(f"  Version  : {version}")
print(f"  Total    : {len(df):,} examples")
print(f"  Splits   : {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"  Leakage  : PASSED (chronological split)")
print(f"  Location : training-data/versioned/v{version}/")
print(f"{'='*55}")
