#!/usr/bin/env python3
"""
Q3 Bonus: Soda data quality checks for Firefly III ML pipeline.
Validates training data before DistilBERT model training.
"""
import io, json, ast
from datetime import datetime
import pandas as pd
from minio import Minio

client = Minio("localhost:9000", access_key="firefly-access-key",
               secret_key="firefly-secret-key", secure=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("="*60)
print("SODA DATA QUALITY CHECKS — Firefly III")
print(f"Timestamp: {timestamp}")
print("="*60)

print("\nLoading training data from MinIO...")
resp = client.get_object("training-data", "bls_pipeline/categorization_training.parquet")
df = pd.read_parquet(io.BytesIO(resp.read()))
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

results = []
passed = 0
failed = 0

def check(name, condition, value, threshold, details=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition: passed += 1
    else: failed += 1
    results.append({"check": name, "status": status,
                    "value": str(value), "threshold": threshold})
    icon = "✓" if condition else "✗"
    print(f"  {icon} [{status}] {name}: {value} (expected: {threshold})")

# --- Check 1: Row count ---
print("\n--- Check 1: Row count ---")
check("row_count >= 10000", len(df) >= 10000, len(df), ">= 10000")

# --- Check 2: No missing values in critical columns ---
print("\n--- Check 2: No missing values ---")
for col in ['description', 'amount', 'categories', 'period']:
    null_count = df[col].isna().sum()
    check(f"no_nulls_in_{col}", null_count == 0, null_count, "== 0")

# --- Check 3: Amount validity ---
print("\n--- Check 3: Amount validity ---")
min_amt = df['amount'].min()
max_amt = df['amount'].max()
mean_amt = df['amount'].mean()
check("min_amount > 0", min_amt > 0, round(min_amt, 2), "> 0")
check("max_amount < 50000", max_amt < 50000, round(max_amt, 2), "< 50000")
check("mean_amount > 10", mean_amt > 10, round(mean_amt, 2), "> 10")

# --- Check 4: Category coverage ---
print("\n--- Check 4: Category coverage ---")
def get_primary(cats):
    try:
        if isinstance(cats, list): return cats[0]
        parsed = ast.literal_eval(str(cats))
        return parsed[0] if parsed else 'Other'
    except: return 'Other'

df['primary_category'] = df['categories'].apply(get_primary)
n_cats = df['primary_category'].nunique()
check("category_count >= 5", n_cats >= 5, n_cats, ">= 5")
min_cat = df['primary_category'].value_counts().min()
check("min_examples_per_category >= 50", min_cat >= 50, min_cat, ">= 50")
print(f"  Category distribution:")
for cat, cnt in df['primary_category'].value_counts().items():
    print(f"    {cat:<20} {cnt:>7,}")

# --- Check 5: No duplicate transactions ---
print("\n--- Check 5: No duplicate transactions ---")
dup_count = df.duplicated(
    subset=['description', 'amount', 'period', 'persona_id']).sum()
dup_pct = round(100 * dup_count / len(df), 2)
check("duplicate_rate < 5%", dup_pct < 5, f"{dup_count} ({dup_pct}%)", "< 5%")

# --- Check 6: Period validity ---
print("\n--- Check 6: Period validity ---")
periods = df['period'].unique()
min_period = min(periods)
max_period = max(periods)
check("earliest_period >= 2019", min_period >= "2019", min_period, ">= 2019")
check("latest_period <= 2026", max_period <= "2026-12", max_period, "<= 2026-12")
n_periods = len(periods)
check("period_spans_multiple_months", n_periods >= 2, n_periods, ">= 2 periods")

# --- Check 7: Description text quality ---
print("\n--- Check 7: Description text quality ---")
empty = (df['description'].str.strip() == '').sum()
check("no_empty_descriptions", empty == 0, empty, "== 0")
avg_len = df['description'].str.len().mean()
check("avg_description_length >= 10", avg_len >= 10, round(avg_len, 1), ">= 10 chars")
short = (df['description'].str.len() < 5).sum()
check("short_descriptions < 1%", short / len(df) < 0.01,
      f"{short} ({round(100*short/len(df),2)}%)", "< 1%")

# --- Check 8: Currency consistency ---
print("\n--- Check 8: Currency consistency ---")
currencies = df['currency'].unique().tolist()
check("single_currency", len(currencies) == 1, currencies, "exactly 1")
check("currency_is_USD", 'USD' in currencies, currencies, "contains USD")

# --- Check 9: Multi-label coverage ---
print("\n--- Check 9: Multi-label coverage ---")
def count_labels(cats):
    try:
        if isinstance(cats, list): return len(cats)
        return len(ast.literal_eval(str(cats)))
    except: return 1

label_counts = df['categories'].apply(count_labels)
avg_labels = label_counts.mean()
min_labels = label_counts.min()
check("avg_labels_per_transaction >= 1", avg_labels >= 1,
      round(avg_labels, 2), ">= 1")
check("min_labels >= 1", min_labels >= 1, min_labels, ">= 1")

# --- Check 10: Persona diversity ---
print("\n--- Check 10: Persona diversity ---")
n_personas = df['persona_id'].nunique()
check("n_personas >= 100", n_personas >= 100, n_personas, ">= 100")
txn_per_persona = df.groupby('persona_id').size()
check("min_txn_per_persona >= 1", txn_per_persona.min() >= 1,
      txn_per_persona.min(), ">= 1")
check("max_txn_per_persona < 500", txn_per_persona.max() < 500,
      txn_per_persona.max(), "< 500")

# --- Check 11: Train/Val/Test split presence ---
print("\n--- Check 11: Split validity ---")
splits = df['split'].unique().tolist()
check("has_train_split", 'train' in splits, splits, "contains 'train'")
check("has_val_split", 'val' in splits, splits, "contains 'val'")
check("has_test_split", 'test' in splits, splits, "contains 'test'")
train_pct = round(100 * (df['split'] == 'train').sum() / len(df), 1)
check("train_ratio_60-80%", 60 <= train_pct <= 80, f"{train_pct}%", "60-80%")

# Final summary
print(f"\n{'='*60}")
print(f"SODA QUALITY CHECK SUMMARY")
print(f"  Dataset    : categorization_training.parquet")
print(f"  Total rows : {len(df):,}")
print(f"  Columns    : {len(df.columns)}")
print(f"  Checks run : {passed + failed}")
print(f"  PASSED     : {passed} ✓")
print(f"  FAILED     : {failed} ✗")
score = 100 * passed // (passed + failed)
print(f"  Score      : {score}%")
if score == 100:
    print(f"  Status     : ALL CHECKS PASSED — safe to train model")
else:
    print(f"  Status     : SOME CHECKS FAILED — review before training")
print(f"{'='*60}")

# Save report to MinIO
report = {
    "pipeline": "soda_data_quality_checks",
    "framework": "soda-inspired (custom Python checks)",
    "timestamp": timestamp,
    "dataset": "training-data/bls_pipeline/categorization_training.parquet",
    "total_rows": len(df),
    "columns": list(df.columns),
    "checks_passed": passed,
    "checks_failed": failed,
    "quality_score_pct": score,
    "results": results,
    "improvement_justification": (
        "Soda-inspired data quality checks prevent corrupted or malformed BLS data "
        "from reaching the DistilBERT categorization model. Concrete example: BLS "
        "suppresses sensitive expenditure values with 'D' codes. Without quality "
        "checks, these silently become NaN or zero amounts in training data, causing "
        "the model to learn incorrect spending patterns. Row-level checks catch this "
        "before training, saving a full retraining cycle (~4-6 hours on GPU)."
    )
}
report_json = json.dumps(report, indent=2).encode()
key = f"quality_checks/soda_report_{timestamp}.json"
client.put_object("training-data", key,
                  io.BytesIO(report_json), len(report_json))
print(f"\nReport saved → training-data/{key}")
print("DONE!")
