"""Pipeline configuration — all constants live here."""
import os
from pathlib import Path


# ── MinIO connection ─────────────────────────────────────────────────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "129.114.24.242:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"

# ── Bucket names ─────────────────────────────────────────────────────────────
BUCKET_PRODUCTION_LOGS = "production-logs"
BUCKET_TRAINING_DATA   = "training-data"
BUCKET_RETRAINING_DATA = "retraining-data"

# ── Path prefixes within production-logs ─────────────────────────────────────
PREFIX_CATEGORIZATION_LOGS = "categorization"
PREFIX_ANOMALY_FEEDBACK    = "anomaly_feedback"

# ── Path prefixes within training-data (existing CE Survey pipeline) ─────────
PREFIX_CE_CATEGORIZATION = "ce_survey/categorization"
PREFIX_CE_ANOMALY        = "ce_survey/anomaly"

# ── Cadence / lookback windows ───────────────────────────────────────────────
CATEGORIZATION_LOOKBACK_DAYS = 7   # weekly retrain
ANOMALY_LOOKBACK_DAYS        = 30  # monthly retrain

# ── Sample weights (project doc §5.1.2) ──────────────────────────────────────
WEIGHT_OVERRIDDEN = 1.0   # strong label — user actively corrected
WEIGHT_ABSTAINED  = 1.0   # strong label — user picked on low-confidence case
WEIGHT_ACCEPTED   = 0.3   # weak label — user didn't verify
WEIGHT_EXTERNAL   = 0.5   # CE Survey background data

WEIGHT_HELPFUL    = 1.0   # anomaly: alert was correct
WEIGHT_EXPECTED   = 0.5   # anomaly: spending is normal for this user
WEIGHT_NOT_USEFUL = 1.0   # anomaly: spurious alert (negative example)

# ── Train/val/test split (chronological, §7.3) ───────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Quality gates ────────────────────────────────────────────────────────────
MIN_TOTAL_ROWS            = 500    # fail run if fewer
MAX_SCHEMA_REJECT_RATE    = 0.05   # fail if >5% rejected
MIN_UNIQUE_CATEGORIES     = 10     # categorization only
MIN_EXAMPLES_PER_CATEGORY = 20     # categorization only
MIN_MIX_RATIO             = 0.4    # production share ≥ 40%
MAX_MIX_RATIO             = 0.6    # production share ≤ 60%

# ── Feature pipeline version (§9 registry field) ─────────────────────────────
CATEGORIZATION_FEATURE_VERSION = "categorization.v1"
ANOMALY_FEATURE_VERSION        = "anomaly.v1"

# ── Schema contract version (bump on breaking change) ────────────────────────
SCHEMA_VERSION = "1.0"

# ── Local temp scratch space ─────────────────────────────────────────────────
TMP_DIR = Path(os.getenv("FIREFLY_TMP", "/tmp/firefly-retraining"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── The 20 feature-vector keys (project doc §4.2.2) ──────────────────────────
ANOMALY_FEATURE_KEYS = [
    "current_spend", "rolling_mean_1m", "rolling_mean_3m", "rolling_mean_6m",
    "rolling_std_3m", "rolling_std_6m", "deviation_ratio", "share_of_wallet",
    "hist_share_of_wallet", "txn_count", "hist_txn_count_mean", "avg_txn_size",
    "hist_avg_txn_size", "days_since_last_txn", "month_of_year",
    "spending_velocity", "weekend_txn_ratio", "total_monthly_spend",
    "elevated_cat_count", "budget_utilization",
]
