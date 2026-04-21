"""Weekly categorization retraining dataset builder.

Reads the last 7 days of production-logs/feedback/categorization/ events,
merges them 50/50 with the external CE Survey training data, and writes a
versioned dataset to retraining-data/categorization/v=YYYY-MM-DD/.

Triggered by cron every Sunday at 3 AM (see cron/firefly-retraining.cron).
Follows the 6-step pipeline:
    1. Ingest    2. Validate    3. Transform
    4. Weight    5. Merge+split 6. Version+write
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from pydantic import ValidationError

import config
import utils
from schemas import CategorizationFeedbackEvent

log = utils.setup_logging("categorization")


# ─────────────────────────────────────────────────────────────────────────────
def step_1_ingest(con, cutoff_iso: str):
    """Read single-event JSON files from production-logs + Parquet from training-data."""
    log.info("STEP 1 — Ingest")

    # Feedback events are written one-per-file:
    #   feedback/categorization/<timestamp>_<id>.json
    # They may or may not be Hive-partitioned under year=/month=/day=/ — glob
    # recursively to cover both layouts.
    prod_glob = (
        f"s3://{config.BUCKET_PRODUCTION_LOGS}/"
        f"{config.PREFIX_CATEGORIZATION_FEEDBACK}/**/*.json"
    )
    try:
        prod_df = con.execute(f"""
            SELECT *
            FROM read_json_auto('{prod_glob}', format='auto', ignore_errors=true)
            WHERE recorded_at >= TIMESTAMP '{cutoff_iso}'
              AND event_type = 'feedback/categorization'
        """).df()
    except Exception as e:
        log.warning("No categorization feedback found (%s) — first run is OK", e)
        prod_df = pd.DataFrame()

    ext_glob = (
        f"s3://{config.BUCKET_TRAINING_DATA}/"
        f"{config.PREFIX_CE_CATEGORIZATION}/*.parquet"
    )
    ext_df = con.execute(f"""
        SELECT
            transaction_description,
            amount,
            currency,
            country,
            primary_category AS final_category,
            CAST(NULL AS TIMESTAMP) AS timestamp
        FROM read_parquet('{ext_glob}')
    """).df()

    log.info("  production rows: %d", len(prod_df))
    log.info("  external rows:   %d", len(ext_df))
    return prod_df, ext_df


# ─────────────────────────────────────────────────────────────────────────────
def step_2_validate(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Pydantic validation against CategorizationFeedbackEvent + dedup."""
    log.info("STEP 2 — Validate")

    if prod_df.empty:
        return prod_df

    valid, rejected = [], 0
    for rec in prod_df.to_dict("records"):
        try:
            CategorizationFeedbackEvent(**rec)
            valid.append(rec)
        except ValidationError as e:
            rejected += 1
            if rejected <= 5:
                err = e.errors()[0]
                log.warning("  reject: %s → %s", err.get("loc"), err.get("msg"))

    df = pd.DataFrame(valid)
    total = len(prod_df)
    reject_rate = rejected / max(total, 1)
    log.info("  valid: %d | rejected: %d (%.1f%%)",
             len(df), rejected, reject_rate * 100)

    if total < config.MIN_REJECT_CHECK_SAMPLE:
        log.info("  sample %d < %d — skipping reject-rate gate (bootstrap)",
                 total, config.MIN_REJECT_CHECK_SAMPLE)
    elif reject_rate > config.MAX_SCHEMA_REJECT_RATE:
        fail(f"Schema reject rate {reject_rate:.1%} exceeds limit "
             f"{config.MAX_SCHEMA_REJECT_RATE:.0%}")

    if not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=["event_id"], keep="last")
        log.info("  dedup removed: %d", before - len(df))

    return df


# ─────────────────────────────────────────────────────────────────────────────
def _flatten_feedback(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Pull the nested feedback dict up into flat training columns."""
    rows = []
    for r in prod_df.to_dict("records"):
        fb = r.get("feedback") or {}
        meta = fb.get("metadata") or {}
        fv = fb.get("final_value") or {}
        amt = meta.get("amount")
        try:
            amount = float(amt) if amt is not None else None
        except (TypeError, ValueError):
            amount = None
        rows.append({
            "event_id":                r.get("event_id"),
            "transaction_id":          fb.get("transaction_id"),
            "timestamp":               fb.get("timestamp"),
            "user_id":                 str(fb.get("user_id")),
            "user_action":             fb.get("action"),
            "transaction_description": meta.get("description"),
            "amount":                  amount,
            "currency":                meta.get("currency"),
            "country":                 meta.get("country"),
            "model_version":           fb.get("model_version"),
            "final_category":          fv.get("category"),
        })
    out = pd.DataFrame(rows)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out


def step_3_transform(prod_df: pd.DataFrame, ext_df: pd.DataFrame):
    """Flatten nested feedback; derive feedback_weight; tag source."""
    log.info("STEP 3 — Transform")

    if not prod_df.empty:
        prod_df = _flatten_feedback(prod_df)
        # Drop rows without a label
        prod_df = prod_df[prod_df["user_action"] != "ignored"].copy()
        prod_df = prod_df[prod_df["final_category"].notna()]

        weight_map = {
            "overridden": config.WEIGHT_OVERRIDDEN,
            "abstained":  config.WEIGHT_ABSTAINED,
            "accepted":   config.WEIGHT_ACCEPTED,
        }
        prod_df["feedback_weight"] = prod_df["user_action"].map(weight_map)
        prod_df["source"] = "production"

    ext_df = ext_df.copy()
    ext_df["feedback_weight"] = config.WEIGHT_EXTERNAL
    ext_df["source"] = "external"
    ext_df["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC")

    log.info("  production labeled: %d | external: %d", len(prod_df), len(ext_df))
    return prod_df, ext_df


# ─────────────────────────────────────────────────────────────────────────────
def step_4_weight_mix(prod_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    """50% production + 50% external sample mix (§5.1.2)."""
    log.info("STEP 4 — Weight / sample mix")

    if prod_df.empty:
        fail("No labeled production rows — cannot build retraining dataset")

    n = min(len(prod_df), len(ext_df))
    if n == 0:
        fail("External training data is empty — check training-data bucket")

    prod_sample = prod_df.sample(n=n, random_state=42)
    ext_sample  = ext_df.sample(n=n, random_state=42)
    common = sorted(set(prod_sample.columns) & set(ext_sample.columns))
    mixed = pd.concat([prod_sample[common], ext_sample[common]], ignore_index=True)

    ratio = (mixed["source"] == "production").mean()
    log.info("  mixed rows: %d | production share: %.2f", len(mixed), ratio)
    if not (config.MIN_MIX_RATIO <= ratio <= config.MAX_MIX_RATIO):
        fail(f"Mix ratio {ratio:.2f} outside "
             f"[{config.MIN_MIX_RATIO}, {config.MAX_MIX_RATIO}]")

    if len(mixed) < config.MIN_TOTAL_ROWS:
        fail(f"Only {len(mixed)} rows — need ≥{config.MIN_TOTAL_ROWS}")

    cat_counts = mixed["final_category"].value_counts()
    if len(cat_counts) < config.MIN_UNIQUE_CATEGORIES:
        fail(f"Only {len(cat_counts)} categories — "
             f"need ≥{config.MIN_UNIQUE_CATEGORIES}")
    sparse = cat_counts[cat_counts < config.MIN_EXAMPLES_PER_CATEGORY]
    if len(sparse) > 0:
        log.warning("  %d categories have <%d examples: %s",
                    len(sparse), config.MIN_EXAMPLES_PER_CATEGORY,
                    sparse.index.tolist()[:5])

    return mixed


# ─────────────────────────────────────────────────────────────────────────────
def step_5_merge_split(mixed: pd.DataFrame):
    """Temporal chronological split 70/15/15 (§7.3)."""
    log.info("STEP 5 — Merge + split")

    mixed = mixed.sort_values("timestamp").reset_index(drop=True)
    n = len(mixed)
    train_end = int(config.TRAIN_RATIO * n)
    val_end   = int((config.TRAIN_RATIO + config.VAL_RATIO) * n)

    train = mixed.iloc[:train_end]
    val   = mixed.iloc[train_end:val_end]
    test  = mixed.iloc[val_end:]

    log.info("  train: %d | val: %d | test: %d", len(train), len(val), len(test))
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
def step_6_version_write(train, val, test, now: datetime):
    """Write parquets + manifest.json + update latest.json."""
    log.info("STEP 6 — Version + write")

    version = utils.version_folder(now)
    local_dir = config.TMP_DIR / "categorization" / version
    local_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        p = local_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p

    data_hash = utils.sha256_of_files(list(paths.values()))

    manifest = {
        "version": version,
        "built_at": now.isoformat(),
        "model_target": "categorization",
        "lookback_days": config.CATEGORIZATION_LOOKBACK_DAYS,
        "rows": {k: int(len(v)) for k, v in
                 [("train", train), ("val", val), ("test", test)]},
        "category_distribution": pd.concat([train, val, test])["final_category"]
            .value_counts().to_dict(),
        "source_mix": pd.concat([train, val, test])["source"]
            .value_counts().to_dict(),
        "training_data_hash": data_hash,
        "feature_version": config.CATEGORIZATION_FEATURE_VERSION,
        "schema_version": config.SCHEMA_VERSION,
    }
    manifest_path = local_dir / "manifest.json"
    utils.write_json(manifest_path, manifest)

    client = utils.get_minio_client()
    utils.ensure_bucket(client, config.BUCKET_RETRAINING_DATA)

    base_key = f"categorization/{version}"
    for fname in ["train.parquet", "val.parquet", "test.parquet", "manifest.json"]:
        utils.upload_file(client, config.BUCKET_RETRAINING_DATA,
                          f"{base_key}/{fname}", local_dir / fname)
    utils.update_latest_pointer(client, "categorization", version, now)

    log.info("  uploaded to s3://%s/%s/", config.BUCKET_RETRAINING_DATA, base_key)
    log.info("  training_data_hash = %s", data_hash)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
def fail(msg: str):
    log.error("PIPELINE FAILED: %s", msg)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=config.CATEGORIZATION_LOOKBACK_DAYS)).isoformat()

    log.info("=" * 70)
    log.info("Categorization retraining build — version=%s",
             utils.version_folder(now))
    log.info("Lookback: last %d days (cutoff %s)",
             config.CATEGORIZATION_LOOKBACK_DAYS, cutoff)
    log.info("=" * 70)

    con = utils.get_duckdb()
    prod_df, ext_df = step_1_ingest(con, cutoff)
    prod_df = step_2_validate(prod_df)
    prod_df, ext_df = step_3_transform(prod_df, ext_df)
    mixed = step_4_weight_mix(prod_df, ext_df)
    train, val, test = step_5_merge_split(mixed)
    manifest = step_6_version_write(train, val, test, now)

    log.info("=" * 70)
    log.info("SUCCESS — %s", json.dumps(manifest["rows"]))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
