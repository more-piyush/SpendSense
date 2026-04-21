"""Monthly trend-detection retraining dataset builder.

Reads the last 30 days of production-logs/feedback/trend/ events, merges
them 50/50 with the external CE Survey monthly aggregates, and writes a
versioned dataset to retraining-data/anomaly/v=YYYY-MM-DD/.

Triggered by cron on the 1st of every month at 3 AM.

NOTE: the trend feedback envelope is PROVISIONAL. It mirrors the confirmed
categorization feedback shape. Once a real feedback/trend sample lands,
verify the nested keys (features, predicted_value, final_value) and adjust.
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from pydantic import ValidationError

import config
import utils
from schemas import TrendFeedbackEvent

log = utils.setup_logging("trend")


# ─────────────────────────────────────────────────────────────────────────────
def step_1_ingest(con, cutoff_iso: str):
    log.info("STEP 1 — Ingest")

    prod_glob = (
        f"s3://{config.BUCKET_PRODUCTION_LOGS}/"
        f"{config.PREFIX_TREND_FEEDBACK}/**/*.json"
    )
    try:
        prod_df = con.execute(f"""
            SELECT *
            FROM read_json_auto('{prod_glob}', format='auto', ignore_errors=true)
            WHERE recorded_at >= TIMESTAMP '{cutoff_iso}'
              AND event_type = 'feedback/trend'
        """).df()
    except Exception as e:
        log.warning("No trend feedback found (%s) — first run is OK", e)
        prod_df = pd.DataFrame()

    ext_glob = f"s3://{config.BUCKET_TRAINING_DATA}/{config.PATH_CE_ANOMALY}"
    ext_df = con.execute(f"""
        SELECT *
        FROM read_parquet('{ext_glob}')
    """).df()

    log.info("  production rows: %d", len(prod_df))
    log.info("  external rows:   %d", len(ext_df))
    return prod_df, ext_df


# ─────────────────────────────────────────────────────────────────────────────
def step_2_validate(prod_df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 2 — Validate")

    if prod_df.empty:
        return prod_df

    valid, rejected = [], 0
    for rec in prod_df.to_dict("records"):
        try:
            TrendFeedbackEvent(**rec)
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
        fail(f"Schema reject rate {reject_rate:.1%} exceeds limit")

    if not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=["event_id"], keep="last")
        log.info("  dedup removed: %d", before - len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _flatten_trend_feedback(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Pull nested feedback up into flat columns + unpack 20-key features."""
    rows = []
    for r in prod_df.to_dict("records"):
        fb = r.get("feedback") or {}
        feats = fb.get("features") or {}
        pv = fb.get("predicted_value") or {}
        fv = fb.get("final_value") or {}
        base = {
            "event_id":        r.get("event_id"),
            "timestamp":       fb.get("timestamp"),
            "user_id":         str(fb.get("user_id")),
            "category":        fb.get("category"),
            "month":           fb.get("period"),
            "model_version":   fb.get("model_version"),
            "user_feedback":   fv.get("user_feedback") or fb.get("action"),
            "predicted_spend": pv.get("predicted_next_month_spend"),
            "actual_spend":    fv.get("actual_next_month_spend"),
        }
        # Flatten 20 feature keys
        for k in config.ANOMALY_FEATURE_KEYS:
            base[k] = feats.get(k)
        rows.append(base)
    out = pd.DataFrame(rows)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out


def step_3_transform(prod_df: pd.DataFrame, ext_df: pd.DataFrame):
    """Unpack nested feedback + 20-feature dict; derive target + weight."""
    log.info("STEP 3 — Transform")

    if not prod_df.empty:
        prod_df = _flatten_trend_feedback(prod_df)
        # Only keep rows with a valid label
        prod_df = prod_df[prod_df["user_feedback"].notna()]
        prod_df = prod_df[prod_df["user_feedback"] != "ignored"].copy()

        weight_map = {
            "helpful":    config.WEIGHT_HELPFUL,
            "expected":   config.WEIGHT_EXPECTED,
            "not_useful": config.WEIGHT_NOT_USEFUL,
        }
        prod_df["feedback_weight"] = prod_df["user_feedback"].map(weight_map)

        # "expected" → user says spending is normal; use predicted_spend as target
        mask_exp = prod_df["user_feedback"] == "expected"
        # Fall back to predicted_spend when actual hasn't been filled in yet
        actual = prod_df["actual_spend"].fillna(prod_df["predicted_spend"])
        prod_df["training_target"] = actual
        prod_df.loc[mask_exp, "training_target"] = prod_df.loc[mask_exp, "predicted_spend"]

        prod_df["source"] = "production"

    ext_df = ext_df.copy()
    ext_df["feedback_weight"] = config.WEIGHT_EXTERNAL
    ext_df["source"] = "external"
    if "training_target" not in ext_df.columns:
        ext_df["training_target"] = ext_df.get(
            "actual_spend", ext_df.get("current_spend")
        )
    if "timestamp" not in ext_df.columns:
        ext_df["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC")

    log.info("  production labeled: %d | external: %d", len(prod_df), len(ext_df))
    return prod_df, ext_df


# ─────────────────────────────────────────────────────────────────────────────
def step_4_weight_mix(prod_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 4 — Weight / sample mix")

    if prod_df.empty:
        fail("No labeled trend feedback rows — cannot build retraining dataset")
    if ext_df.empty:
        fail("External anomaly training data is empty — check training-data bucket")

    if len(prod_df) >= config.MIN_PRODUCTION_ROWS_FOR_BALANCED_MIX:
        n = min(len(prod_df), len(ext_df))
        prod_sample = prod_df.sample(n=n, random_state=42)
        ext_target = n
        mix_mode = "balanced"
    else:
        prod_sample = prod_df.copy()
        ext_target = min(
            len(ext_df),
            max(config.MIN_TOTAL_ROWS - len(prod_sample), len(prod_sample)),
        )
        mix_mode = "bootstrap"

    ext_sample = ext_df.sample(n=ext_target, random_state=42)

    common = sorted(set(prod_sample.columns) & set(ext_sample.columns))
    mixed = pd.concat(
        [prod_sample[common], ext_sample[common]], ignore_index=True
    )

    ratio = (mixed["source"] == "production").mean()
    log.info("  mix mode: %s | mixed rows: %d | production share: %.2f",
             mix_mode, len(mixed), ratio)
    if (
        mix_mode == "balanced"
        and not (config.MIN_MIX_RATIO <= ratio <= config.MAX_MIX_RATIO)
    ):
        fail(f"Mix ratio {ratio:.2f} outside "
             f"[{config.MIN_MIX_RATIO}, {config.MAX_MIX_RATIO}]")

    if len(mixed) < config.MIN_TOTAL_ROWS:
        log.warning("  dataset below target size: %d rows < %d",
                    len(mixed), config.MIN_TOTAL_ROWS)

    return mixed


# ─────────────────────────────────────────────────────────────────────────────
def step_5_merge_split(mixed: pd.DataFrame):
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
    log.info("STEP 6 — Version + write")

    version = utils.version_folder(now)
    local_dir = config.TMP_DIR / "anomaly" / version
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
        "model_target": "anomaly",
        "lookback_days": config.ANOMALY_LOOKBACK_DAYS,
        "rows": {k: int(len(v)) for k, v in
                 [("train", train), ("val", val), ("test", test)]},
        "source_mix": pd.concat([train, val, test])["source"]
            .value_counts().to_dict(),
        "production_feedback_rows": int((pd.concat([train, val, test])["source"] == "production").sum()),
        "external_rows": int((pd.concat([train, val, test])["source"] == "external").sum()),
        "training_data_hash": data_hash,
        "feature_version": config.ANOMALY_FEATURE_VERSION,
        "feature_keys": config.ANOMALY_FEATURE_KEYS,
        "schema_version": config.SCHEMA_VERSION,
    }
    manifest_path = local_dir / "manifest.json"
    utils.write_json(manifest_path, manifest)

    client = utils.get_minio_client()
    utils.ensure_bucket(client, config.BUCKET_RETRAINING_DATA)

    base_key = f"anomaly/{version}"
    for fname in ["train.parquet", "val.parquet", "test.parquet", "manifest.json"]:
        utils.upload_file(client, config.BUCKET_RETRAINING_DATA,
                          f"{base_key}/{fname}", local_dir / fname)
    utils.update_latest_pointer(client, "anomaly", version, now)

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
    cutoff = (now - timedelta(days=config.ANOMALY_LOOKBACK_DAYS)).isoformat()

    log.info("=" * 70)
    log.info("Trend-detection retraining build — version=%s",
             utils.version_folder(now))
    log.info("Lookback: last %d days (cutoff %s)",
             config.ANOMALY_LOOKBACK_DAYS, cutoff)
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
