"""
retrain.py — Production retraining pipeline with feedback integration.

Implements Phase 2 retraining as defined in the documentation:
  - Collects user feedback (overrides, acceptances, anomaly feedback)
  - Mixes production feedback with external data (50/50)
  - Applies sample weighting (overrides=1.0, accepted=0.3, external=0.5)
  - Triggers fine-tuning with appropriate hyperparameters
  - Runs promotion gating after training
  - Supports both scheduled (cron) and threshold-triggered retraining

Usage:
  python retrain.py configs/retrain_categorization.yaml
  python retrain.py configs/retrain_trend.yaml
  python retrain.py configs/retrain_categorization.yaml --force

Retraining Schedule (from documentation):
  DistilBERT: Weekly (Sundays 3 AM) or ad-hoc when override rate > 15%
  XGBoost: Monthly (1st of month, 3 AM) or ad-hoc with 10+ feedback signals
  Isolation Forest: Monthly (alongside XGBoost)
  Global XGBoost baseline: Quarterly
"""

import sys
import os
import json
import argparse
import subprocess
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import boto3

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None

from export_serving_artifacts import materialize_active_serving_artifacts
from utils import (
    load_json_document,
    load_parquet,
    load_active_models,
    set_active_models,
    save_json_document,
    _registry_file_path,
    _registry_storage_config,
    _load_registry_entries,
    _resolve_registry_entry,
    _categorization_score,
    _trend_score,
    get_s3_storage_options,
)


# ============================================================
# FEEDBACK COLLECTION
# ============================================================

# Sample weights as defined in documentation (Section 5.1.2)
SAMPLE_WEIGHTS = {
    "user_override": 1.0,    # Strong labels — user changed prediction
    "accepted": 0.3,         # Weak labels — user did not change prediction
    "external": 0.5,         # External CE Survey data
}

TREND_FEATURE_COLUMNS = [
    "current_spend", "rolling_mean_1m", "rolling_mean_3m", "rolling_mean_6m",
    "rolling_std_3m", "rolling_std_6m", "deviation_ratio", "share_of_wallet",
    "hist_share_of_wallet", "txn_count", "hist_txn_count_mean", "avg_txn_size",
    "hist_avg_txn_size", "days_since_last_txn", "month_of_year",
    "spending_velocity", "weekend_txn_ratio", "total_monthly_spend",
    "elevated_cat_count", "budget_utilization",
]
TREND_TARGET_COLUMN = "next_month_spend"

DEFAULT_DYNAMIC_MIX_TIERS = [
    {"min_user_rows": 0, "production_share": 0.10},
    {"min_user_rows": 25, "production_share": 0.20},
    {"min_user_rows": 50, "production_share": 0.30},
    {"min_user_rows": 100, "production_share": 0.40},
    {"min_user_rows": 200, "production_share": 0.50},
]


def _dynamic_mix_tiers(config: dict) -> list[dict]:
    tiers = config.get("dynamic_mix_tiers") or DEFAULT_DYNAMIC_MIX_TIERS
    return sorted(tiers, key=lambda item: int(item.get("min_user_rows", 0)))


def determine_production_share(config: dict, user_rows: int) -> float:
    selected = DEFAULT_DYNAMIC_MIX_TIERS[0]["production_share"]
    for tier in _dynamic_mix_tiers(config):
        if user_rows >= int(tier.get("min_user_rows", 0)):
            selected = float(tier.get("production_share", selected))
    return min(max(selected, 0.10), 0.50)


def _build_split_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].notna().any():
            df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        else:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = max(1, int(n * 0.8))
    val_end = max(train_end + 1, int(n * 0.9))
    val_end = min(val_end, n)

    df["split"] = "test"
    df.loc[: train_end - 1, "split"] = "train"
    if val_end > train_end:
        df.loc[train_end: val_end - 1, "split"] = "val"
    return df


def _write_retraining_dataset(df: pd.DataFrame, config: dict, task_name: str) -> str:
    if df.empty:
        raise ValueError(f"Cannot persist empty retraining dataset for {task_name}")

    version = datetime.utcnow().strftime("v=%Y-%m-%dT%H-%M-%SZ")
    output_root = config.get("retraining_output_root", "s3://retraining-data").rstrip("/")
    dataset_uri = f"{output_root}/{task_name}/{version}/combined.parquet"
    latest_uri = f"{output_root}/{task_name}/latest.json"

    if not dataset_uri.startswith("s3://"):
        out_path = Path(dataset_uri)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    else:
        storage_options = get_s3_storage_options(config.get("s3", {}))
        df.to_parquet(dataset_uri, index=False, storage_options=storage_options)

    payload = {
        "version": version,
        "path": dataset_uri,
        "rows": int(len(df)),
        "built_at": datetime.utcnow().isoformat(),
        "task": task_name,
        "source_mix": df["data_source"].value_counts().to_dict() if "data_source" in df.columns else {},
        "split_counts": df["split"].value_counts().to_dict() if "split" in df.columns else {},
    }
    save_json_document(latest_uri, payload, config=_registry_storage_config(config))
    print(f"[RETRAIN-DATA] Wrote {task_name} dataset to {dataset_uri}")
    return dataset_uri


def _latest_registry_entry_for_run(registry_path: str, run_id: str, config: dict) -> dict | None:
    registry_file = _registry_file_path(registry_path, "registry.json")
    entries = _load_registry_entries(registry_file, config=_registry_storage_config(config))
    for entry in reversed(entries):
        if entry.get("mlflow_run_id") == run_id:
            return entry
    return None


def _comparison_active_models_filename(config: dict) -> str:
    return config.get("comparison_active_models_file", "active_models.json")


def _target_active_models_filename(config: dict) -> str:
    return config.get("target_active_models_file") or _comparison_active_models_filename(config)


def _promotion_channel(config: dict) -> str:
    if config.get("promotion_channel"):
        return str(config["promotion_channel"])
    return "canary" if _target_active_models_filename(config) != _comparison_active_models_filename(config) else "production"


def _sync_active_models_selection(
    registry_path: str,
    source_file: str,
    target_file: str,
    config: dict,
    tasks: list[str],
) -> dict:
    registry_config = _registry_storage_config(config)
    source = load_active_models(registry_path, active_models_filename=source_file)
    target_path = _registry_file_path(registry_path, target_file)
    target_exists = load_json_document(target_path, default=None, config=registry_config) is not None

    sync_tasks = ["categorization", "trend"] if not target_exists else tasks
    kwargs = {
        "registry_path": registry_path,
        "active_models_filename": target_file,
    }
    if "categorization" in sync_tasks and source.get("categorization"):
        kwargs["active_categorization_registry_id"] = source["categorization"]["registry_id"]
    if "trend" in sync_tasks and source.get("trend"):
        kwargs["active_trend_registry_id"] = source["trend"]["registry_id"]
    return set_active_models(**kwargs)


def _current_active_registry_entry(
    registry_path: str,
    task_type: str,
    config: dict,
    active_models_filename: str,
) -> dict | None:
    active = load_active_models(registry_path, active_models_filename=active_models_filename).get(task_type)
    if not active:
        return None
    registry_file = _registry_file_path(registry_path, "registry.json")
    entries = _load_registry_entries(registry_file, config=_registry_storage_config(config))
    return _resolve_registry_entry(
        entries,
        registry_id=active.get("registry_id"),
        model_id=active.get("model_id"),
    )


def _candidate_is_better(task_type: str, candidate: dict, current_active: dict | None) -> bool:
    if current_active is None:
        return True
    if task_type == "categorization":
        return _categorization_score(candidate) > _categorization_score(current_active)
    return _trend_score(candidate) > _trend_score(current_active)


def compare_and_maybe_activate(config: dict, run_id: str) -> dict:
    registry_path = config.get("registry_path", "s3://mlflow/registry")
    task_type = "categorization" if config["model_type"] == "DISTILBERT_CATEGORIZATION" else "trend"
    comparison_file = _comparison_active_models_filename(config)
    target_file = _target_active_models_filename(config)
    promotion_channel = _promotion_channel(config)

    candidate = _latest_registry_entry_for_run(registry_path, run_id, config)
    if candidate is None:
        raise RuntimeError(f"Could not locate registry entry for run {run_id}")

    active_entry = _current_active_registry_entry(
        registry_path,
        task_type,
        config,
        active_models_filename=comparison_file,
    )
    promoted = _candidate_is_better(task_type, candidate, active_entry)

    if promoted:
        if target_file != comparison_file:
            _sync_active_models_selection(
                registry_path,
                comparison_file,
                target_file,
                config,
                tasks=[task_type],
            )
        if task_type == "categorization":
            updated = set_active_models(
                registry_path=registry_path,
                active_categorization_registry_id=candidate["registry_id"],
                active_models_filename=target_file,
            )
        else:
            updated = set_active_models(
                registry_path=registry_path,
                active_trend_registry_id=candidate["registry_id"],
                active_models_filename=target_file,
            )
        updated = materialize_active_serving_artifacts(
            registry_path,
            config=config,
            active_models_filename=target_file,
        )
        print(
            f"[PROMOTE] Promoted {task_type} model "
            f"{candidate['model_id']} ({candidate['registry_id']}) to {promotion_channel}"
        )
        return {
            "promoted": True,
            "task_type": task_type,
            "promotion_channel": promotion_channel,
            "comparison_active_models_file": comparison_file,
            "target_active_models_file": target_file,
            "requires_rollout": promotion_channel == "canary",
            "active_models": updated,
        }

    if target_file != comparison_file:
        updated = _sync_active_models_selection(
            registry_path,
            comparison_file,
            target_file,
            config,
            tasks=[task_type],
        )
        updated = materialize_active_serving_artifacts(
            registry_path,
            config=config,
            active_models_filename=target_file,
        )
    else:
        updated = None
    print(
        f"[PROMOTE] Kept existing {task_type} active model "
        f"{active_entry.get('model_id') if active_entry else None}; "
        f"candidate {candidate['model_id']} did not beat current metrics."
    )
    return {
        "promoted": False,
        "task_type": task_type,
        "promotion_channel": promotion_channel,
        "comparison_active_models_file": comparison_file,
        "target_active_models_file": target_file,
        "requires_rollout": False,
        "candidate_registry_id": candidate.get("registry_id"),
        "active_registry_id": active_entry.get("registry_id") if active_entry else None,
        "active_models": updated,
    }


def load_retrain_state(config: dict) -> dict:
    state_path = config.get("state_file")
    if not state_path or not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read retrain state from {state_path}: {exc}")
        return {}


def save_retrain_state(config: dict, payload: dict) -> None:
    state_path = config.get("state_file", "/data/retrain_state.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def collect_categorization_feedback(config: dict) -> pd.DataFrame:
    """Collect user feedback for categorization model from Firefly III database.

    Feedback types:
      - User overrides (strong labels, weight=1.0): user changed predicted category
      - Accepted predictions (weak labels, weight=0.3): user kept predicted category
    """
    source = config.get("feedback_source", "database")

    if source == "database":
        return _collect_feedback_from_db(config, "categorization")
    else:
        return _collect_feedback_from_file(config, "categorization")


def collect_trend_feedback(config: dict) -> pd.DataFrame:
    """Collect user feedback for trend detection model.

    Feedback types:
      - "Helpful": confirms anomaly detection was useful
      - "Not Useful": false positive signal
      - "Expected": known spending pattern, adjust baseline
    """
    source = config.get("feedback_source", "database")

    if source == "database":
        return _collect_feedback_from_db(config, "trend")
    else:
        return _collect_feedback_from_file(config, "trend")


def _collect_feedback_from_db(config: dict, model_type: str) -> pd.DataFrame:
    """Collect feedback from PostgreSQL database."""
    import psycopg2
    from psycopg2 import errors

    db = config.get("database", {})
    conn = psycopg2.connect(
        host=db.get("host", "localhost"),
        port=db.get("port", 5432),
        dbname=db.get("dbname", "firefly"),
        user=db.get("user", "firefly"),
        password=db.get("password") or os.environ.get("POSTGRES_PASSWORD", ""),
    )

    # Cutoff: only feedback since last retraining
    last_retrain = config.get("last_retrain_date")
    if last_retrain:
        cutoff = f"AND f.created_at >= '{last_retrain}'"
    else:
        cutoff = ""

    if model_type == "categorization":
        query = f"""
        SELECT
            f.id AS feedback_id,
            f.user_id,
            f.transaction_id,
            t.description,
            t.amount,
            f.predicted_category,
            f.actual_category,
            f.feedback_type,
            f.created_at
        FROM ml_feedback f
        JOIN transactions t ON f.transaction_id = t.id
        WHERE f.model_type = 'categorization'
        {cutoff}
        ORDER BY f.created_at
        """
    else:
        query = f"""
        SELECT
            f.id AS feedback_id,
            f.user_id,
            f.category_name,
            f.anomaly_score,
            f.predicted_spend,
            f.actual_spend,
            f.feedback_type,
            f.created_at
        FROM ml_feedback f
        WHERE f.model_type = 'trend'
        {cutoff}
        ORDER BY f.created_at
        """

    try:
        df = pd.read_sql(query, conn)
    except errors.UndefinedTable:
        print("[FEEDBACK] Table 'ml_feedback' does not exist yet, treating feedback as empty")
        df = pd.DataFrame()
    finally:
        conn.close()

    print(f"[FEEDBACK] Collected {len(df)} {model_type} feedback records from database")
    return df


def _collect_feedback_from_file(config: dict, model_type: str) -> pd.DataFrame:
    """Load feedback from a file (for testing/offline)."""
    path = config.get("feedback_path")
    if not path:
        print(f"[FEEDBACK] No feedback file configured")
        return pd.DataFrame()

    if path.startswith("s3://"):
        records = _load_feedback_records_from_s3(path, config, model_type)
        df = pd.DataFrame(records)
        if df.empty:
            print(f"[FEEDBACK] Loaded 0 {model_type} feedback records from {path}")
            return df
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            cutoff = _feedback_cutoff_timestamp(config)
            if cutoff is not None:
                df = df[df["created_at"] >= cutoff].reset_index(drop=True)
        print(f"[FEEDBACK] Loaded {len(df)} {model_type} feedback records from {path}")
        return df

    if not os.path.exists(path):
        print(f"[FEEDBACK] No feedback file found at {path}")
        return pd.DataFrame()

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"[FEEDBACK] Loaded {len(df)} {model_type} feedback records from {path}")
    return df


def _feedback_cutoff_timestamp(config: dict) -> pd.Timestamp | None:
    if config.get("last_retrain_date"):
        return pd.to_datetime(config["last_retrain_date"], utc=True, errors="coerce")
    lookback_days = config.get("lookback_days")
    if lookback_days is None:
        return None
    return pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    bucket_and_key = uri[5:]
    bucket, key = bucket_and_key.split("/", 1)
    return bucket, key


def _extract_primary_category(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "category" in value:
            return value["category"]
        preds = value.get("predicted_categories")
        if isinstance(preds, list) and preds:
            first = preds[0]
            if isinstance(first, dict):
                return first.get("category")
            return first
    return None


def _feedback_action_to_type(action: str, model_type: str) -> str:
    action = (action or "").lower()
    if model_type == "categorization":
        if action == "overridden":
            return "override"
        if action == "accepted":
            return "accepted"
        return action or "accepted"

    if action in {"confirmed", "accepted"}:
        return "helpful"
    if action in {"rejected", "dismissed"}:
        return "not_useful"
    if action == "overridden":
        return "expected"
    return action or "helpful"


def _normalize_feedback_event(record: dict, model_type: str) -> dict | None:
    payload = record.get("feedback", record)
    task = payload.get("task")
    if model_type == "categorization" and task != "categorization":
        return None
    if model_type == "trend" and task not in {"trend", "trend_detection"}:
        return None

    action = payload.get("action", "")
    metadata = payload.get("metadata", {}) or {}
    predicted_value = payload.get("predicted_value", {}) or {}
    final_value = payload.get("final_value", {}) or {}

    if model_type == "categorization":
        return {
            "feedback_id": record.get("event_id"),
            "user_id": payload.get("user_id"),
            "transaction_id": payload.get("transaction_id"),
            "description": metadata.get("description", ""),
            "amount": metadata.get("amount", 0),
            "currency": metadata.get("currency"),
            "country": metadata.get("country"),
            "predicted_category": _extract_primary_category(predicted_value),
            "actual_category": _extract_primary_category(final_value)
            or _extract_primary_category(predicted_value),
            "feedback_type": _feedback_action_to_type(action, model_type),
            "created_at": payload.get("timestamp") or record.get("recorded_at"),
        }

    return {
        "feedback_id": record.get("event_id"),
        "user_id": payload.get("user_id"),
        "category_name": metadata.get("category") or final_value.get("category") or predicted_value.get("category"),
        "anomaly_score": predicted_value.get("ensemble_score"),
        "predicted_spend": predicted_value.get("predicted_next_month_spend"),
        "actual_spend": final_value.get("actual_next_month_spend") or final_value.get("actual_spend"),
        "feedback_type": _feedback_action_to_type(action, model_type),
        "created_at": payload.get("timestamp") or record.get("recorded_at"),
        "period": payload.get("period") or payload.get("timestamp"),
        "features": payload.get("features") or {},
    }


def _load_feedback_records_from_s3(path: str, config: dict, model_type: str) -> list[dict]:
    bucket, prefix = _parse_s3_uri(path.rstrip("/"))
    s3_cfg = config.get("s3", {})
    client = boto3.client(
        "s3",
        endpoint_url=s3_cfg.get("endpoint_url") or os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=s3_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=s3_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=s3_cfg.get("region", "us-east-1"),
    )

    paginator = client.get_paginator("list_objects_v2")
    rows = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            raw = json.loads(body.decode("utf-8"))
            normalized = _normalize_feedback_event(raw, model_type)
            if normalized is not None:
                rows.append(normalized)
    return rows


# ============================================================
# RETRAINING TRIGGER CHECKS
# ============================================================
def check_categorization_triggers(config: dict, feedback: pd.DataFrame) -> dict:
    """Check if categorization retraining should be triggered.

    Triggers:
      - Scheduled: weekly (Sundays 3 AM)
      - Ad-hoc: override rate > 15%
    """
    result = {"should_retrain": False, "reasons": []}

    # Check schedule
    schedule_days = int(config.get("schedule_days", 7))
    last_retrain = config.get("last_retrain_date")
    if last_retrain:
        days_since = (datetime.utcnow() - datetime.fromisoformat(last_retrain)).days
        if days_since >= schedule_days:
            result["should_retrain"] = True
            result["reasons"].append(
                f"Scheduled: {days_since} days since last retrain (target={schedule_days})"
            )
    else:
        result["should_retrain"] = True
        result["reasons"].append("No previous retraining recorded")

    # Check override rate threshold
    if config.get("feedback_trigger_enabled", True) and not feedback.empty and "feedback_type" in feedback.columns:
        overrides = (feedback["feedback_type"] == "override").sum()
        total = len(feedback)
        override_rate = overrides / max(total, 1)
        threshold = config.get("override_rate_threshold", 0.15)

        if override_rate > threshold:
            result["should_retrain"] = True
            result["reasons"].append(
                f"Override rate {override_rate:.1%} > threshold {threshold:.1%}"
            )

    print(f"[TRIGGER] Categorization: retrain={result['should_retrain']}, "
          f"reasons={result['reasons']}")
    return result


def check_trend_triggers(config: dict, feedback: pd.DataFrame) -> dict:
    """Check if trend detection retraining should be triggered.

    Triggers:
      - Scheduled: monthly (1st of month, 3 AM)
      - Ad-hoc: 10+ feedback signals
    """
    result = {"should_retrain": False, "reasons": []}

    # Check schedule
    schedule_days = int(config.get("schedule_days", 28))
    last_retrain = config.get("last_retrain_date")
    if last_retrain:
        days_since = (datetime.utcnow() - datetime.fromisoformat(last_retrain)).days
        if days_since >= schedule_days:
            result["should_retrain"] = True
            result["reasons"].append(
                f"Scheduled: {days_since} days since last retrain (target={schedule_days})"
            )
    else:
        result["should_retrain"] = True
        result["reasons"].append("No previous retraining recorded")

    # Check feedback count threshold
    if config.get("feedback_trigger_enabled", True):
        min_feedback = config.get("min_feedback_signals", 10)
        if len(feedback) >= min_feedback:
            result["should_retrain"] = True
            result["reasons"].append(
                f"{len(feedback)} feedback signals >= threshold {min_feedback}"
            )

    print(f"[TRIGGER] Trend: retrain={result['should_retrain']}, "
          f"reasons={result['reasons']}")
    return result


# ============================================================
# DATA MIXING
# ============================================================
def prepare_categorization_data(
    feedback: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Mix production feedback with external data for categorization retraining.

    Documentation spec:
      - 50% production feedback + 50% external data
      - Sample weights: override=1.0, accepted=0.3, external=0.5
      - Preserves general patterns from external data
    """
    external_path = config.get("external_data_path", "/data/categorization_training.parquet")
    external = load_parquet(external_path, config)
    if "categories" in external.columns:
        external["categories"] = external["categories"].apply(
            lambda value: json.dumps(value) if isinstance(value, list) else value
        )
    if "amount" in external.columns:
        external["amount"] = pd.to_numeric(external["amount"], errors="coerce")
    if "timestamp" not in external.columns:
        external["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC")
    external["sample_weight"] = SAMPLE_WEIGHTS["external"]
    external["data_source"] = "external"

    if feedback.empty:
        print("[DATA MIX] No production feedback, skipping categorization retraining dataset build")
        return pd.DataFrame(), {
            "production_share": 0.0,
            "external_share": 1.0,
            "production_rows": 0,
            "external_rows": len(external),
        }

    # Prepare production feedback
    prod_records = []
    for _, row in feedback.iterrows():
        ftype = row.get("feedback_type", "accepted")

        if ftype == "override":
            # User changed the category — strong label
            categories = [row["actual_category"]]
            weight = SAMPLE_WEIGHTS["user_override"]
        else:
            # User accepted prediction — weak label
            categories = [row.get("predicted_category", row.get("actual_category"))]
            weight = SAMPLE_WEIGHTS["accepted"]
        categories = [category for category in categories if category]
        if not categories:
            continue

        prod_records.append(
            {
                "description": row.get("description", ""),
                "categories": json.dumps(categories),
                "amount": row.get("amount", 0),
                "currency": row.get("currency") or "USD",
                "country": row.get("country") or "US",
                "sample_weight": weight,
                "data_source": "production",
                "timestamp": row.get("created_at"),
            }
        )

    prod_df = pd.DataFrame(prod_records)
    if "amount" in prod_df.columns:
        prod_df["amount"] = pd.to_numeric(prod_df["amount"], errors="coerce")
    prod_df["timestamp"] = pd.to_datetime(prod_df["timestamp"], utc=True, errors="coerce")

    n_prod = len(prod_df)
    production_share = determine_production_share(config, n_prod)
    n_external = math.ceil(n_prod * (1 - production_share) / production_share)
    n_external = min(n_external, len(external))

    external_sample = external.sample(n=n_external, random_state=42)
    external_sample["data_source"] = "external"

    required_columns = [
        "description",
        "categories",
        "amount",
        "currency",
        "country",
        "sample_weight",
        "data_source",
        "timestamp",
    ]
    combined = pd.concat(
        [prod_df.reindex(columns=required_columns), external_sample.reindex(columns=required_columns)],
        ignore_index=True,
    )
    combined = _build_split_column(combined)

    print(f"[DATA MIX] Combined: {n_prod} production + {n_external} external = {len(combined)} total")
    print(f"  Production ratio: {n_prod / len(combined):.1%}")
    return combined, {
        "production_share": production_share,
        "external_share": 1 - production_share,
        "production_rows": n_prod,
        "external_rows": n_external,
    }


def prepare_trend_data(
    feedback: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Prepare trend detection retraining data incorporating feedback.

    For "Expected" feedback: adjust target values in training data.
    For "Helpful": confirm existing predictions.
    For "Not Useful": flag as false positives for reweighting.
    """
    external_path = config.get("external_data_path", "/data/trend_training.parquet")
    external = load_parquet(external_path, config)
    external["sample_weight"] = SAMPLE_WEIGHTS["external"]
    external["data_source"] = "external"
    if "timestamp" not in external.columns:
        external["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC")
    if TREND_TARGET_COLUMN not in external.columns and "training_target" in external.columns:
        external[TREND_TARGET_COLUMN] = external["training_target"]

    if feedback.empty:
        print("[DATA MIX] No trend feedback, skipping trend retraining dataset build")
        return pd.DataFrame(), {
            "production_share": 0.0,
            "external_share": 1.0,
            "production_rows": 0,
            "external_rows": len(external),
        }

    # Process feedback labels
    for _, row in feedback.iterrows():
        ftype = row.get("feedback_type", "")
        user_id = row.get("user_id")
        category = row.get("category_name")

        if ftype == "expected":
            # User says this spending is expected — adjust baseline
            # Find matching rows and update target toward actual spend
            mask = (
                (external["user_id"] == user_id) &
                (external["category"] == category)
            ) if "user_id" in external.columns else pd.Series([False] * len(external))

            if mask.any():
                actual = row.get("actual_spend", row.get("predicted_spend"))
                if pd.notna(actual):
                    external.loc[mask, "next_month_spend"] = (
                        external.loc[mask, "next_month_spend"] * 0.7 + float(actual) * 0.3
                    )

    prod_records = []
    for _, row in feedback.iterrows():
        feature_payload = row.get("features") or {}
        if not feature_payload:
            continue

        actual_spend = row.get("actual_spend")
        predicted_spend = row.get("predicted_spend")
        feedback_type = row.get("feedback_type")
        if feedback_type == "expected":
            target_value = predicted_spend
        else:
            target_value = actual_spend if pd.notna(actual_spend) else predicted_spend
        if pd.isna(target_value):
            continue

        record = {
            "timestamp": row.get("created_at"),
            "period": row.get("period"),
            "category": row.get("category_name"),
            TREND_TARGET_COLUMN: target_value,
            "sample_weight": {
                "expected": SAMPLE_WEIGHTS["accepted"],
                "helpful": SAMPLE_WEIGHTS["user_override"],
                "not_useful": SAMPLE_WEIGHTS["user_override"],
            }.get(feedback_type, SAMPLE_WEIGHTS["accepted"]),
            "data_source": "production",
        }
        for feature_name in TREND_FEATURE_COLUMNS:
            record[feature_name] = feature_payload.get(feature_name)
        prod_records.append(record)

    prod_df = pd.DataFrame(prod_records)
    if prod_df.empty:
        print("[DATA MIX] Trend feedback did not contain feature payloads, skipping retraining dataset build")
        return pd.DataFrame(), {
            "production_share": 0.0,
            "external_share": 1.0,
            "production_rows": 0,
            "external_rows": len(external),
        }

    prod_df["timestamp"] = pd.to_datetime(prod_df["timestamp"], utc=True, errors="coerce")

    production_share = determine_production_share(config, len(prod_df))
    n_external = math.ceil(len(prod_df) * (1 - production_share) / production_share)
    n_external = min(n_external, len(external))
    external_sample = external.sample(n=n_external, random_state=42)

    required_columns = TREND_FEATURE_COLUMNS + [
        TREND_TARGET_COLUMN,
        "period",
        "category",
        "sample_weight",
        "data_source",
        "timestamp",
    ]
    combined = pd.concat(
        [prod_df.reindex(columns=required_columns), external_sample.reindex(columns=required_columns)],
        ignore_index=True,
    )
    combined = _build_split_column(combined)
    print(f"[DATA MIX] Trend data: {len(prod_df)} production + {n_external} external = {len(combined)} rows")
    return combined, {
        "production_share": production_share,
        "external_share": 1 - production_share,
        "production_rows": len(prod_df),
        "external_rows": n_external,
    }


# ============================================================
# RETRAINING EXECUTION
# ============================================================
def retrain_categorization(config: dict, training_data: pd.DataFrame) -> str:
    """Execute categorization model retraining (Phase 2 fine-tuning).

    Phase 2 settings from documentation:
      - Learning rate: 5e-6 (10x lower than Phase 1)
      - Data mixing: 50% production + 50% external
      - Layer freezing: layers 0-3 frozen, layers 4-5 + head trainable
      - Sample weighting: override=1.0, accepted=0.3, external=0.5
    """
    # Save mixed training data to temp location
    data_path = config.get("retrain_data_path")
    if not data_path:
        data_path = "/tmp/retrain_categorization.parquet"
        training_data.to_parquet(data_path, index=False)

    # Build Phase 2 config
    retrain_config = {
        "run_name": f"retrain_categorization_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        "model_type": "distilbert",
        "task_type": config.get("task_type", "categorization"),
        "model_id": config.get("model_id", "cat_distilbert_retrain"),
        "model_family": config.get("model_family", "distilbert"),
        "training_mode": config.get("training_mode", "retraining"),
        "initial_status": config.get("initial_status", "CANDIDATE"),
        "pretrained_model": config.get("base_model", "distilbert-base-uncased"),
        "max_length": 64,
        "learning_rate": config.get("phase2_learning_rate", 5e-6),
        "batch_size": config.get("batch_size", 32),
        "epochs": config.get("epochs", 3),
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "dropout": 0.3,
        "freeze_layers": config.get("freeze_layers", 3),
        "early_stopping_patience": 2,
        "mixed_precision": True,
        "data_path": data_path,
        "mlflow_tracking_uri": config.get("mlflow_tracking_uri", "http://localhost:5000"),
        "experiment_name": config.get("experiment_name", "categorization_retrain"),
        "direct_artifact_experiment_name": config.get("direct_artifact_experiment_name"),
        "artifact_location": config.get("artifact_location"),
        "registry_path": config.get("registry_path", "s3://mlflow/registry"),
        "s3": config.get("s3", {}),
    }

    # Load previous model weights if specified
    prev_model_path = config.get("previous_model_path")
    if prev_model_path:
        retrain_config["pretrained_weights"] = prev_model_path

    # Save config to temp file
    config_path = "/tmp/retrain_cat_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(retrain_config, f)

    # Execute training script
    print(f"\n[RETRAIN] Launching categorization retraining...")
    print(f"  Config: {json.dumps(retrain_config, indent=2, default=str)}")

    result = subprocess.run(
        [sys.executable, "train_categorization.py", config_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Categorization retraining failed with exit code {result.returncode}"
        )

    print(f"[RETRAIN] Categorization retraining complete")

    # Extract run ID from output
    run_id = _extract_run_id("", retrain_config)
    return run_id


def retrain_trend(config: dict, training_data: pd.DataFrame) -> str:
    """Execute trend detection model retraining.

    Per-user fine-tuning: 50-100 additional trees, learning_rate=0.01.
    """
    data_path = config.get("retrain_data_path")
    if not data_path:
        data_path = "/tmp/retrain_trend.parquet"
        training_data.to_parquet(data_path, index=False)

    retrain_config = {
        "run_name": f"retrain_trend_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        "model_type": "xgboost",
        "task_type": config.get("task_type", "trend"),
        "model_id": config.get("model_id", "trend_xgb_retrain"),
        "model_family": config.get("model_family", "xgboost"),
        "training_mode": config.get("training_mode", "retraining"),
        "initial_status": config.get("initial_status", "CANDIDATE"),
        "objective": "reg:squarederror",
        "n_estimators": config.get("n_estimators", 100),
        "max_depth": config.get("max_depth", 6),
        "learning_rate": config.get("phase2_learning_rate", 0.01),
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": config.get("reg_alpha", 0.2),
        "reg_lambda": config.get("reg_lambda", 2.0),
        "gamma": 0.1,
        "verbose": 10,
        "train_isolation_forest": True,
        "iso_n_estimators": 100,
        "iso_contamination": 0.1,
        "iso_max_samples": 256,
        "xgb_ensemble_weight": 0.6,
        "data_path": data_path,
        "mlflow_tracking_uri": config.get("mlflow_tracking_uri", "http://localhost:5000"),
        "experiment_name": config.get("experiment_name", "trend_retrain"),
        "direct_artifact_experiment_name": config.get("direct_artifact_experiment_name"),
        "artifact_location": config.get("artifact_location"),
        "registry_path": config.get("registry_path", "s3://mlflow/registry"),
        "s3": config.get("s3", {}),
    }

    config_path = "/tmp/retrain_trend_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(retrain_config, f)

    print(f"\n[RETRAIN] Launching trend detection retraining...")

    result = subprocess.run(
        [sys.executable, "train_trend_detection.py", config_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Trend retraining failed with exit code {result.returncode}"
        )

    print(f"[RETRAIN] Trend detection retraining complete")

    run_id = _extract_run_id("", retrain_config)
    return run_id


def _extract_run_id(output: str, config: dict) -> str:
    """Try to extract the MLflow run ID from training output."""
    # Try to get the latest run from MLflow
    if mlflow is None:
        return "unknown"

    try:
        tracking_uri = config.get("mlflow_tracking_uri", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(config.get("experiment_name", "default"))
        if experiment:
            runs = client.search_runs(
                experiment.experiment_id,
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0].info.run_id
    except Exception as e:
        print(f"[WARN] Could not extract run ID: {e}")

    return "unknown"


# ============================================================
# PROMOTION GATING (POST-RETRAIN)
# ============================================================
def run_promotion_gating(config: dict, run_id: str) -> dict:
    """Run promotion gating after retraining."""
    promote_config = {
        "model_type": config["model_type"],
        "mlflow_tracking_uri": config.get("mlflow_tracking_uri", "http://localhost:5000"),
        "registry_path": config.get("registry_path", "s3://mlflow/registry"),
    }

    config_path = "/tmp/promote_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(promote_config, f)

    print(f"\n[PROMOTE] Running promotion gating for run {run_id}...")

    result = subprocess.run(
        [sys.executable, "promote_model.py", config_path,
         "--run-id", run_id, "--action", "promote"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"[WARN] Promotion gating failed:\n{result.stderr}")

    return {"run_id": run_id, "promotion_output": result.stdout}


# ============================================================
# DRIFT DETECTION
# ============================================================
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI > 0.2 triggers alert; > 0.25 triggers immediate retraining.
    """
    # Bin the distributions
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Normalize to proportions
    expected_pct = (expected_counts + 1e-8) / (expected_counts.sum() + bins * 1e-8)
    actual_pct = (actual_counts + 1e-8) / (actual_counts.sum() + bins * 1e-8)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def check_data_drift(config: dict) -> dict:
    """Check for data drift using PSI on input features.

    Computed weekly per input feature.
    PSI > 0.2 triggers alert; > 0.25 triggers immediate retraining.
    """
    from compute_features import FEATURE_COLUMNS

    # Load reference (training) and current feature distributions
    ref_path = config.get("reference_features_path")
    current_path = config.get("current_features_path")

    if not ref_path or not current_path:
        print("[DRIFT] No reference/current feature paths configured, skipping")
        return {"drift_detected": False}

    try:
        ref_df = load_parquet(ref_path, config)
        current_df = load_parquet(current_path, config)
    except FileNotFoundError:
        print("[DRIFT] Reference/current feature files not found, skipping")
        return {"drift_detected": False}
    except Exception as e:
        print(f"[DRIFT] Could not load drift feature files, skipping: {e}")
        return {"drift_detected": False}

    results = {}
    alert_features = []
    retrain_features = []

    for feature in FEATURE_COLUMNS:
        if feature not in ref_df.columns or feature not in current_df.columns:
            continue

        ref_vals = ref_df[feature].dropna().values
        cur_vals = current_df[feature].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        results[feature] = {
            "psi": round(psi, 4),
            "alert": psi > 0.2,
            "retrain": psi > 0.25,
        }

        if psi > 0.25:
            retrain_features.append(feature)
        elif psi > 0.2:
            alert_features.append(feature)

    drift_detected = len(retrain_features) > 0

    print(f"[DRIFT] PSI results:")
    for feat, r in sorted(results.items(), key=lambda x: -x[1]["psi"]):
        status = "RETRAIN" if r["retrain"] else ("ALERT" if r["alert"] else "OK")
        print(f"  [{status}] {feat}: PSI={r['psi']:.4f}")

    if alert_features:
        print(f"[DRIFT] Alert features: {alert_features}")
    if retrain_features:
        print(f"[DRIFT] Retrain features: {retrain_features}")

    return {
        "drift_detected": drift_detected,
        "alert_features": alert_features,
        "retrain_features": retrain_features,
        "psi_results": results,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_retraining_pipeline(config: dict, force: bool = False):
    """Execute the full retraining pipeline."""
    model_type = config["model_type"]

    print(f"\n{'=' * 70}")
    print(f"RETRAINING PIPELINE: {model_type}")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print(f"{'=' * 70}")

    # Step 1: Collect feedback
    print(f"\n--- Step 1: Collecting feedback ---")
    if model_type == "DISTILBERT_CATEGORIZATION":
        feedback = collect_categorization_feedback(config)
    else:
        feedback = collect_trend_feedback(config)

    # Step 2: Check triggers
    print(f"\n--- Step 2: Checking retraining triggers ---")
    if model_type == "DISTILBERT_CATEGORIZATION":
        triggers = check_categorization_triggers(config, feedback)
    else:
        triggers = check_trend_triggers(config, feedback)

    # Step 2b: Check data drift
    drift_result = check_data_drift(config)
    if drift_result.get("drift_detected"):
        triggers["should_retrain"] = True
        triggers["reasons"].append("Data drift detected (PSI > 0.25)")

    if not triggers["should_retrain"] and not force:
        print(f"\n[SKIP] No retraining triggers fired. Use --force to override.")
        return

    # Step 3: Prepare training data
    print(f"\n--- Step 3: Preparing training data ---")
    if model_type == "DISTILBERT_CATEGORIZATION":
        training_data, mix_stats = prepare_categorization_data(feedback, config)
        task_name = "categorization"
    else:
        training_data, mix_stats = prepare_trend_data(feedback, config)
        task_name = "trend"

    if training_data.empty:
        print("\n[SKIP] Retraining dataset is empty after mixing. Active models unchanged.")
        return

    dataset_uri = _write_retraining_dataset(training_data, config, task_name)
    config["retrain_data_path"] = dataset_uri

    # Step 4: Execute retraining
    print(f"\n--- Step 4: Retraining model ---")
    if model_type == "DISTILBERT_CATEGORIZATION":
        run_id = retrain_categorization(config, training_data)
    else:
        run_id = retrain_trend(config, training_data)

    # Step 5: Compare against current active model and optionally promote
    print(f"\n--- Step 5: Comparative promotion analysis ---")
    promotion_result = compare_and_maybe_activate(config, run_id)
    if promotion_result.get("promoted") and promotion_result.get("promotion_channel") == "canary":
        print(
            "[PROMOTE] Candidate is now staged in canary configuration. "
            "Run the serving canary rollout stages before promoting to production."
        )

    # Step 6: Update last retrain date
    config["last_retrain_date"] = datetime.utcnow().isoformat()
    save_retrain_state(
        config,
        {
            "last_retrain_date": config["last_retrain_date"],
            "run_id": run_id,
            "model_type": model_type,
            "task_name": task_name,
            "dataset_uri": dataset_uri,
            "mix_stats": mix_stats,
            "promotion_result": promotion_result,
        },
    )

    print(f"\n{'=' * 70}")
    print(f"RETRAINING COMPLETE")
    print(f"  Run ID: {run_id}")
    print(f"  Triggers: {triggers['reasons']}")
    print(f"  Retraining dataset: {dataset_uri}")
    print(f"  Promotion result: {json.dumps(promotion_result, indent=2, default=str)}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Production retraining pipeline")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if no triggers fired")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    state = load_retrain_state(config)
    if state.get("last_retrain_date"):
        config["last_retrain_date"] = state["last_retrain_date"]

    run_retraining_pipeline(config, force=args.force)


if __name__ == "__main__":
    main()
