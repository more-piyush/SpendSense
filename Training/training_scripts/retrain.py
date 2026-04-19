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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None


# ============================================================
# FEEDBACK COLLECTION
# ============================================================

# Sample weights as defined in documentation (Section 5.1.2)
SAMPLE_WEIGHTS = {
    "user_override": 1.0,    # Strong labels — user changed prediction
    "accepted": 0.3,         # Weak labels — user did not change prediction
    "external": 0.5,         # External CE Survey data
}


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

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"[FEEDBACK] Collected {len(df)} {model_type} feedback records from database")
    return df


def _collect_feedback_from_file(config: dict, model_type: str) -> pd.DataFrame:
    """Load feedback from a file (for testing/offline)."""
    path = config.get("feedback_path")
    if not path or not os.path.exists(path):
        print(f"[FEEDBACK] No feedback file found at {path}")
        return pd.DataFrame()

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"[FEEDBACK] Loaded {len(df)} {model_type} feedback records from {path}")
    return df


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
    schedule = config.get("schedule", "weekly")
    if schedule == "weekly":
        last_retrain = config.get("last_retrain_date")
        if last_retrain:
            days_since = (datetime.utcnow() - datetime.fromisoformat(last_retrain)).days
            if days_since >= 7:
                result["should_retrain"] = True
                result["reasons"].append(f"Scheduled: {days_since} days since last retrain")
        else:
            result["should_retrain"] = True
            result["reasons"].append("No previous retraining recorded")

    # Check override rate threshold
    if not feedback.empty and "feedback_type" in feedback.columns:
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
    schedule = config.get("schedule", "monthly")
    if schedule == "monthly":
        last_retrain = config.get("last_retrain_date")
        if last_retrain:
            days_since = (datetime.utcnow() - datetime.fromisoformat(last_retrain)).days
            if days_since >= 28:
                result["should_retrain"] = True
                result["reasons"].append(f"Scheduled: {days_since} days since last retrain")
        else:
            result["should_retrain"] = True
            result["reasons"].append("No previous retraining recorded")

    # Check feedback count threshold
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
) -> pd.DataFrame:
    """Mix production feedback with external data for categorization retraining.

    Documentation spec:
      - 50% production feedback + 50% external data
      - Sample weights: override=1.0, accepted=0.3, external=0.5
      - Preserves general patterns from external data
    """
    external_path = config.get("external_data_path", "/data/categorization_training.parquet")
    mix_ratio = config.get("production_mix_ratio", 0.5)

    # Load external data
    external = pd.read_parquet(external_path)
    external["sample_weight"] = SAMPLE_WEIGHTS["external"]
    external["data_source"] = "external"

    if feedback.empty:
        print("[DATA MIX] No production feedback, using external data only")
        return external

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

        prod_records.append({
            "description": row.get("description", ""),
            "categories": json.dumps(categories),
            "amount": row.get("amount", 0),
            "currency": "USD",
            "country": "US",
            "sample_weight": weight,
            "data_source": "production",
        })

    prod_df = pd.DataFrame(prod_records)

    # Mix: target 50/50 ratio
    n_prod = len(prod_df)
    n_external = int(n_prod * (1 - mix_ratio) / mix_ratio)
    n_external = min(n_external, len(external))

    external_sample = external.sample(n=n_external, random_state=42)
    external_sample["data_source"] = "external"

    combined = pd.concat([prod_df, external_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add split column (80/10/10)
    n = len(combined)
    combined["split"] = "test"
    combined.loc[:int(n * 0.8), "split"] = "train"
    combined.loc[int(n * 0.8):int(n * 0.9), "split"] = "val"

    print(f"[DATA MIX] Combined: {n_prod} production + {n_external} external = {len(combined)} total")
    print(f"  Production ratio: {n_prod / len(combined):.1%}")
    return combined


def prepare_trend_data(
    feedback: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Prepare trend detection retraining data incorporating feedback.

    For "Expected" feedback: adjust target values in training data.
    For "Helpful": confirm existing predictions.
    For "Not Useful": flag as false positives for reweighting.
    """
    external_path = config.get("external_data_path", "/data/trend_training.parquet")
    external = pd.read_parquet(external_path)

    if feedback.empty:
        print("[DATA MIX] No trend feedback, using external data only")
        return external

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
                if actual:
                    external.loc[mask, "next_month_spend"] = (
                        external.loc[mask, "next_month_spend"] * 0.7 + float(actual) * 0.3
                    )

    print(f"[DATA MIX] Trend data: {len(external)} rows with {len(feedback)} feedback adjustments")
    return external


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
    data_path = config.get("retrain_data_path", "/tmp/retrain_categorization.parquet")
    training_data.to_parquet(data_path, index=False)

    # Build Phase 2 config
    retrain_config = {
        "run_name": f"retrain_categorization_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        "model_type": "distilbert",
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
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print(f"[ERROR] Training failed:\n{result.stderr}")
        raise RuntimeError(f"Categorization retraining failed: {result.stderr}")

    print(result.stdout)
    print(f"[RETRAIN] Categorization retraining complete")

    # Extract run ID from output
    run_id = _extract_run_id(result.stdout, retrain_config)
    return run_id


def retrain_trend(config: dict, training_data: pd.DataFrame) -> str:
    """Execute trend detection model retraining.

    Per-user fine-tuning: 50-100 additional trees, learning_rate=0.01.
    """
    data_path = config.get("retrain_data_path", "/tmp/retrain_trend.parquet")
    training_data.to_parquet(data_path, index=False)

    retrain_config = {
        "run_name": f"retrain_trend_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        "model_type": "xgboost",
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
    }

    config_path = "/tmp/retrain_trend_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(retrain_config, f)

    print(f"\n[RETRAIN] Launching trend detection retraining...")

    result = subprocess.run(
        [sys.executable, "train_trend_detection.py", config_path],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print(f"[ERROR] Training failed:\n{result.stderr}")
        raise RuntimeError(f"Trend retraining failed: {result.stderr}")

    print(result.stdout)
    print(f"[RETRAIN] Trend detection retraining complete")

    run_id = _extract_run_id(result.stdout, retrain_config)
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
        "registry_path": config.get("registry_path", "/data/model_registry"),
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

    ref_df = pd.read_parquet(ref_path)
    current_df = pd.read_parquet(current_path)

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
        training_data = prepare_categorization_data(feedback, config)
    else:
        training_data = prepare_trend_data(feedback, config)

    # Step 4: Execute retraining
    print(f"\n--- Step 4: Retraining model ---")
    if model_type == "DISTILBERT_CATEGORIZATION":
        run_id = retrain_categorization(config, training_data)
    else:
        run_id = retrain_trend(config, training_data)

    # Step 5: Promotion gating
    print(f"\n--- Step 5: Promotion gating ---")
    promotion_result = run_promotion_gating(config, run_id)

    # Step 6: Update last retrain date
    config["last_retrain_date"] = datetime.utcnow().isoformat()
    state_path = config.get("state_file", "/data/retrain_state.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({"last_retrain_date": config["last_retrain_date"],
                    "run_id": run_id, "model_type": model_type}, f)

    print(f"\n{'=' * 70}")
    print(f"RETRAINING COMPLETE")
    print(f"  Run ID: {run_id}")
    print(f"  Triggers: {triggers['reasons']}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Production retraining pipeline")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if no triggers fired")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_retraining_pipeline(config, force=args.force)


if __name__ == "__main__":
    main()
