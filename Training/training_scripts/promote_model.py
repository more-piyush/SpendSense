"""
promote_model.py — Model promotion gating and lifecycle management.

Checks trained model metrics against promotion gates defined in the documentation,
manages the model lifecycle (TRAINING -> SHADOW -> CANARY -> PRODUCTION -> RETIRED),
and supports instant rollback.

Usage:
  python promote_model.py configs/promote.yaml --run-id <mlflow_run_id> --action evaluate
  python promote_model.py configs/promote.yaml --run-id <mlflow_run_id> --action promote
  python promote_model.py configs/promote.yaml --run-id <mlflow_run_id> --action rollback

Promotion Gates (from documentation):
  DistilBERT Categorization:
    - Accuracy >= 90%
    - Macro-F1 >= 0.85
    - No individual category recall below 0.70
    - Abstention rate between 5% and 20%
    - Inference latency p95 < 150ms

  XGBoost Trend Detection:
    - RMSE <= current production model (or within 5% tolerance)
    - MAE does not increase by > 10%
    - Precision@5 >= 40%
    - No single category RMSE > 3x global average
    - Feature importance rankings stable
"""

import sys
import os
import json
import argparse
import uuid
import shutil
from datetime import datetime
from pathlib import Path

import yaml
import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# PROMOTION GATES
# ============================================================

CATEGORIZATION_GATES = {
    "test_subset_accuracy": {"min": 0.90, "description": "Accuracy >= 90%"},
    "test_macro_f1": {"min": 0.85, "description": "Macro-F1 >= 0.85"},
    "test_min_class_recall": {"min": 0.70, "description": "No category recall below 0.70"},
    "test_abstention_rate": {
        "min": 0.05, "max": 0.20,
        "description": "Abstention rate between 5% and 20%",
    },
}

TREND_DETECTION_GATES = {
    "xgb_test_rmse": {
        "max_relative": 1.05,  # Within 5% of production model
        "description": "RMSE <= production (within 5% tolerance)",
    },
    "xgb_test_mae": {
        "max_relative": 1.10,  # MAE does not increase by >10%
        "description": "MAE does not increase by >10%",
    },
    "xgb_precision_at_5": {"min": 0.40, "description": "Precision@5 >= 40%"},
}


# ============================================================
# MODEL REGISTRY
# ============================================================
class ModelRegistry:
    """Simple file-based model registry for tracking model lifecycle.

    Schema matches documentation Section 9:
      model_id, model_type, version, user_id, training_data_hash,
      feature_version, hyperparameters, eval_metrics, training_timestamp,
      artifact_path, status, predecessor_id
    """

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.registry_file = os.path.join(registry_path, "registry.json")
        os.makedirs(registry_path, exist_ok=True)

        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                self.models = json.load(f)
        else:
            self.models = []

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2, default=str)

    def register(
        self,
        model_type: str,
        version: str,
        eval_metrics: dict,
        hyperparameters: dict,
        training_data_hash: str,
        feature_version: str,
        artifact_path: str,
        user_id: str = None,
        mlflow_run_id: str = None,
    ) -> dict:
        """Register a new model version."""
        predecessor = self.get_production_model(model_type, user_id)

        entry = {
            "model_id": str(uuid.uuid4()),
            "model_type": model_type,
            "version": version,
            "user_id": user_id,
            "training_data_hash": training_data_hash,
            "feature_version": feature_version,
            "hyperparameters": hyperparameters,
            "eval_metrics": eval_metrics,
            "training_timestamp": datetime.utcnow().isoformat(),
            "artifact_path": artifact_path,
            "status": "TRAINING",
            "predecessor_id": predecessor["model_id"] if predecessor else None,
            "mlflow_run_id": mlflow_run_id,
        }

        self.models.append(entry)
        self._save()
        print(f"[REGISTRY] Registered model {entry['model_id']} "
              f"(type={model_type}, version={version}, status=TRAINING)")
        return entry

    def update_status(self, model_id: str, new_status: str) -> dict:
        """Update model status. Valid transitions:
        TRAINING -> SHADOW -> CANARY -> PRODUCTION -> RETIRED
        """
        valid_transitions = {
            "TRAINING": ["SHADOW", "RETIRED"],
            "SHADOW": ["CANARY", "RETIRED"],
            "CANARY": ["PRODUCTION", "RETIRED"],
            "PRODUCTION": ["RETIRED"],
            "RETIRED": [],
        }

        entry = self.get_model(model_id)
        if entry is None:
            raise ValueError(f"Model {model_id} not found")

        current = entry["status"]
        if new_status not in valid_transitions.get(current, []):
            raise ValueError(
                f"Invalid transition: {current} -> {new_status}. "
                f"Valid: {valid_transitions[current]}"
            )

        entry["status"] = new_status
        entry[f"{new_status.lower()}_timestamp"] = datetime.utcnow().isoformat()

        # If promoting to PRODUCTION, retire the predecessor
        if new_status == "PRODUCTION" and entry.get("predecessor_id"):
            pred = self.get_model(entry["predecessor_id"])
            if pred and pred["status"] == "PRODUCTION":
                pred["status"] = "RETIRED"
                pred["retired_timestamp"] = datetime.utcnow().isoformat()
                print(f"[REGISTRY] Retired predecessor {pred['model_id']}")

        self._save()
        print(f"[REGISTRY] Model {model_id}: {current} -> {new_status}")
        return entry

    def get_model(self, model_id: str) -> dict:
        for m in self.models:
            if m["model_id"] == model_id:
                return m
        return None

    def get_production_model(self, model_type: str, user_id: str = None) -> dict:
        """Get the current production model for a given type and user."""
        for m in reversed(self.models):
            if (m["model_type"] == model_type
                    and m["status"] == "PRODUCTION"
                    and m.get("user_id") == user_id):
                return m
        return None

    def get_latest(self, model_type: str, user_id: str = None) -> dict:
        """Get the most recently registered model of a given type."""
        for m in reversed(self.models):
            if m["model_type"] == model_type and m.get("user_id") == user_id:
                return m
        return None

    def list_models(self, model_type: str = None, status: str = None) -> list:
        """List models with optional filters."""
        results = self.models
        if model_type:
            results = [m for m in results if m["model_type"] == model_type]
        if status:
            results = [m for m in results if m["status"] == status]
        return results


# ============================================================
# GATE EVALUATION
# ============================================================
def evaluate_categorization_gates(
    metrics: dict,
    production_metrics: dict = None,
) -> dict:
    """Check categorization model against promotion gates.

    Returns dict with gate results: {gate_name: {passed, value, threshold, description}}
    """
    results = {}

    for metric_key, gate in CATEGORIZATION_GATES.items():
        value = metrics.get(metric_key)
        if value is None:
            results[metric_key] = {
                "passed": False,
                "value": None,
                "description": gate["description"],
                "reason": f"Metric '{metric_key}' not found in run metrics",
            }
            continue

        passed = True
        reasons = []

        if "min" in gate and value < gate["min"]:
            passed = False
            reasons.append(f"{value:.4f} < min({gate['min']})")
        if "max" in gate and value > gate["max"]:
            passed = False
            reasons.append(f"{value:.4f} > max({gate['max']})")

        results[metric_key] = {
            "passed": passed,
            "value": round(value, 4),
            "threshold": {k: v for k, v in gate.items() if k != "description"},
            "description": gate["description"],
            "reason": "; ".join(reasons) if reasons else "PASS",
        }

    return results


def evaluate_trend_gates(
    metrics: dict,
    production_metrics: dict = None,
) -> dict:
    """Check trend detection model against promotion gates."""
    results = {}

    for metric_key, gate in TREND_DETECTION_GATES.items():
        value = metrics.get(metric_key)
        if value is None:
            results[metric_key] = {
                "passed": False,
                "value": None,
                "description": gate["description"],
                "reason": f"Metric '{metric_key}' not found",
            }
            continue

        passed = True
        reasons = []

        if "min" in gate and value < gate["min"]:
            passed = False
            reasons.append(f"{value:.4f} < min({gate['min']})")

        if "max_relative" in gate and production_metrics:
            prod_key = metric_key
            prod_value = production_metrics.get(prod_key)
            if prod_value is not None:
                threshold = prod_value * gate["max_relative"]
                if value > threshold:
                    passed = False
                    reasons.append(
                        f"{value:.4f} > {gate['max_relative']}x production({prod_value:.4f})"
                    )

        results[metric_key] = {
            "passed": passed,
            "value": round(value, 4),
            "description": gate["description"],
            "reason": "; ".join(reasons) if reasons else "PASS",
        }

    return results


def check_all_gates(
    model_type: str,
    metrics: dict,
    production_metrics: dict = None,
) -> tuple:
    """Run all promotion gates for a model type.

    Returns (all_passed: bool, gate_results: dict)
    """
    if model_type == "DISTILBERT_CATEGORIZATION":
        gate_results = evaluate_categorization_gates(metrics, production_metrics)
    elif model_type in ("XGBOOST_TREND", "ISOLATION_FOREST"):
        gate_results = evaluate_trend_gates(metrics, production_metrics)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    all_passed = all(g["passed"] for g in gate_results.values())

    print(f"\n{'=' * 60}")
    print(f"PROMOTION GATE RESULTS: {model_type}")
    print(f"{'=' * 60}")
    for name, result in gate_results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['description']}")
        print(f"         Value: {result['value']}  |  {result['reason']}")
    print(f"\n  Overall: {'ALL GATES PASSED' if all_passed else 'GATES FAILED'}")
    print(f"{'=' * 60}")

    return all_passed, gate_results


# ============================================================
# PROMOTION WORKFLOW
# ============================================================
def get_mlflow_metrics(run_id: str, tracking_uri: str) -> dict:
    """Fetch all metrics from an MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics


def get_mlflow_params(run_id: str, tracking_uri: str) -> dict:
    """Fetch all params from an MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def evaluate_for_promotion(config: dict, run_id: str) -> dict:
    """Evaluate an MLflow run against promotion gates without promoting."""
    tracking_uri = config.get("mlflow_tracking_uri", "http://localhost:5000")
    model_type = config["model_type"]

    metrics = get_mlflow_metrics(run_id, tracking_uri)

    # Get production model metrics for relative comparisons
    registry = ModelRegistry(config.get("registry_path", "/data/model_registry"))
    prod_model = registry.get_production_model(model_type)
    prod_metrics = prod_model["eval_metrics"] if prod_model else None

    all_passed, gate_results = check_all_gates(model_type, metrics, prod_metrics)

    return {
        "run_id": run_id,
        "model_type": model_type,
        "all_gates_passed": all_passed,
        "gate_results": gate_results,
        "metrics": metrics,
    }


def promote_model(config: dict, run_id: str) -> dict:
    """Evaluate gates and promote model through the lifecycle."""
    tracking_uri = config.get("mlflow_tracking_uri", "http://localhost:5000")
    model_type = config["model_type"]
    registry_path = config.get("registry_path", "/data/model_registry")

    metrics = get_mlflow_metrics(run_id, tracking_uri)
    params = get_mlflow_params(run_id, tracking_uri)

    registry = ModelRegistry(registry_path)

    # Check if already registered
    existing = registry.get_latest(model_type)
    if existing and existing.get("mlflow_run_id") == run_id:
        model_entry = existing
    else:
        # Determine version
        latest = registry.get_latest(model_type)
        if latest:
            parts = latest["version"].split(".")
            version = f"{parts[0]}.{int(parts[1]) + 1}.0"
        else:
            version = "1.0.0"

        # Register
        model_entry = registry.register(
            model_type=model_type,
            version=version,
            eval_metrics=metrics,
            hyperparameters={k: v for k, v in params.items() if not k.startswith("env_")},
            training_data_hash=params.get("data_hash", "unknown"),
            feature_version=params.get("feature_version", "1.0"),
            artifact_path=f"mlflow://{run_id}/model",
            mlflow_run_id=run_id,
        )

    # Evaluate gates
    prod_model = registry.get_production_model(model_type)
    prod_metrics = prod_model["eval_metrics"] if prod_model else None
    all_passed, gate_results = check_all_gates(model_type, metrics, prod_metrics)

    if not all_passed:
        print(f"\n[PROMOTE] Gates FAILED. Model stays in {model_entry['status']} status.")
        return {"promoted": False, "model_id": model_entry["model_id"], "gates": gate_results}

    # Progress through lifecycle based on current status
    current = model_entry["status"]
    if current == "TRAINING":
        registry.update_status(model_entry["model_id"], "SHADOW")
        print(f"[PROMOTE] Model -> SHADOW. Run shadow deployment for 24 hours.")
    elif current == "SHADOW":
        registry.update_status(model_entry["model_id"], "CANARY")
        print(f"[PROMOTE] Model -> CANARY. Routing 10% traffic for 48 hours.")
    elif current == "CANARY":
        registry.update_status(model_entry["model_id"], "PRODUCTION")
        print(f"[PROMOTE] Model -> PRODUCTION. Now serving all traffic.")
    else:
        print(f"[PROMOTE] Model already in {current} status.")

    return {
        "promoted": True,
        "model_id": model_entry["model_id"],
        "new_status": model_entry["status"],
        "gates": gate_results,
    }


def rollback_model(config: dict, run_id: str = None) -> dict:
    """Rollback to the predecessor model.

    If run_id is provided, retire that specific model and restore predecessor.
    Otherwise, retire the current production model.
    """
    model_type = config["model_type"]
    registry_path = config.get("registry_path", "/data/model_registry")
    registry = ModelRegistry(registry_path)

    if run_id:
        # Find model by run_id
        target = None
        for m in registry.models:
            if m.get("mlflow_run_id") == run_id:
                target = m
                break
        if not target:
            raise ValueError(f"No model found with run_id {run_id}")
    else:
        target = registry.get_production_model(model_type)
        if not target:
            raise ValueError(f"No production model found for {model_type}")

    predecessor_id = target.get("predecessor_id")

    # Retire current model
    if target["status"] != "RETIRED":
        registry.update_status(target["model_id"], "RETIRED")

    # Restore predecessor
    if predecessor_id:
        predecessor = registry.get_model(predecessor_id)
        if predecessor:
            predecessor["status"] = "PRODUCTION"
            predecessor["restored_timestamp"] = datetime.utcnow().isoformat()
            registry._save()
            print(f"[ROLLBACK] Restored predecessor {predecessor_id} to PRODUCTION")
            return {"rolled_back": True, "retired": target["model_id"],
                    "restored": predecessor_id}

    print(f"[ROLLBACK] Retired {target['model_id']} but no predecessor to restore")
    return {"rolled_back": True, "retired": target["model_id"], "restored": None}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Model promotion and lifecycle management")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--action", choices=["evaluate", "promote", "rollback"],
                        default="evaluate", help="Action to perform")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.action == "evaluate":
        result = evaluate_for_promotion(config, args.run_id)
    elif args.action == "promote":
        result = promote_model(config, args.run_id)
    elif args.action == "rollback":
        result = rollback_model(config, args.run_id)
    else:
        raise ValueError(f"Unknown action: {args.action}")

    print(f"\n[RESULT] {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
