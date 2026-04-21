"""
utils.py — Shared utilities for training scripts.
Handles config loading, MLflow setup, environment logging, and data loading.
"""

import yaml
import sys
import os
import time
import platform
import psutil
import torch
import mlflow
import hashlib
import json
import subprocess
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from mlflow.tracking import MlflowClient


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from {config_path}")
    print(f"[INFO] Config: {json.dumps(config, indent=2, default=str)}")
    return config


def setup_mlflow(config: dict) -> str:
    """Configure MLflow tracking and start a run. Returns the run ID."""
    tracking_uri = config.get("mlflow_tracking_uri", "http://localhost:8000")
    experiment_name = config.get("experiment_name", "default")
    artifact_location = config.get("artifact_location")
    direct_experiment_name = config.get(
        "direct_artifact_experiment_name",
        f"{experiment_name}_s3",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    chosen_experiment_name = experiment_name
    experiment = client.get_experiment_by_name(experiment_name)

    if artifact_location:
        if experiment is None:
            client.create_experiment(experiment_name, artifact_location=artifact_location)
            experiment = client.get_experiment_by_name(experiment_name)
        elif not str(experiment.artifact_location or "").startswith("s3://"):
            chosen_experiment_name = direct_experiment_name
            direct_experiment = client.get_experiment_by_name(chosen_experiment_name)
            if direct_experiment is None:
                client.create_experiment(
                    chosen_experiment_name,
                    artifact_location=artifact_location,
                )
                direct_experiment = client.get_experiment_by_name(chosen_experiment_name)
            experiment = direct_experiment
        else:
            chosen_experiment_name = experiment_name

    mlflow.set_experiment(chosen_experiment_name)
    mlflow.enable_system_metrics_logging()
    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow experiment: {chosen_experiment_name}")
    if experiment is not None:
        print(f"[INFO] MLflow artifact location: {experiment.artifact_location}")
    return tracking_uri


def tag_primary_model_artifact(model_info):
    """Attach direct artifact metadata for the primary logged model to the active run."""
    active_run = mlflow.active_run()
    if active_run is None or model_info is None:
        return

    model_id = getattr(model_info, "model_id", None)
    model_uri = getattr(model_info, "model_uri", None)

    if model_id:
        mlflow.set_tag("registry_model_id", model_id)

    artifact_uri = active_run.info.artifact_uri or ""
    if artifact_uri.startswith("s3://") and model_id:
        suffix = f"/{active_run.info.run_id}/artifacts"
        if artifact_uri.endswith(suffix):
            experiment_root = artifact_uri[: -len(suffix)]
        else:
            experiment_root = artifact_uri.rsplit("/artifacts", 1)[0]
        direct_model_uri = f"{experiment_root}/models/{model_id}/artifacts"
        mlflow.set_tag("registry_model_artifact_path", direct_model_uri)
    elif model_uri:
        mlflow.set_tag("registry_model_artifact_path", model_uri)


def log_environment_info():
    """Log hardware and software environment to MLflow."""
    env_info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "os": platform.system(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
        )
        env_info["cuda_version"] = torch.version.cuda

    # Try to get git SHA
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        env_info["git_sha"] = sha[:8]
    except Exception:
        env_info["git_sha"] = "unknown"

    for k, v in env_info.items():
        mlflow.log_param(f"env_{k}", v)

    print(f"[INFO] Environment: {json.dumps(env_info, indent=2)}")
    return env_info


def _parse_s3_uri(uri: str):
    """Split s3://bucket/key into (bucket, key)."""
    without_scheme = uri[len("s3://"):]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    return parts[0], parts[1]


def _registry_file_path(registry_path: str, filename: str) -> str:
    """Resolve a registry file path for either local fs or s3:// storage."""
    if registry_path.startswith("s3://"):
        return registry_path.rstrip("/") + f"/{filename}"
    return os.path.join(registry_path, filename)


def get_s3_storage_options(s3_cfg: dict) -> dict:
    """Build pandas/fsspec storage_options dict for MinIO-compatible S3.

    Credentials fall back to AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars.
    """
    return {
        "key": s3_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret": s3_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {
            "endpoint_url": s3_cfg.get("endpoint_url"),
            "region_name": s3_cfg.get("region", "us-east-1"),
        },
    }


def get_s3_client(s3_cfg: dict):
    """Create a boto3 S3 client for MinIO-compatible endpoints."""
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise ImportError("boto3 is required for S3-backed registry storage") from e

    return boto3.client(
        "s3",
        endpoint_url=s3_cfg.get("endpoint_url"),
        aws_access_key_id=s3_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=s3_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=s3_cfg.get("region", "us-east-1"),
    )


def load_json_document(path: str, default, config: dict = None):
    """Load a JSON document from local disk or s3://, returning default if missing."""
    if path.startswith("s3://"):
        if config is None:
            raise ValueError("load_json_document(s3://...) requires a config")
        s3_cfg = config.get("s3", {})
        client = get_s3_client(s3_cfg)
        bucket, key = _parse_s3_uri(path)
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except client.exceptions.NoSuchKey:
            return default
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                return default
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_json_document(path: str, payload, config: dict = None):
    """Persist a JSON document to local disk or s3://."""
    if path.startswith("s3://"):
        if config is None:
            raise ValueError("save_json_document(s3://...) requires a config")
        s3_cfg = config.get("s3", {})
        client = get_s3_client(s3_cfg)
        bucket, key = _parse_s3_uri(path)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_parquet(data_path: str, config: dict) -> pd.DataFrame:
    """Load a parquet from a local path or s3:// URI.

    For s3://, uses config['s3'] endpoint_url + env-var credentials so MinIO works.
    """
    if data_path.startswith("s3://"):
        s3_cfg = config.get("s3", {})
        if not s3_cfg.get("endpoint_url"):
            raise ValueError(
                "data_path is s3:// but config.s3.endpoint_url is not set"
            )
        storage_options = get_s3_storage_options(s3_cfg)
        print(f"[INFO] Reading parquet from {data_path} via {s3_cfg['endpoint_url']}")
        return pd.read_parquet(data_path, storage_options=storage_options)
    return pd.read_parquet(data_path)


def compute_data_hash(data_path: str, config: dict = None) -> str:
    """Compute a stable content fingerprint of training data for reproducibility.

    - Local path: SHA-256 of file bytes (first 16 hex chars).
    - s3:// URI: MinIO/S3 ETag from head_object (cheap, server-computed).
    """
    if data_path.startswith("s3://"):
        if config is None:
            raise ValueError("compute_data_hash(s3://...) requires a config")
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise ImportError("boto3 is required to hash S3 objects") from e
        s3_cfg = config.get("s3", {})
        bucket, key = _parse_s3_uri(data_path)
        client = boto3.client(
            "s3",
            endpoint_url=s3_cfg.get("endpoint_url"),
            aws_access_key_id=s3_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=s3_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=s3_cfg.get("region", "us-east-1"),
        )
        head = client.head_object(Bucket=bucket, Key=key)
        etag = head["ETag"].strip('"')
        return f"etag:{etag[:16]}"

    sha = hashlib.sha256()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def get_device():
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


class TrainingTimer:
    """Context manager for timing training phases."""

    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[TIMER] {self.phase_name} started...")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"[TIMER] {self.phase_name} completed in {self.elapsed:.2f}s")
        mlflow.log_metric(
            f"{self.phase_name.lower().replace(' ', '_')}_time_sec",
            round(self.elapsed, 2),
        )


def log_peak_memory():
    """Log peak memory usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    mlflow.log_metric("peak_memory_mb", round(mem_mb, 2))
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mlflow.log_metric("peak_gpu_memory_mb", round(gpu_mem_mb, 2))
    return mem_mb


def _load_registry_entries(registry_file: str, config: dict = None) -> list:
    """Load registry entries while tolerating older list-only formats."""
    payload = load_json_document(registry_file, default=[], config=config)

    if isinstance(payload, dict):
        return payload.get("models", [])
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported registry format in {registry_file}")


def _save_registry_entries(registry_file: str, entries: list, config: dict = None):
    """Persist registry entries in a backward-compatible structure."""
    sanitized_entries = []
    for entry in entries:
        clean_entry = dict(entry)
        clean_entry.pop("selection_state", None)
        sanitized_entries.append(clean_entry)

    payload = {
        "schema_version": 3,
        "updated_at": datetime.utcnow().isoformat(),
        "models": sanitized_entries,
    }
    save_json_document(registry_file, payload, config=config)


def _default_active_models_payload() -> dict:
    """Return the canonical active-model payload structure."""
    return {
        "schema_version": 2,
        "updated_at": datetime.utcnow().isoformat(),
        "categorization": None,
        "trend": None,
    }


def _registry_storage_config(config: dict = None) -> dict:
    """Build storage config for registry operations with env fallbacks."""
    config = config or {}
    return {
        "s3": {
            "endpoint_url": (
                config.get("s3", {}).get("endpoint_url")
                or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
            ),
            "region": config.get("s3", {}).get("region", "us-east-1"),
            "access_key": config.get("s3", {}).get("access_key"),
            "secret_key": config.get("s3", {}).get("secret_key"),
        }
    }


def _build_active_model_record(entry: dict) -> dict:
    """Materialize the fields serving/retraining need for an active model."""
    return {
        "registry_id": entry.get("registry_id"),
        "task_type": entry.get("task_type"),
        "model_id": entry.get("model_id"),
        "model_family": entry.get("model_family"),
        "legacy_model_type": entry.get("legacy_model_type"),
        "version": entry.get("version"),
        "status": entry.get("status"),
        "training_mode": entry.get("training_mode"),
        "run_name": entry.get("run_name"),
        "mlflow_run_id": entry.get("mlflow_run_id"),
        "experiment_name": entry.get("experiment_name"),
        "artifact_path": entry.get("artifact_path"),
        "data_path": entry.get("data_path"),
        "training_data_hash": entry.get("training_data_hash"),
        "feature_version": entry.get("feature_version"),
        "serving_artifacts": entry.get("serving_artifacts", {}),
        "updated_at": datetime.utcnow().isoformat(),
    }


def _resolve_registry_entry(entries: list, registry_id: str = None, model_id: str = None):
    """Resolve a registry entry by registry_id first, then by model_id."""
    if registry_id:
        for entry in reversed(entries):
            if entry.get("registry_id") == registry_id:
                return entry
    if model_id:
        for entry in reversed(entries):
            if entry.get("model_id") == model_id:
                return entry
    return None


def _normalize_active_models_payload(payload, entries: list) -> dict:
    """Migrate older active-model formats into the canonical structure."""
    if not isinstance(payload, dict):
        return _default_active_models_payload()

    if "categorization" in payload or "trend" in payload:
        normalized = _default_active_models_payload()
        normalized["schema_version"] = payload.get("schema_version", 2)
        normalized["updated_at"] = payload.get("updated_at", normalized["updated_at"])
        normalized["categorization"] = payload.get("categorization")
        normalized["trend"] = payload.get("trend")
        return normalized

    normalized = _default_active_models_payload()
    normalized["updated_at"] = payload.get("updated_at", normalized["updated_at"])

    legacy_categorization = payload.get("active_categorization_model")
    legacy_trend = payload.get("active_trend_model")

    categorization_entry = _resolve_registry_entry(entries, model_id=legacy_categorization)
    trend_entry = _resolve_registry_entry(entries, model_id=legacy_trend)

    normalized["categorization"] = (
        _build_active_model_record(categorization_entry) if categorization_entry else None
    )
    normalized["trend"] = _build_active_model_record(trend_entry) if trend_entry else None
    return normalized


def _ensure_active_models_file(active_models_file: str, entries: list = None, config: dict = None):
    """Create or migrate the active-model selection file."""
    entries = entries or []
    existing = load_json_document(active_models_file, default=None, config=config)
    normalized = _normalize_active_models_payload(existing, entries)
    if existing != normalized:
        save_json_document(active_models_file, normalized, config=config)
    return normalized


def _infer_task_type(config: dict) -> str:
    task_type = config.get("task_type")
    if task_type:
        return task_type

    model_type = config.get("model_type", "")
    if model_type in {"logistic_regression", "distilbert", "DISTILBERT_CATEGORIZATION"}:
        return "categorization"
    return "trend"


def _infer_model_family(config: dict) -> str:
    family = config.get("model_family")
    if family:
        return family
    return config.get("model_type", "unknown")


def _next_model_version(entries: list, model_id: str) -> str:
    """Generate a monotonic semantic-ish version per candidate model ID."""
    current_versions = []
    for entry in entries:
        if entry.get("model_id") == model_id:
            version = entry.get("version")
            if isinstance(version, str):
                current_versions.append(version)

    if not current_versions:
        return "1.0.0"

    latest = current_versions[-1].split(".")
    major = latest[0] if len(latest) > 0 else "1"
    minor = int(latest[1]) + 1 if len(latest) > 1 and latest[1].isdigit() else 1
    return f"{major}.{minor}.0"


def load_active_models(registry_path: str) -> dict:
    """Load the active-model selection JSON from the registry directory."""
    active_models_file = _registry_file_path(registry_path, "active_models.json")
    registry_file = _registry_file_path(registry_path, "registry.json")
    registry_config = _registry_storage_config()
    entries = _load_registry_entries(registry_file, config=registry_config)
    return _ensure_active_models_file(active_models_file, entries=entries, config=registry_config)


def get_active_model_record(registry_path: str, task_type: str, config: dict = None):
    """Return the authoritative active model record for a task."""
    registry_path = registry_path or "s3://mlflow/registry"
    registry_file = _registry_file_path(registry_path, "registry.json")
    active_models_file = _registry_file_path(registry_path, "active_models.json")
    registry_config = _registry_storage_config(config)
    entries = _load_registry_entries(registry_file, config=registry_config)
    payload = _ensure_active_models_file(active_models_file, entries=entries, config=registry_config)
    return payload.get(task_type)


def get_active_model_artifact_path(registry_path: str, task_type: str, config: dict = None):
    """Convenience helper to fetch the current active model artifact path for a task."""
    record = get_active_model_record(registry_path, task_type, config=config)
    if record is None:
        return None
    return record.get("artifact_path")


def set_active_models(
    registry_path: str,
    active_categorization_model: str = None,
    active_trend_model: str = None,
    active_categorization_registry_id: str = None,
    active_trend_registry_id: str = None,
) -> dict:
    """Update active model selections for categorization and trend tasks."""
    registry_path = registry_path or "s3://mlflow/registry"
    active_models_file = _registry_file_path(registry_path, "active_models.json")
    registry_file = _registry_file_path(registry_path, "registry.json")
    registry_config = _registry_storage_config()
    entries = _load_registry_entries(registry_file, config=registry_config)
    current = _ensure_active_models_file(active_models_file, entries=entries, config=registry_config)

    if active_categorization_model is not None or active_categorization_registry_id is not None:
        entry = _resolve_registry_entry(
            entries,
            registry_id=active_categorization_registry_id,
            model_id=active_categorization_model,
        )
        if entry is None:
            raise ValueError(
                "Categorization selection not found in registry: "
                f"registry_id={active_categorization_registry_id}, "
                f"model_id={active_categorization_model}"
            )
        updated = _build_active_model_record(entry)
        existing = current.get("categorization") or {}
        if existing.get("registry_id") == updated.get("registry_id") and existing.get("serving_artifacts"):
            updated["serving_artifacts"] = existing["serving_artifacts"]
        current["categorization"] = updated
    if active_trend_model is not None or active_trend_registry_id is not None:
        entry = _resolve_registry_entry(
            entries,
            registry_id=active_trend_registry_id,
            model_id=active_trend_model,
        )
        if entry is None:
            raise ValueError(
                "Trend selection not found in registry: "
                f"registry_id={active_trend_registry_id}, "
                f"model_id={active_trend_model}"
            )
        updated = _build_active_model_record(entry)
        existing = current.get("trend") or {}
        if existing.get("registry_id") == updated.get("registry_id") and existing.get("serving_artifacts"):
            updated["serving_artifacts"] = existing["serving_artifacts"]
        current["trend"] = updated

    current["updated_at"] = datetime.utcnow().isoformat()
    save_json_document(active_models_file, current, config=registry_config)

    print(f"[REGISTRY] Active models updated: {json.dumps(current, indent=2)}")
    return current


def _get_first_metric(metrics: dict, candidates: list[str]):
    """Return the first present metric value from a list of candidate keys."""
    for key in candidates:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def _categorization_score(entry: dict) -> tuple:
    """Score categorization candidates. Higher is better."""
    metrics = entry.get("eval_metrics", {})
    macro_f1 = _get_first_metric(metrics, ["test_macro_f1", "macro_f1"])
    accuracy = _get_first_metric(
        metrics,
        ["test_subset_accuracy", "test_accuracy", "test_exact_match_accuracy"],
    )
    abstention = _get_first_metric(
        metrics,
        ["test_abstention_rate", "abstention_rate"],
    )

    if macro_f1 is None:
        macro_f1 = float("-inf")
    if accuracy is None:
        accuracy = float("-inf")

    # Prefer abstention inside the recommended 5%-20% band, then closer to 10%.
    abstention_in_band = 0
    abstention_closeness = float("-inf")
    if abstention is not None:
        abstention_in_band = 1 if 0.05 <= abstention <= 0.20 else 0
        abstention_closeness = -abs(abstention - 0.10)

    return (
        macro_f1,
        accuracy,
        abstention_in_band,
        abstention_closeness,
        entry.get("created_at", ""),
    )


def _trend_score(entry: dict) -> tuple:
    """Score trend candidates. Higher is better after sign-normalization."""
    metrics = entry.get("eval_metrics", {})
    rmse = _get_first_metric(
        metrics,
        ["xgb_optuna_test_rmse", "xgb_test_rmse", "rf_test_rmse"],
    )
    mae = _get_first_metric(
        metrics,
        ["xgb_optuna_test_mae", "xgb_test_mae", "rf_test_mae"],
    )
    r2 = _get_first_metric(
        metrics,
        ["xgb_optuna_test_r2", "xgb_test_r2", "rf_test_r2"],
    )
    precision_at_5 = _get_first_metric(
        metrics,
        ["xgb_optuna_precision_at_5", "xgb_precision_at_5", "rf_precision_at_5"],
    )

    if rmse is None:
        rmse = float("inf")
    if mae is None:
        mae = float("inf")
    if r2 is None:
        r2 = float("-inf")
    if precision_at_5 is None:
        precision_at_5 = float("-inf")

    return (
        -rmse,
        -mae,
        r2,
        precision_at_5,
        entry.get("created_at", ""),
    )


def _choose_best_candidate(entries: list, task_type: str):
    """Pick the best candidate entry for a task based on stored evaluation metrics."""
    eligible = [
        entry for entry in entries
        if entry.get("task_type") == task_type
        and entry.get("training_mode", "initial") == "initial"
        and entry.get("status") in {"CANDIDATE", "TRAINING", "SHADOW", "CANARY", "PRODUCTION"}
    ]
    if not eligible:
        return None

    if task_type == "categorization":
        return max(eligible, key=_categorization_score)
    return max(eligible, key=_trend_score)


def update_active_model_selection(registry_path: str, config: dict = None) -> dict:
    """Recompute active model winners from registry metrics and persist selections."""
    registry_file = _registry_file_path(registry_path, "registry.json")
    active_models_file = _registry_file_path(registry_path, "active_models.json")
    config = config or {}
    registry_config = _registry_storage_config(config)

    entries = _load_registry_entries(registry_file, config=registry_config)
    current = _ensure_active_models_file(active_models_file, entries=entries, config=registry_config)

    best_categorization = _choose_best_candidate(entries, "categorization")
    best_trend = _choose_best_candidate(entries, "trend")

    next_categorization = (
        _build_active_model_record(best_categorization) if best_categorization else None
    )
    next_trend = _build_active_model_record(best_trend) if best_trend else None

    existing_categorization = current.get("categorization") or {}
    if (
        next_categorization
        and existing_categorization.get("registry_id") == next_categorization.get("registry_id")
        and existing_categorization.get("serving_artifacts")
    ):
        next_categorization["serving_artifacts"] = existing_categorization["serving_artifacts"]

    existing_trend = current.get("trend") or {}
    if (
        next_trend
        and existing_trend.get("registry_id") == next_trend.get("registry_id")
        and existing_trend.get("serving_artifacts")
    ):
        next_trend["serving_artifacts"] = existing_trend["serving_artifacts"]

    current["categorization"] = next_categorization
    current["trend"] = next_trend
    current["updated_at"] = datetime.utcnow().isoformat()

    save_json_document(active_models_file, current, config=registry_config)

    print(
        "[REGISTRY] Auto-selected active models: "
        f"categorization={current['categorization']['model_id'] if current['categorization'] else None}, "
        f"trend={current['trend']['model_id'] if current['trend'] else None}"
    )
    return current


def register_candidate_model(config: dict, run_id: str, metrics: dict) -> dict:
    """Register a successfully trained model candidate in the file registry."""
    registry_path = config.get("registry_path", "s3://mlflow/registry")

    registry_file = _registry_file_path(registry_path, "registry.json")
    active_models_file = _registry_file_path(registry_path, "active_models.json")
    entries = _load_registry_entries(registry_file, config=config)
    _ensure_active_models_file(active_models_file, entries=entries, config=config)
    for entry in entries:
        if entry.get("mlflow_run_id") == run_id:
            print(f"[REGISTRY] Run {run_id} already registered as {entry.get('model_id')}")
            return entry

    task_type = _infer_task_type(config)
    model_id = config.get("model_id", config.get("run_name", "unknown_model"))
    model_family = _infer_model_family(config)
    version = _next_model_version(entries, model_id)
    artifact_subpath = config.get("artifact_subpath", "model")
    tracking_uri = config.get("mlflow_tracking_uri", "http://localhost:5000")

    artifact_path = f"mlflow://{run_id}/{artifact_subpath}"
    try:
        run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
        artifact_path = run.data.tags.get("registry_model_artifact_path", artifact_path)
        if (
            artifact_path == f"mlflow://{run_id}/{artifact_subpath}"
            and str(run.info.artifact_uri or "").startswith("s3://")
        ):
            artifact_path = run.info.artifact_uri.rstrip("/") + f"/{artifact_subpath}"
    except Exception as exc:
        print(f"[WARN] Could not resolve direct artifact path for run {run_id}: {exc}")

    hyperparameters = {
        k: v for k, v in config.items()
        if not k.startswith("_")
        and k not in {"s3", "database"}
    }

    entry = {
        "registry_id": str(uuid.uuid4()),
        "task_type": task_type,
        "model_id": model_id,
        "model_family": model_family,
        "legacy_model_type": config.get("model_type"),
        "version": version,
        "status": config.get("initial_status", "CANDIDATE"),
        "training_mode": config.get("training_mode", "initial"),
        "run_name": config.get("run_name"),
        "mlflow_run_id": run_id,
        "experiment_name": config.get("experiment_name"),
        "artifact_path": artifact_path,
        "data_path": config.get("data_path"),
        "training_data_hash": hyperparameters.get("data_hash", "unknown"),
        "feature_version": hyperparameters.get("feature_version", "1.0"),
        "hyperparameters": hyperparameters,
        "eval_metrics": metrics,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    entries.append(entry)
    _save_registry_entries(registry_file, entries, config=config)
    print(
        f"[REGISTRY] Registered candidate {model_id} "
        f"(task={task_type}, family={model_family}, version={version}, status={entry['status']})"
    )
    if config.get("training_mode", "initial") == "initial":
        update_active_model_selection(registry_path, config=config)
    return entry
