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
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


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

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()
    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow experiment: {experiment_name}")
    return tracking_uri


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
