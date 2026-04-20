"""Shared helpers — MinIO client, DuckDB connection, logging, hashing."""
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
from minio import Minio

import config


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(name: str) -> logging.Logger:
    """Stdout logger with timestamps — cron-friendly."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# MinIO client
# ─────────────────────────────────────────────────────────────────────────────
def get_minio_client() -> Minio:
    if not config.MINIO_ACCESS_KEY or not config.MINIO_SECRET_KEY:
        raise RuntimeError(
            "MINIO_ACCESS_KEY / MINIO_SECRET_KEY not set — check your .env"
        )
    return Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=config.MINIO_SECURE,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


# ─────────────────────────────────────────────────────────────────────────────
# DuckDB connection with MinIO (httpfs) configured
# ─────────────────────────────────────────────────────────────────────────────
def get_duckdb() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection that can read s3:// from our MinIO."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{config.MINIO_ENDPOINT}';")
    con.execute(f"SET s3_access_key_id='{config.MINIO_ACCESS_KEY}';")
    con.execute(f"SET s3_secret_access_key='{config.MINIO_SECRET_KEY}';")
    con.execute(f"SET s3_use_ssl={'true' if config.MINIO_SECURE else 'false'};")
    con.execute("SET s3_url_style='path';")
    return con


# ─────────────────────────────────────────────────────────────────────────────
# Hashing — used for manifest.training_data_hash (§9 registry)
# ─────────────────────────────────────────────────────────────────────────────
def sha256_of_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return "sha256:" + h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Version folder naming — matches the retraining pipeline flow diagram
# ─────────────────────────────────────────────────────────────────────────────
def version_folder(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"v={now.strftime('%Y-%m-%d')}"


# ─────────────────────────────────────────────────────────────────────────────
# Upload helpers
# ─────────────────────────────────────────────────────────────────────────────
def upload_file(client: Minio, bucket: str, key: str, local_path: Path) -> None:
    client.fput_object(bucket, key, str(local_path))


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def update_latest_pointer(
    client: Minio, task: str, version: str, built_at: datetime
) -> None:
    """Write <bucket>/<task>/latest.json so the training team has one stable
    path to check for the newest dataset."""
    pointer = {
        "version": version,
        "path": f"{config.BUCKET_RETRAINING_DATA}/{task}/{version}/",
        "built_at": built_at.isoformat(),
    }
    local = config.TMP_DIR / f"latest_{task}.json"
    write_json(local, pointer)
    upload_file(client, config.BUCKET_RETRAINING_DATA, f"{task}/latest.json", local)
