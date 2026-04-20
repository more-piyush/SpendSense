"""Materialize serving-ready artifacts for active models.

Current behavior:
  - Categorization / distilbert: export custom model to ONNX and quantize to int8
  - Trend / random_forest|xgboost|xgboost_optuna: export primary regressor to ONNX

Artifacts are written to a serving-artifact root (default: s3://mlflow/serving-artifacts)
and referenced from active_models.json under each task's `serving_artifacts` field.
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import boto3
import mlflow.sklearn
import mlflow.xgboost
import onnx
import onnxmltools
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from transformers import DistilBertTokenizer

from train_categorization import DistilBertCategorizer
from train_trend_detection import FEATURE_COLUMNS
from utils import (
    _parse_s3_uri,
    _registry_file_path,
    _registry_storage_config,
    load_json_document,
    save_json_document,
)


def _s3_client(config: dict):
    s3_cfg = (config or {}).get("s3", {})
    return boto3.client(
        "s3",
        endpoint_url=s3_cfg.get("endpoint_url") or os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=s3_cfg.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=s3_cfg.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=s3_cfg.get("region", "us-east-1"),
    )


def _download_s3_prefix(s3_uri: str, local_root: Path, client):
    bucket, prefix = _parse_s3_uri(s3_uri.rstrip("/"))
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            found = True
            relative = key[len(prefix):].lstrip("/")
            destination = local_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(destination))
    if not found:
        raise FileNotFoundError(f"No objects found under {s3_uri}")
    return local_root


def _materialize_artifact(artifact_path: str, client, temp_root: Path) -> Path:
    if artifact_path.startswith("s3://"):
        return _download_s3_prefix(artifact_path, temp_root, client)
    return Path(artifact_path)


def _upload_dir(local_dir: Path, dest_s3_uri: str, client):
    bucket, prefix = _parse_s3_uri(dest_s3_uri.rstrip("/"))
    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        client.upload_file(str(file_path), bucket, key)


def _locate_checkpoint(artifact_dir: Path) -> Path:
    for candidate in artifact_dir.rglob("model.pt"):
        return candidate
    raise FileNotFoundError(f"Unable to locate model.pt under {artifact_dir}")


def _locate_tokenizer_dir(artifact_dir: Path) -> Path | None:
    for candidate in artifact_dir.rglob("tokenizer_config.json"):
        return candidate.parent
    return None


def _export_distilbert_categorization(record: dict, export_root: str, client) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        artifact_dir = _materialize_artifact(record["artifact_path"], client, tmp_root / "source")
        checkpoint_path = _locate_checkpoint(artifact_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})

        model = DistilBertCategorizer(
            pretrained_model=config.get("pretrained_model", "distilbert-base-uncased"),
            n_classes=checkpoint.get("n_classes", len(checkpoint.get("classes", []))),
            dropout=config.get("dropout", 0.3),
            freeze_layers=config.get("freeze_layers", 0),
            freeze_embeddings=config.get("freeze_embeddings", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        max_length = int(config.get("max_length", 64))
        input_ids = torch.ones((1, max_length), dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        amount = torch.zeros((1,), dtype=torch.float32)
        currency_idx = torch.zeros((1,), dtype=torch.long)
        country_idx = torch.zeros((1,), dtype=torch.long)

        export_dir = tmp_root / "export"
        tokenizer_dir = export_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        source_tokenizer_dir = _locate_tokenizer_dir(artifact_dir)
        if source_tokenizer_dir is not None:
            tokenizer = DistilBertTokenizer.from_pretrained(source_tokenizer_dir)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained(
                config.get("pretrained_model", "distilbert-base-uncased")
            )
        tokenizer.save_pretrained(tokenizer_dir)

        onnx_path = export_dir / "model.onnx"
        quantized_path = export_dir / "model.quantized.onnx"

        torch.onnx.export(
            model,
            (input_ids, attention_mask, amount, currency_idx, country_idx),
            str(onnx_path),
            input_names=[
                "input_ids",
                "attention_mask",
                "amount",
                "currency_idx",
                "country_idx",
            ],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "amount": {0: "batch_size"},
                "currency_idx": {0: "batch_size"},
                "country_idx": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
        )
        onnx.checker.check_model(str(onnx_path))
        quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)

        metadata = {
            "task_type": "categorization",
            "model_family": record["model_family"],
            "input_names": [
                "input_ids",
                "attention_mask",
                "amount",
                "currency_idx",
                "country_idx",
            ],
            "output_name": "logits",
            "classes": checkpoint.get("classes", []),
            "max_length": max_length,
            "pretrained_model": config.get("pretrained_model", "distilbert-base-uncased"),
            "currency_vocab": checkpoint.get("currency_vocab", {}),
            "country_vocab": checkpoint.get("country_vocab", {}),
            "source_artifact_path": record["artifact_path"],
            "source_registry_id": record["registry_id"],
        }
        (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        root_path = f"{export_root.rstrip('/')}/categorization/{record['registry_id']}"
        _upload_dir(export_dir, root_path, client)
        return {
            "preferred_format": "onnx_quantized",
            "root_path": root_path,
            "onnx_model": "model.onnx",
            "onnx_quantized_model": "model.quantized.onnx",
            "metadata": "metadata.json",
            "tokenizer_dir": "tokenizer",
            "source_artifact_path": record["artifact_path"],
        }


def _export_trend_model(record: dict, export_root: str, client) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        artifact_dir = _materialize_artifact(record["artifact_path"], client, tmp_root / "source")
        family = record["model_family"]

        if family == "random_forest":
            model = mlflow.sklearn.load_model(str(artifact_dir))
            onx = convert_sklearn(
                model,
                initial_types=[("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                target_opset=15,
            )
        elif family in {"xgboost", "xgboost_optuna"}:
            model = mlflow.xgboost.load_model(str(artifact_dir))
            onx = onnxmltools.convert_xgboost(
                model,
                initial_types=[("features", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                target_opset=15,
            )
        else:
            return {}

        export_dir = tmp_root / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_dir / "model.onnx"
        onnx.save_model(onx, str(onnx_path))

        metadata = {
            "task_type": "trend",
            "model_family": family,
            "input_name": "features",
            "feature_columns": FEATURE_COLUMNS,
            "source_artifact_path": record["artifact_path"],
            "source_registry_id": record["registry_id"],
        }
        (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        root_path = f"{export_root.rstrip('/')}/trend/{record['registry_id']}"
        _upload_dir(export_dir, root_path, client)
        return {
            "preferred_format": "onnx",
            "root_path": root_path,
            "onnx_model": "model.onnx",
            "metadata": "metadata.json",
            "source_artifact_path": record["artifact_path"],
        }


def _build_serving_artifacts(record: dict, export_root: str, client) -> dict:
    if not record or not record.get("artifact_path"):
        return {}
    if record.get("task_type") == "categorization" and record.get("model_family") == "distilbert":
        return _export_distilbert_categorization(record, export_root, client)
    if record.get("task_type") == "trend":
        return _export_trend_model(record, export_root, client)
    return {}


def materialize_active_serving_artifacts(registry_path: str, config: dict | None = None) -> dict:
    registry_path = registry_path or "s3://mlflow/registry"
    config = _registry_storage_config(config)
    active_models_file = _registry_file_path(registry_path, "active_models.json")
    payload = load_json_document(active_models_file, default={}, config=config)
    client = _s3_client(config)
    export_root = os.environ.get("SERVING_ARTIFACT_ROOT", "s3://mlflow/serving-artifacts")

    for task_type in ("categorization", "trend"):
        record = payload.get(task_type)
        if not record:
            continue
        existing = record.get("serving_artifacts", {})
        if existing.get("source_artifact_path") == record.get("artifact_path") and existing.get("preferred_format"):
            continue
        artifacts = _build_serving_artifacts(record, export_root, client)
        if artifacts:
            record["serving_artifacts"] = artifacts

    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_json_document(active_models_file, payload, config=config)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export serving artifacts for active models")
    parser.add_argument("--registry-path", default="s3://mlflow/registry")
    args = parser.parse_args()
    updated = materialize_active_serving_artifacts(args.registry_path)
    print(json.dumps(updated, indent=2))
