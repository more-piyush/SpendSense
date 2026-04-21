import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from transformers import DistilBertModel, DistilBertTokenizer

from service.config import Settings

TREND_FEATURE_COLUMNS = [
    "current_spend",
    "rolling_mean_1m",
    "rolling_mean_3m",
    "rolling_mean_6m",
    "rolling_std_3m",
    "rolling_std_6m",
    "deviation_ratio",
    "share_of_wallet",
    "hist_share_of_wallet",
    "txn_count",
    "hist_txn_count_mean",
    "avg_txn_size",
    "hist_avg_txn_size",
    "days_since_last_txn",
    "month_of_year",
    "spending_velocity",
    "weekend_txn_ratio",
    "total_monthly_spend",
    "elevated_cat_count",
    "budget_utilization",
]


class DistilBertCategorizer(nn.Module):
    """Inference replica of the training architecture."""

    def __init__(
        self,
        pretrained_model: str,
        n_classes: int,
        n_currencies: int = 20,
        n_countries: int = 50,
        dropout: float = 0.3,
        freeze_layers: int = 0,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.currency_emb = nn.Embedding(n_currencies + 1, 20, padding_idx=0)
        self.country_emb = nn.Embedding(n_countries + 1, 50, padding_idx=0)

        hidden_dim = 768 + 1 + 20 + 50
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, amount, currency_idx, country_idx):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        currency_feat = self.currency_emb(currency_idx)
        country_feat = self.country_emb(country_idx)
        amount_feat = amount.unsqueeze(1)
        combined = torch.cat(
            [cls_embedding, amount_feat, currency_feat, country_feat], dim=1
        )
        return self.classifier(combined)


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.settings.local_artifact_root.mkdir(parents=True, exist_ok=True)

        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=self._env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self._env("AWS_SECRET_ACCESS_KEY"),
            region_name=settings.s3_region,
        )

        self.active_models: Dict[str, Any] = {}
        self.categorization_model = None
        self.categorization_bundle: Dict[str, Any] = {}
        self.trend_model = None
        self.trend_bundle: Dict[str, Any] = {}

        self._load_models()

    def _env(self, name: str) -> str | None:
        import os

        return os.getenv(name)

    def get_active_models(self) -> Dict[str, Any]:
        return self.active_models

    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {uri}")
        bucket_and_key = uri[5:]
        bucket, key = bucket_and_key.split("/", 1)
        return bucket, key

    def _read_json(self, path: str) -> Dict[str, Any]:
        if path.startswith("s3://"):
            bucket, key = self._parse_s3_uri(path)
            payload = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            return json.loads(payload.decode("utf-8"))
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _download_s3_prefix(self, s3_uri: str) -> Path:
        bucket, prefix = self._parse_s3_uri(s3_uri.rstrip("/"))
        digest = hashlib.sha256(s3_uri.encode("utf-8")).hexdigest()[:16]
        local_root = self.settings.local_artifact_root / digest

        if local_root.exists() and any(local_root.rglob("*")):
            return local_root

        local_root.mkdir(parents=True, exist_ok=True)
        paginator = self.s3.get_paginator("list_objects_v2")
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
                self.s3.download_file(bucket, key, str(destination))

        if not found:
            raise FileNotFoundError(f"No objects found under artifact prefix: {s3_uri}")
        return local_root

    def _materialize_artifact(self, artifact_path: str) -> Path:
        if artifact_path.startswith("s3://"):
            return self._download_s3_prefix(artifact_path)
        return Path(artifact_path)

    def _locate_checkpoint(self, artifact_dir: Path) -> Path:
        for candidate in artifact_dir.rglob("model.pt"):
            return candidate
        raise FileNotFoundError(f"Unable to locate model.pt under {artifact_dir}")

    def _load_json_file(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_onnx_session(self, model_path: Path) -> ort.InferenceSession:
        return ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def _resolve_serving_artifact_dir(self, record: Dict[str, Any]) -> Path | None:
        serving_artifacts = record.get("serving_artifacts", {})
        root_path = serving_artifacts.get("root_path")
        if not root_path:
            return None
        return self._materialize_artifact(root_path)

    def _extract_onnx_scalar(self, raw_output: Any) -> float:
        array = np.asarray(raw_output)
        if array.dtype == object:
            array = np.asarray(list(array), dtype=np.float32)
        return float(array.reshape(-1)[0])

    def _load_categorization_onnx(self, record: Dict[str, Any]):
        artifact_dir = self._resolve_serving_artifact_dir(record)
        if artifact_dir is None:
            raise FileNotFoundError("Categorization ONNX artifact root is not configured")

        serving_artifacts = record.get("serving_artifacts", {})
        model_name = serving_artifacts.get("onnx_quantized_model") or serving_artifacts.get("onnx_model")
        metadata_name = serving_artifacts.get("metadata", "metadata.json")
        tokenizer_dir_name = serving_artifacts.get("tokenizer_dir", "tokenizer")

        metadata = self._load_json_file(artifact_dir / metadata_name)
        tokenizer = DistilBertTokenizer.from_pretrained(str(artifact_dir / tokenizer_dir_name))
        session = self._load_onnx_session(artifact_dir / model_name)

        return session, {
            "format": serving_artifacts.get("preferred_format", "onnx_quantized"),
            "classes": metadata.get("classes", []),
            "tokenizer": tokenizer,
            "max_length": int(metadata.get("max_length", 64)),
            "currency_vocab": metadata.get("currency_vocab", {}),
            "country_vocab": metadata.get("country_vocab", {}),
        }

    def _load_trend_onnx(self, record: Dict[str, Any]):
        artifact_dir = self._resolve_serving_artifact_dir(record)
        if artifact_dir is None:
            raise FileNotFoundError("Trend ONNX artifact root is not configured")

        serving_artifacts = record.get("serving_artifacts", {})
        metadata = self._load_json_file(artifact_dir / serving_artifacts.get("metadata", "metadata.json"))
        session = self._load_onnx_session(artifact_dir / serving_artifacts.get("onnx_model", "model.onnx"))

        isolation_forest = None
        iso_artifact_path = self._derive_isolation_forest_path(record["artifact_path"])
        if iso_artifact_path is not None:
            try:
                iso_dir = self._materialize_artifact(iso_artifact_path)
                isolation_forest = mlflow.sklearn.load_model(str(iso_dir))
            except FileNotFoundError:
                isolation_forest = None

        return session, {
            "format": serving_artifacts.get("preferred_format", "onnx"),
            "metadata": metadata,
            "isolation_forest": isolation_forest,
        }

    def _derive_isolation_forest_path(self, artifact_path: str) -> str | None:
        if not artifact_path.startswith("s3://") or not artifact_path.endswith("/model"):
            return None
        return artifact_path[: -len("/model")] + "/isolation_forest_model"

    def _load_categorization_model(self, record: Dict[str, Any]):
        serving_artifacts = record.get("serving_artifacts", {})
        if serving_artifacts.get("preferred_format") == "onnx_quantized":
            try:
                return self._load_categorization_onnx(record)
            except Exception:
                pass

        artifact_dir = self._materialize_artifact(record["artifact_path"])
        family = record.get("model_family")

        if family == "logistic_regression":
            model = mlflow.sklearn.load_model(str(artifact_dir))
            return model, {"classes": list(model.classes_)}

        if family == "distilbert":
            checkpoint_path = self._locate_checkpoint(artifact_dir)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            config = checkpoint.get("config", {})
            classes = checkpoint.get("classes", [])

            model = DistilBertCategorizer(
                pretrained_model=config.get("pretrained_model", "distilbert-base-uncased"),
                n_classes=checkpoint.get("n_classes", len(classes)),
                dropout=config.get("dropout", 0.3),
                freeze_layers=config.get("freeze_layers", 0),
                freeze_embeddings=config.get("freeze_embeddings", False),
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            tokenizer = DistilBertTokenizer.from_pretrained(
                config.get("pretrained_model", "distilbert-base-uncased")
            )

            return model, {
                "classes": classes,
                "tokenizer": tokenizer,
                "max_length": int(config.get("max_length", 64)),
                "currency_vocab": checkpoint.get("currency_vocab", {}),
                "country_vocab": checkpoint.get("country_vocab", {}),
            }

        raise ValueError(f"Unsupported categorization model_family: {family}")

    def _load_trend_model(self, record: Dict[str, Any]):
        serving_artifacts = record.get("serving_artifacts", {})
        if serving_artifacts.get("preferred_format") == "onnx":
            try:
                return self._load_trend_onnx(record)
            except Exception:
                pass

        artifact_dir = self._materialize_artifact(record["artifact_path"])
        family = record.get("model_family")

        if family in {"xgboost", "xgboost_optuna"}:
            model = mlflow.xgboost.load_model(str(artifact_dir))
        elif family == "random_forest":
            model = mlflow.sklearn.load_model(str(artifact_dir))
        else:
            raise ValueError(f"Unsupported trend model_family: {family}")

        isolation_forest = None
        iso_artifact_path = self._derive_isolation_forest_path(record["artifact_path"])
        if iso_artifact_path is not None:
            try:
                iso_dir = self._materialize_artifact(iso_artifact_path)
                isolation_forest = mlflow.sklearn.load_model(str(iso_dir))
            except FileNotFoundError:
                isolation_forest = None

        return model, {"isolation_forest": isolation_forest}

    def _load_models(self):
        self.active_models = self._read_json(self.settings.active_models_path)

        categorization = self.active_models.get("categorization")
        trend = self.active_models.get("trend")
        if categorization is None or trend is None:
            raise RuntimeError("Both categorization and trend active models must be set")

        self.categorization_model, self.categorization_bundle = self._load_categorization_model(
            categorization
        )
        self.trend_model, self.trend_bundle = self._load_trend_model(trend)

    def predict_categories(
        self,
        text: str,
        amount: float = 0.0,
        currency: str | None = None,
        country: str | None = None,
    ) -> Tuple[List[Dict[str, float]], float]:
        record = self.active_models["categorization"]
        family = record["model_family"]
        serving_format = self.categorization_bundle.get("format")

        if family == "logistic_regression":
            probabilities = self.categorization_model.predict_proba([text])[0]
            classes = self.categorization_bundle["classes"]
            pairs = [
                {"category": category, "confidence": float(score)}
                for category, score in zip(classes, probabilities)
            ]
            pairs.sort(key=lambda item: item["confidence"], reverse=True)
            filtered = [
                item
                for item in pairs
                if item["confidence"] >= self.settings.categorization_threshold
            ]
            return filtered[:5], float(pairs[0]["confidence"])

        if serving_format == "onnx_quantized":
            tokenizer = self.categorization_bundle["tokenizer"]
            encoded = tokenizer(
                [text],
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.categorization_bundle["max_length"],
            )
            currency_vocab = self.categorization_bundle.get("currency_vocab", {})
            country_vocab = self.categorization_bundle.get("country_vocab", {})
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
                "amount": np.array([math.log1p(max(amount, 0.0))], dtype=np.float32),
                "currency_idx": np.array([int(currency_vocab.get(currency or "", 0))], dtype=np.int64),
                "country_idx": np.array([int(country_vocab.get(country or "", 0))], dtype=np.int64),
            }
            logits = np.asarray(self.categorization_model.run(None, inputs)[0], dtype=np.float32)[0]
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            classes = self.categorization_bundle["classes"]
            pairs = [
                {"category": category, "confidence": float(score)}
                for category, score in zip(classes, probabilities)
            ]
            pairs.sort(key=lambda item: item["confidence"], reverse=True)
            filtered = [
                item
                for item in pairs
                if item["confidence"] >= self.settings.categorization_threshold
            ]
            max_confidence = float(pairs[0]["confidence"]) if pairs else 0.0
            return filtered[:5], max_confidence

        tokenizer = self.categorization_bundle["tokenizer"]
        classes = self.categorization_bundle["classes"]
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.categorization_bundle["max_length"],
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        amount_tensor = torch.tensor([math.log1p(max(amount, 0.0))], dtype=torch.float32).to(
            self.device
        )
        currency_vocab = self.categorization_bundle.get("currency_vocab", {})
        country_vocab = self.categorization_bundle.get("country_vocab", {})
        currency_idx = torch.tensor(
            [int(currency_vocab.get(currency or "", 0))],
            dtype=torch.long,
        ).to(self.device)
        country_idx = torch.tensor(
            [int(country_vocab.get(country or "", 0))],
            dtype=torch.long,
        ).to(self.device)

        with torch.no_grad():
            logits = self.categorization_model(
                input_ids,
                attention_mask,
                amount_tensor,
                currency_idx,
                country_idx,
            )
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        pairs = [
            {"category": category, "confidence": float(score)}
            for category, score in zip(classes, probabilities)
        ]
        pairs.sort(key=lambda item: item["confidence"], reverse=True)
        filtered = [
            item
            for item in pairs
            if item["confidence"] >= self.settings.categorization_threshold
        ]
        max_confidence = float(pairs[0]["confidence"]) if pairs else 0.0
        return filtered[:5], max_confidence

    def predict_trend(self, features: Dict[str, float]) -> Dict[str, float | bool | str]:
        ordered = np.array(
            [[float(features.get(column, 0.0)) for column in TREND_FEATURE_COLUMNS]],
            dtype=np.float32,
        )
        if self.trend_bundle.get("format") == "onnx":
            predicted = self._extract_onnx_scalar(
                self.trend_model.run(None, {"features": ordered})[0]
            )
        else:
            predicted = float(self.trend_model.predict(ordered)[0])
        current_spend = float(features.get("current_spend", 0.0))
        rolling_mean_3m = float(features.get("rolling_mean_3m", current_spend))

        deviation_base = max(abs(rolling_mean_3m), 1.0)
        residual_score = min(abs(predicted - current_spend) / deviation_base, 1.0)

        isolation_forest: IsolationForest | None = self.trend_bundle.get("isolation_forest")
        iso_score = 0.0
        if isolation_forest is not None:
            raw = float(-isolation_forest.score_samples(ordered)[0])
            iso_score = 1.0 / (1.0 + math.exp(-raw))

        ensemble_score = residual_score if isolation_forest is None else (0.6 * residual_score + 0.4 * iso_score)
        predicted_change_pct = 0.0
        if abs(rolling_mean_3m) > 1e-8:
            predicted_change_pct = ((predicted - rolling_mean_3m) / abs(rolling_mean_3m)) * 100.0

        if predicted_change_pct > 2.0:
            direction = "increasing"
        elif predicted_change_pct < -2.0:
            direction = "decreasing"
        else:
            direction = "stable"

        if predicted > rolling_mean_3m * 1.05:
            spending_vs_history = "above_average"
        elif predicted < rolling_mean_3m * 0.95:
            spending_vs_history = "below_average"
        else:
            spending_vs_history = "normal"

        return {
            "predicted_next_month_spend": round(predicted, 2),
            "ensemble_score": round(float(ensemble_score), 4),
            "xgb_residual_score": round(float(residual_score), 4),
            "isolation_forest_score": round(float(iso_score), 4),
            "is_anomaly": bool(ensemble_score >= self.settings.trend_anomaly_threshold),
            "predicted_change_pct": round(float(predicted_change_pct), 2),
            "trend_direction": direction,
            "spending_vs_history": spending_vs_history,
            "deviation_from_3m_mean": round(float(predicted - rolling_mean_3m), 2),
        }
