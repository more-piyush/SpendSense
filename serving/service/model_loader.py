import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import boto3
import joblib
from transformers import DistilBertTokenizerFast

from service.config import Settings


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            settings.categorization_tokenizer_path
        )

        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        self.categorization_model = None
        self.trend_model = None

        self._load_models()

    # -------------------------
    # READ ACTIVE MODEL JSON
    # -------------------------
    def get_active_models(self) -> Dict:
        with open(self.settings.active_models_path, "r") as f:
            return json.load(f)

    # -------------------------
    # DOWNLOAD FROM S3/MinIO
    # -------------------------
    def _download(self, s3_path: str) -> str:
        """
        s3://bucket/key -> local temp file
        """
        assert s3_path.startswith("s3://")

        path = s3_path.replace("s3://", "")
        bucket, key = path.split("/", 1)

        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.s3.download_file(bucket, key, tmp_file.name)

        return tmp_file.name

    # -------------------------
    # LOAD MODELS
    # -------------------------
    def _load_models(self):
        data = self.get_active_models()

        # ---- Categorization ----
        cat_path = data["categorization"]["artifact_path"]
        cat_local = self._download(cat_path)

        self.categorization_model = ort.InferenceSession(
            cat_local,
            providers=["CUDAExecutionProvider"],
        )

        # ---- Trend ----
        trend_path = data["trend"]["artifact_path"]
        trend_local = self._download(trend_path)

        self.trend_model = joblib.load(trend_local)

    # -------------------------
    # INFERENCE: CATEGORIZATION
    # -------------------------
    def predict_categories(self, text: str):
        encoded = self.tokenizer(
            [text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=64,
        )

        outputs = self.categorization_model.run(
            None,
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )

        scores = 1 / (1 + np.exp(-outputs[0][0]))
        return scores.tolist()

    # -------------------------
    # INFERENCE: TREND
    # -------------------------
    def predict_trend(self, features: Dict[str, float]):
        return self.trend_model.predict([list(features.values())])[0]
