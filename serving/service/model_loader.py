import numpy as np
import onnxruntime as ort
import joblib
from transformers import DistilBertTokenizerFast
from pathlib import Path

from service.artifact_store import MinioStore


class ModelManager:
    def __init__(self, settings):
        self.settings = settings
        self.store = MinioStore()

        # 🔥 ALWAYS FRESH
        self.active = self.store.get_json("active_models.json")
        self.registry = self.store.get_json("registry.json")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            settings.categorization_tokenizer_path
        )

        self.categorization_model = self._load_categorization()
        self.trend_model = self._load_trend()

    # -------------------------
    # RESOLVE MODEL PATH
    # -------------------------
    def _resolve(self, model_id: str):
        return self.registry.get(model_id)

    # -------------------------
    # LOAD CATEGORIZATION (GPU)
    # -------------------------
    def _load_categorization(self):
        model_id = self.active["active_categorization_model"]
        s3_key = self._resolve(model_id)

        local = f"/tmp/models/{model_id}/model.onnx"
        self.store.download(s3_key, local)

        return ort.InferenceSession(
            local,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    # -------------------------
    # LOAD TREND MODEL
    # -------------------------
    def _load_trend(self):
        model_id = self.active["active_trend_model"]
        s3_key = self._resolve(model_id)

        local = f"/tmp/models/{model_id}/model.joblib"
        self.store.download(s3_key, local)

        return joblib.load(local)

    # -------------------------
    # ACTIVE MODELS (SOURCE OF TRUTH)
    # -------------------------
    def get_active_models(self):
        return self.active

    def get_serving_metadata(self):
        return {
            "categorization": {
                "active_model": self.active["active_categorization_model"],
                "serving_backend": "onnx_gpu",
            },
            "trend_detection": {
                "active_model": self.active["active_trend_model"],
                "serving_backend": "xgboost_gpu",
            },
        }
