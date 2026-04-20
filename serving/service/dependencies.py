from functools import lru_cache

from service.config import Settings, load_settings
from service.metrics import MetricsRegistry
from service.storage import JsonLineStore


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_metrics_registry() -> MetricsRegistry:
    return MetricsRegistry()


@lru_cache
def get_prediction_store() -> JsonLineStore:
    settings = get_settings()
    return JsonLineStore(settings.data_dir / "predictions.jsonl")


@lru_cache
def get_feedback_store() -> JsonLineStore:
    settings = get_settings()
    return JsonLineStore(settings.data_dir / "feedback.jsonl")
