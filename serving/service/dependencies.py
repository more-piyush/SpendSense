from functools import lru_cache

from service.config import load_settings
from service.metrics import MetricsRegistry
from service.model_loader import ModelManager
from service.storage import JsonLineStore


@lru_cache
def get_settings():
    return load_settings()


@lru_cache
def get_model_manager():
    return ModelManager(get_settings())


@lru_cache
def get_metrics_registry():
    return MetricsRegistry()


@lru_cache
def get_feedback_store():
    return JsonLineStore(get_settings().feedback_store_path)
