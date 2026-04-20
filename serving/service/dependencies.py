from functools import lru_cache
from service.config import load_settings
from service.model_loader import ModelManager

@lru_cache
def get_settings():
    return load_settings()

@lru_cache
def get_model_manager():
    return ModelManager(get_settings())
