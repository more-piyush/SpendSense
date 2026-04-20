import os
from dataclasses import dataclass
from pathlib import Path


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


@dataclass(frozen=True)
class Settings:
    service_name: str
    environment: str
    log_dir: Path
    data_dir: Path
    inference_provider: str
    categorization_registry_path: str
    trend_registry_path: str
    categorization_abstention_threshold: float
    trend_anomaly_threshold: float


def load_settings() -> Settings:
    log_dir = Path(_env("SERVING_LOG_DIR", "/tmp/spendsense-serving/logs"))
    data_dir = Path(_env("SERVING_DATA_DIR", "/tmp/spendsense-serving/data"))
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        service_name=_env("SERVING_SERVICE_NAME", "spendsense-serving"),
        environment=_env("SERVING_ENVIRONMENT", "dev"),
        log_dir=log_dir,
        data_dir=data_dir,
        inference_provider=_env("INFERENCE_PROVIDER", "cuda"),
        categorization_registry_path=_env("CATEGORIZATION_REGISTRY_PATH", ""),
        trend_registry_path=_env("TREND_REGISTRY_PATH", ""),
        categorization_abstention_threshold=_env_float("CATEGORIZATION_ABSTENTION_THRESHOLD", 0.7),
        trend_anomaly_threshold=_env_float("TREND_ANOMALY_THRESHOLD", 0.7),
    )
