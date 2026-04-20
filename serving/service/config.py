import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str
    environment: str
    active_models_path: str
    feedback_store_path: Path
    local_artifact_root: Path
    s3_endpoint_url: str | None
    s3_region: str
    production_logs_bucket: str
    production_logs_prefix: str
    categorization_threshold: float
    categorization_abstention_threshold: float
    trend_anomaly_threshold: float


def load_settings() -> Settings:
    return Settings(
        service_name=os.getenv("SERVING_SERVICE_NAME", "spendsense-serving"),
        environment=os.getenv("SERVING_ENVIRONMENT", "dev"),
        active_models_path=os.getenv(
            "ACTIVE_MODELS_PATH",
            "s3://mlflow/registry/active_models.json",
        ),
        feedback_store_path=Path(
            os.getenv("FEEDBACK_STORE_PATH", "/tmp/spendsense-serving-feedback.jsonl")
        ),
        local_artifact_root=Path(
            os.getenv("LOCAL_ARTIFACT_ROOT", "/tmp/spendsense-serving-artifacts")
        ),
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL") or os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        s3_region=os.getenv("AWS_REGION", "us-east-1"),
        production_logs_bucket=os.getenv("PRODUCTION_LOGS_BUCKET", "production-logs"),
        production_logs_prefix=os.getenv("PRODUCTION_LOGS_PREFIX", "serving"),
        categorization_threshold=float(os.getenv("CATEGORIZATION_THRESHOLD", "0.5")),
        categorization_abstention_threshold=float(
            os.getenv("CATEGORIZATION_ABSTENTION_THRESHOLD", "0.7")
        ),
        trend_anomaly_threshold=float(os.getenv("TREND_ANOMALY_THRESHOLD", "0.7")),
    )
