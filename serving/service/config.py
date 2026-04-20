import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str
    environment: str
    active_models_path: Path
    categorization_tokenizer_path: str


def load_settings() -> Settings:
    return Settings(
        service_name=os.getenv("SERVING_SERVICE_NAME", "spendsense-serving"),
        environment=os.getenv("SERVING_ENVIRONMENT", "dev"),
        active_models_path=Path(
            os.getenv(
                "ACTIVE_MODELS_PATH",
                "/app/model_registry/active_models.json",
            )
        ),
        categorization_tokenizer_path=os.getenv(
            "CATEGORIZATION_TOKENIZER_PATH",
            "distilbert-base-uncased",
        ),
    )
