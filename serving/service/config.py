from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    categorization_tokenizer_path: str


def load_settings():
    return Settings(
        categorization_tokenizer_path=os.getenv(
            "CATEGORIZATION_TOKENIZER_PATH",
            "distilbert-base-uncased",
        )
    )
