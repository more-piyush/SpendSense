"""Pydantic models that validate every row from production-logs.

Any row that fails validation is rejected in Step 2 of the pipeline.
These models are the machine-enforceable version of SCHEMA_CONTRACT.md.
"""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import config


# ─────────────────────────────────────────────────────────────────────────────
# Categorization event
# ─────────────────────────────────────────────────────────────────────────────
class CategorizationEvent(BaseModel):
    """One inference-time prediction event written by the serving service."""

    event_id: str = Field(..., min_length=8)
    timestamp: datetime
    user_id: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    transaction_description: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0)
    currency: str = Field(..., pattern=r"^[A-Z]{3}$")
    country: str = Field(..., pattern=r"^[A-Z]{2}$")
    predicted_categories: List[str] = Field(..., min_length=1)
    prediction_probabilities: List[float]
    model_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    user_action: Literal["accepted", "overridden", "abstained", "ignored"]
    final_category: Optional[str] = None

    @field_validator("prediction_probabilities")
    @classmethod
    def probs_in_range(cls, v):
        if any(p < 0.0 or p > 1.0 for p in v):
            raise ValueError("prediction_probabilities values must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def check_lengths_and_final_category(self):
        if len(self.predicted_categories) != len(self.prediction_probabilities):
            raise ValueError(
                "predicted_categories and prediction_probabilities must be same length"
            )
        if self.user_action == "ignored":
            if self.final_category is not None:
                raise ValueError("final_category must be null when user_action='ignored'")
        else:
            if not self.final_category:
                raise ValueError(
                    f"final_category required when user_action='{self.user_action}'"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly feedback event
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyFeedbackEvent(BaseModel):
    """One anomaly alert + (optional) user feedback."""

    event_id: str = Field(..., min_length=8)
    timestamp: datetime
    user_id: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    category: str = Field(..., min_length=1)
    month: str = Field(..., pattern=r"^\d{4}-\d{2}$")
    feature_vector: dict
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    predicted_spend: float
    actual_spend: float
    direction: Literal[-1, 1]
    magnitude_pct: float
    alert_severity: Literal["low", "medium", "high"]
    top_factors: Optional[List[str]] = Field(default=None, max_length=3)
    user_feedback: Optional[Literal["helpful", "not_useful", "expected"]] = None
    model_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")

    @field_validator("feature_vector")
    @classmethod
    def vector_has_all_20_keys(cls, v):
        missing = set(config.ANOMALY_FEATURE_KEYS) - set(v.keys())
        if missing:
            raise ValueError(f"feature_vector missing keys: {sorted(missing)}")
        return v
