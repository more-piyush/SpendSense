"""Pydantic models that validate rows from production-logs.

Serving writes two event streams:
  - interactions/{categorization,trend}  → inference events (monitoring only)
  - feedback/{categorization,trend}      → labeled events (used for retraining)

This pipeline reads the feedback/* events. The inference/* events don't carry
the user's correction, so they can't be used to retrain supervised models.

The trend feedback envelope is provisional — it mirrors the categorization
envelope we have a real sample of. Tighten once a real feedback/trend sample
is available.
"""
from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config


# `model_*` fields trigger a protected-namespace warning in pydantic v2.
# Our serving team already logs these names — opt out of the namespace check.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


# ─────────────────────────────────────────────────────────────────────────────
# Categorization feedback event (confirmed against real log sample)
# ─────────────────────────────────────────────────────────────────────────────
class CategorizationPredictedValue(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    category: str = Field(..., min_length=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CategorizationFinalValue(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    category: str = Field(..., min_length=1)


class CategorizationMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    source: Optional[str] = None
    description: str = Field(..., min_length=1)
    amount: str                                         # logged as string/number, coerced below
    currency: str = Field(..., pattern=r"^[A-Z]{3}$")
    country: Optional[str] = None                       # absent in current logs
    feedback_origin: Optional[str] = None

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, value: Any) -> str:
        if value is None:
            raise ValueError("amount is required")
        return str(value)

    @field_validator("country", mode="before")
    @classmethod
    def coerce_country(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)


class CategorizationFeedback(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    task: Literal["categorization"]
    transaction_id: str = Field(..., min_length=1)
    user_id: Optional[str] = None                       # can be null if unlinked
    model_family: str = Field(..., min_length=1)
    model_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    action: Literal[
        "accepted", "overridden", "abstained", "ignored",
        "dismissed", "confirmed", "rejected"
    ]
    predicted_value: Optional[CategorizationPredictedValue] = None
    final_value: Optional[CategorizationFinalValue] = None
    metadata: CategorizationMetadata
    timestamp: datetime

    @field_validator("transaction_id", "user_id", "model_family", "model_version", mode="before")
    @classmethod
    def coerce_required_strings(cls, value: Any) -> str:
        if value is None:
            raise ValueError("required string field is null")
        value = str(value)
        if value == "":
            raise ValueError("required string field is empty")
        return value

    @model_validator(mode="after")
    def check_final_value(self):
        if self.action in {"ignored", "dismissed", "rejected"}:
            if self.final_value is not None:
                raise ValueError("final_value must be null when action is ignored/dismissed/rejected")
        else:
            if self.final_value is None:
                raise ValueError(
                    f"final_value required when action='{self.action}'"
                )
        return self


class CategorizationFeedbackEvent(BaseModel):
    event_id: str = Field(..., min_length=1)
    recorded_at: datetime
    event_type: Literal["feedback/categorization"]
    feedback: CategorizationFeedback


# ─────────────────────────────────────────────────────────────────────────────
# Trend feedback event (PROVISIONAL — verify against real log sample)
# ─────────────────────────────────────────────────────────────────────────────
class TrendPredictedValue(BaseModel):
    predicted_next_month_spend: float
    anomaly_detection: dict = Field(default_factory=dict)
    trend_analysis: dict = Field(default_factory=dict)


class TrendFinalValue(BaseModel):
    user_feedback: Literal["helpful", "not_useful", "expected"]
    actual_next_month_spend: Optional[float] = None


class TrendFeedback(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    task: Literal["trend_detection"]
    user_id: Optional[str] = None
    category: str = Field(..., min_length=1)
    period: str = Field(..., pattern=r"^\d{4}-\d{2}$")
    model_family: str = Field(..., min_length=1)
    model_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    action: Literal["helpful", "not_useful", "expected", "ignored"]
    features: dict
    predicted_value: Optional[TrendPredictedValue] = None
    final_value: Optional[TrendFinalValue] = None
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime

    @field_validator("user_id", mode="before")
    @classmethod
    def user_id_to_str(cls, v):
        return None if v is None else str(v)

    @field_validator("features")
    @classmethod
    def features_has_20_keys(cls, v):
        missing = set(config.ANOMALY_FEATURE_KEYS) - set(v.keys())
        if missing:
            raise ValueError(f"features missing keys: {sorted(missing)}")
        return v

    @model_validator(mode="after")
    def check_final_value(self):
        if self.action == "ignored":
            if self.final_value is not None:
                raise ValueError("final_value must be null when action='ignored'")
        else:
            if self.final_value is None:
                raise ValueError(
                    f"final_value required when action='{self.action}'"
                )
        return self


class TrendFeedbackEvent(BaseModel):
    event_id: str = Field(..., min_length=1)
    recorded_at: datetime
    event_type: Literal["feedback/trend"]
    feedback: TrendFeedback
