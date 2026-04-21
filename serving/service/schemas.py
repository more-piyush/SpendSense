from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str
    description: str
    amount: float = 0.0
    currency: str = "USD"
    country: str = "US"
    timestamp: str = ""
    user_id: Optional[str] = None


class CategoryPrediction(BaseModel):
    category: str
    confidence: float


class CategorizationResponse(BaseModel):
    transaction_id: str
    model: str = "categorization"
    model_family: str
    model_version: str
    predicted_categories: List[CategoryPrediction]
    abstained: bool
    abstention_threshold: float
    max_confidence: float
    inference_time_ms: float
    timestamp: str


class TrendFeatures(BaseModel):
    current_spend: float
    rolling_mean_1m: float
    rolling_mean_3m: float
    rolling_mean_6m: float
    rolling_std_3m: float
    rolling_std_6m: float
    deviation_ratio: float
    share_of_wallet: float
    hist_share_of_wallet: float
    txn_count: int
    hist_txn_count_mean: float
    avg_txn_size: float
    hist_avg_txn_size: float
    days_since_last_txn: int
    month_of_year: int
    spending_velocity: float
    weekend_txn_ratio: float
    total_monthly_spend: float
    elevated_cat_count: int
    budget_utilization: float


class TrendInput(BaseModel):
    user_id: str
    category: str
    period: str
    features: TrendFeatures


class TrendAnomalyDetection(BaseModel):
    ensemble_score: float
    xgb_residual_score: float
    isolation_forest_score: float
    is_anomaly: bool
    anomaly_threshold: float


class TrendAnalysis(BaseModel):
    predicted_change_pct: float
    trend_direction: Literal["increasing", "decreasing", "stable"]
    spending_vs_history: Literal["above_average", "below_average", "normal"]
    deviation_from_3m_mean: float


class TrendResponse(BaseModel):
    user_id: str
    category: str
    period: str
    model: str = "trend_detection"
    model_family: str
    model_version: str
    predicted_next_month_spend: float
    anomaly_detection: TrendAnomalyDetection
    trend_analysis: TrendAnalysis
    inference_time_ms: float
    timestamp: str


class FeedbackEvent(BaseModel):
    task: Literal["categorization", "trend_detection"]
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    model_family: str
    model_version: str
    action: Literal["accepted", "overridden", "rejected", "dismissed", "confirmed"]
    predicted_value: Dict[str, Any]
    final_value: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    event_id: str
    timestamp: str


class FireflyTransactionRequest(BaseModel):
    transaction: TransactionInput
    include_trend: bool = False
    trend_payload: Optional[TrendInput] = None


class FireflyTransactionResponse(BaseModel):
    categorization: CategorizationResponse
    trend: Optional[TrendResponse] = None
