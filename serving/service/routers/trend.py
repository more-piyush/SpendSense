import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from service.dependencies import get_metrics_registry
from service.metrics import MetricsRegistry
from service.schemas import TrendAnomalyDetection, TrendAnalysis, TrendInput, TrendResponse

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/trend", response_model=TrendResponse)
def predict_trend(
    payload: TrendInput,
    metrics: MetricsRegistry = Depends(get_metrics_registry),
) -> TrendResponse:
    started = time.perf_counter()

    anomaly = TrendAnomalyDetection(
        ensemble_score=0.34,
        xgb_residual_score=0.28,
        isolation_forest_score=0.43,
        is_anomaly=False,
        anomaly_threshold=0.7,
    )
    analysis = TrendAnalysis(
        predicted_change_pct=3.68,
        trend_direction="increasing",
        spending_vs_history="above_average",
        deviation_from_3m_mean=46.70,
    )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics.record_prediction("trend_detection", latency_ms, False)

    return TrendResponse(
        user_id=payload.user_id,
        category=payload.category,
        period=payload.period,
        model_family="xgboost_optuna",
        model_version="1.0.0",
        predicted_next_month_spend=355.12,
        anomaly_detection=anomaly,
        trend_analysis=analysis,
        inference_time_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
