import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from service.dependencies import (
    get_feedback_store,
    get_metrics_registry,
    get_model_manager,
    get_settings,
)
from service.metrics import MetricsRegistry
from service.model_loader import ModelManager
from service.schemas import TrendAnomalyDetection, TrendAnalysis, TrendInput, TrendResponse
from service.storage import EventStore

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/trend", response_model=TrendResponse)
def predict_trend(
    payload: TrendInput,
    manager: ModelManager = Depends(get_model_manager),
    metrics: MetricsRegistry = Depends(get_metrics_registry),
    settings=Depends(get_settings),
    store: EventStore = Depends(get_feedback_store),
) -> TrendResponse:
    started = time.perf_counter()

    prediction = manager.predict_trend(payload.features.model_dump())

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics.record_prediction(
        "trend_detection",
        latency_ms,
        bool(prediction["is_anomaly"]),
    )

    anomaly = TrendAnomalyDetection(
        ensemble_score=prediction["ensemble_score"],
        xgb_residual_score=prediction["xgb_residual_score"],
        isolation_forest_score=prediction["isolation_forest_score"],
        is_anomaly=prediction["is_anomaly"],
        anomaly_threshold=settings.trend_anomaly_threshold,
    )
    analysis = TrendAnalysis(
        predicted_change_pct=prediction["predicted_change_pct"],
        trend_direction=prediction["trend_direction"],
        spending_vs_history=prediction["spending_vs_history"],
        deviation_from_3m_mean=prediction["deviation_from_3m_mean"],
    )

    active = manager.get_active_models()["trend"]
    response = TrendResponse(
        user_id=payload.user_id,
        category=payload.category,
        period=payload.period,
        model_family=active["model_family"],
        model_version=active["version"],
        predicted_next_month_spend=prediction["predicted_next_month_spend"],
        anomaly_detection=anomaly,
        trend_analysis=analysis,
        inference_time_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    store.write(
        {
            "task": "trend_detection",
            "request": payload.model_dump(),
            "active_model": {
                "registry_id": active["registry_id"],
                "model_id": active["model_id"],
                "model_family": active["model_family"],
                "version": active["version"],
            },
            "response": response.model_dump(),
        },
        event_type="interactions/trend",
    )
    return response
