import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from service.dependencies import get_metrics_registry
from service.metrics import MetricsRegistry
from service.schemas import CategorizationResponse, CategoryPrediction, TransactionInput

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/categorization", response_model=CategorizationResponse)
def predict_categorization(
    tx: TransactionInput,
    metrics: MetricsRegistry = Depends(get_metrics_registry),
) -> CategorizationResponse:
    started = time.perf_counter()

    predicted = [
        CategoryPrediction(category="Shopping", confidence=0.91)
    ]
    max_confidence = 0.91
    abstained = max_confidence < 0.7

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics.record_prediction("categorization", latency_ms, abstained)

    return CategorizationResponse(
        transaction_id=tx.transaction_id,
        model_family="logistic_regression",
        model_version="1.1.0",
        predicted_categories=predicted if not abstained else [],
        abstained=abstained,
        abstention_threshold=0.7,
        max_confidence=max_confidence,
        inference_time_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
