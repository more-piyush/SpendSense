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
from service.schemas import CategorizationResponse, CategoryPrediction, TransactionInput
from service.storage import EventStore

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/categorization", response_model=CategorizationResponse)
def predict_categorization(
    tx: TransactionInput,
    manager: ModelManager = Depends(get_model_manager),
    metrics: MetricsRegistry = Depends(get_metrics_registry),
    settings=Depends(get_settings),
    store: EventStore = Depends(get_feedback_store),
) -> CategorizationResponse:
    started = time.perf_counter()

    predicted_raw, max_confidence = manager.predict_categories(
        tx.description,
        amount=tx.amount,
        currency=tx.currency,
        country=tx.country,
    )
    abstained = max_confidence < settings.categorization_abstention_threshold
    predicted = [
        CategoryPrediction(category=item["category"], confidence=round(item["confidence"], 4))
        for item in predicted_raw
    ]

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics.record_prediction("categorization", latency_ms, abstained)

    active = manager.get_active_models()["categorization"]
    response = CategorizationResponse(
        transaction_id=tx.transaction_id,
        model_family=active["model_family"],
        model_version=active["version"],
        predicted_categories=[] if abstained else predicted,
        abstained=abstained,
        abstention_threshold=settings.categorization_abstention_threshold,
        max_confidence=round(max_confidence, 4),
        inference_time_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    store.write(
        {
            "task": "categorization",
            "transaction": tx.model_dump(),
            "active_model": {
                "registry_id": active["registry_id"],
                "model_id": active["model_id"],
                "model_family": active["model_family"],
                "version": active["version"],
            },
            "response": response.model_dump(),
        },
        event_type="interactions/categorization",
    )
    return response
