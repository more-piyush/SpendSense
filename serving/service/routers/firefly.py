from service.routers.categorization import predict_categorization
from service.routers.trend import predict_trend

from fastapi import APIRouter, Depends

from service.dependencies import (
    get_feedback_store,
    get_metrics_registry,
    get_model_manager,
    get_settings,
)
from service.metrics import MetricsRegistry
from service.model_loader import ModelManager
from service.schemas import FireflyTransactionRequest, FireflyTransactionResponse
from service.storage import EventStore

router = APIRouter(prefix="/firefly", tags=["firefly"])


@router.post("/transactions/enrich", response_model=FireflyTransactionResponse)
def enrich_transaction(
    request: FireflyTransactionRequest,
    manager: ModelManager = Depends(get_model_manager),
    metrics: MetricsRegistry = Depends(get_metrics_registry),
    settings=Depends(get_settings),
    store: EventStore = Depends(get_feedback_store),
) -> FireflyTransactionResponse:
    categorization = predict_categorization(
        request.transaction,
        manager=manager,
        metrics=metrics,
        settings=settings,
        store=store,
    )

    trend = None
    if request.include_trend and request.trend_payload is not None:
        trend = predict_trend(
            request.trend_payload,
            manager=manager,
            metrics=metrics,
            settings=settings,
            store=store,
        )

    return FireflyTransactionResponse(
        categorization=categorization,
        trend=trend,
    )
