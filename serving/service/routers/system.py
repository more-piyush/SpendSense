from fastapi import APIRouter, Depends

from service.dependencies import get_metrics_registry
from service.metrics import MetricsRegistry

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "spendsense-serving"}


@router.get("/ready")
def ready():
    return {
        "status": "ready",
        "active_categorization_model": "cat_logreg_baseline",
        "active_trend_model": "trend_xgb_optuna",
    }


@router.get("/metrics/summary")
def metrics_summary(
    metrics: MetricsRegistry = Depends(get_metrics_registry),
):
    return metrics.snapshot()
