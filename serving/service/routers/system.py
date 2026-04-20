from fastapi import APIRouter, Depends
from service.dependencies import get_model_manager

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/ready")
def ready(manager = Depends(get_model_manager)):
    models = manager.get_active_models()
    return {
        "status": "ready",
        "active_models_source": "active_models.json",
        "categorization_model": models["categorization"]["model_id"],
        "categorization_registry_id": models["categorization"]["registry_id"],
        "trend_model": models["trend"]["model_id"],
        "trend_registry_id": models["trend"]["registry_id"],
    }
