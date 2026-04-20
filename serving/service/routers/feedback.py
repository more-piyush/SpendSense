import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from service.dependencies import get_feedback_store, get_metrics_registry
from service.metrics import MetricsRegistry
from service.schemas import FeedbackEvent, FeedbackResponse
from service.storage import JsonLineStore

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def record_feedback(
    event: FeedbackEvent,
    metrics: MetricsRegistry = Depends(get_metrics_registry),
    store: JsonLineStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    event_id = f"fb_{uuid.uuid4().hex[:16]}"
    stored_timestamp = datetime.now(timezone.utc).isoformat()

    metrics.record_feedback(event.task, event.action)
    store.write(
        {
            "event_id": event_id,
            "recorded_at": stored_timestamp,
            "feedback": event.model_dump(),
        }
    )

    return FeedbackResponse(
        status="recorded",
        event_id=event_id,
        timestamp=stored_timestamp,
    )
