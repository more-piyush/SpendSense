from datetime import datetime, timezone

from fastapi import APIRouter

from service.schemas import (
    CategoryPrediction,
    FireflyTransactionRequest,
    FireflyTransactionResponse,
    CategorizationResponse,
    TrendAnomalyDetection,
    TrendAnalysis,
    TrendResponse,
)

router = APIRouter(prefix="/firefly", tags=["firefly"])


@router.post("/transactions/enrich", response_model=FireflyTransactionResponse)
def enrich_transaction(request: FireflyTransactionRequest) -> FireflyTransactionResponse:
    categorization = CategorizationResponse(
        transaction_id=request.transaction.transaction_id,
        model_family="logistic_regression",
        model_version="1.1.0",
        predicted_categories=[CategoryPrediction(category="Shopping", confidence=0.91)],
        abstained=False,
        abstention_threshold=0.7,
        max_confidence=0.91,
        inference_time_ms=1.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    trend = None
    if request.include_trend and request.trend_payload is not None:
        trend = TrendResponse(
            user_id=request.trend_payload.user_id,
            category=request.trend_payload.category,
            period=request.trend_payload.period,
            model_family="xgboost_optuna",
            model_version="1.0.0",
            predicted_next_month_spend=355.12,
            anomaly_detection=TrendAnomalyDetection(
                ensemble_score=0.34,
                xgb_residual_score=0.28,
                isolation_forest_score=0.43,
                is_anomaly=False,
                anomaly_threshold=0.7,
            ),
            trend_analysis=TrendAnalysis(
                predicted_change_pct=3.68,
                trend_direction="increasing",
                spending_vs_history="above_average",
                deviation_from_3m_mean=46.70,
            ),
            inference_time_ms=1.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    return FireflyTransactionResponse(
        categorization=categorization,
        trend=trend,
    )
