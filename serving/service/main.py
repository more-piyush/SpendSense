from fastapi import FastAPI

from service.routers.categorization import router as categorization_router
from service.routers.trend import router as trend_router
from service.routers.firefly import router as firefly_router
from service.routers.feedback import router as feedback_router
from service.routers.system import router as system_router

app = FastAPI(
    title="SpendSense Unified Serving",
    version="1.0.0",
    description="Unified serving API for categorization and trend detection.",
)

app.include_router(system_router)
app.include_router(categorization_router)
app.include_router(trend_router)
app.include_router(firefly_router)
app.include_router(feedback_router)
