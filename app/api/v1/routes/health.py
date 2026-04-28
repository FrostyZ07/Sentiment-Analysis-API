"""Health and readiness check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.response import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Check",
    description=(
        "Confirms the service process is running. "
        "Use for load balancer liveness probes."
    ),
)
async def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description=(
        "Confirms the model is loaded and ready to serve predictions. "
        "Returns 503 if model is not loaded."
    ),
)
async def ready(request: Request):
    loaded = getattr(request.app.state, "model_loaded", False)
    bundle = getattr(request.app.state, "bundle_v1", None)

    if loaded and bundle:
        return ReadinessResponse(
            status="ready",
            model=bundle.model_name,
            model_loaded=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    else:
        return JSONResponse(
            status_code=503,
            content=ReadinessResponse(
                status="not_ready",
                model=None,
                model_loaded=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )
