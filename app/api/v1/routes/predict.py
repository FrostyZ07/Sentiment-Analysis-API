"""Single prediction and model info endpoints."""

import random
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.schemas.request import PredictRequest
from app.schemas.response import ErrorResponse, ModelInfoResponse, PredictResponse
from app.services.inference import predict_single

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_bundle(request: Request, version: str = "v1"):
    """Resolve the correct ModelBundle based on version param."""
    if version == "ab":
        version = random.choice(["v1", "v2"])

    if version == "v2":
        bundle = getattr(request.app.state, "bundle_v2", None)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Model v2 not loaded.")
        return bundle

    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "MODEL_NOT_READY",
                    "message": "Model is not loaded yet.",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
    return bundle


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"description": "Validation Error", "model": ErrorResponse},
        429: {"description": "Rate Limit Exceeded", "model": ErrorResponse},
        503: {"description": "Model Not Ready", "model": ErrorResponse},
    },
    summary="Predict Sentiment",
    description=(
        "Classify a single product review as **positive** or **negative**. "
        "Returns confidence score and optional per-class probabilities."
    ),
)
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def predict(
    request: Request,
    body: PredictRequest,
):
    bundle = get_bundle(request, body.version)
    result = predict_single(
        text=body.text,
        bundle=bundle,
        return_probabilities=body.return_probabilities,
    )
    return PredictResponse(**result)


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Returns metadata about the currently loaded model.",
)
async def model_info(request: Request):
    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return ModelInfoResponse(
        model_name=bundle.model_name,
        base_model="distilbert-base-uncased",
        dataset="amazon_polarity (HuggingFace)",
        labels=list(bundle.labels.values()),
        test_accuracy=None,
        test_f1_macro=None,
        training_date=None,
        wandb_run_url=None,
    )
