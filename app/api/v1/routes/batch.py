"""Batch prediction endpoint."""
import time

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.schemas.request import BatchPredictRequest
from app.schemas.response import BatchPredictResponse, BatchResultItem
from app.services.inference import predict_batch

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch Predict Sentiment",
    description=(
        "Classify up to **32 reviews** in a single request. "
        "Uses micro-batching internally for GPU efficiency."
    ),
)
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def batch_predict(
    request: Request,
    body: BatchPredictRequest,
):
    bundle = getattr(request.app.state, "bundle_v1", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.perf_counter()

    raw_results = predict_batch(
        texts=body.texts,
        bundle=bundle,
        return_probabilities=body.return_probabilities,
    )

    results = [
        BatchResultItem(
            index=i,
            text_preview=(
                (body.texts[i][:50] + "...")
                if len(body.texts[i]) > 50
                else body.texts[i]
            ),
            sentiment=r["sentiment"],
            confidence=r["confidence"],
            probabilities=r.get("probabilities"),
        )
        for i, r in enumerate(raw_results)
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchPredictResponse(
        results=results,
        total=len(results),
        processing_time_ms=round(elapsed_ms, 2),
        model_version=bundle.model_name,
    )
