"""Pydantic response schemas."""
from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """Single prediction response."""

    sentiment: str = Field(..., description="Predicted sentiment label.")
    label_id: int = Field(..., description="Numeric label ID.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Max class probability."
    )
    probabilities: dict[str, float] | None = Field(
        default=None, description="Per-class probabilities (only if requested)."
    )
    processing_time_ms: float = Field(
        ..., description="Total inference time in ms."
    )
    model_version: str = Field(..., description="Model version used.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sentiment": "positive",
                "label_id": 1,
                "confidence": 0.9847,
                "probabilities": {"negative": 0.0153, "positive": 0.9847},
                "processing_time_ms": 42.3,
                "model_version": "distilbert-sentiment-v1",
            }
        }
    }


class BatchResultItem(BaseModel):
    """Single item in a batch prediction response."""

    index: int
    text_preview: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float] | None = None


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    results: list[BatchResultItem]
    total: int
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str
    model: str | None = None
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    base_model: str
    dataset: str
    labels: list[str]
    test_accuracy: float | None = None
    test_f1_macro: float | None = None
    training_date: str | None = None
    wandb_run_url: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
