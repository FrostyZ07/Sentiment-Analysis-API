"""Pydantic request schemas."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single review prediction request."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Product review text to classify.",
        json_schema_extra={"example": "This product exceeded all my expectations!"},
    )
    return_probabilities: bool = Field(
        default=False,
        description="If true, include per-class probability scores in response.",
    )
    version: str = Field(
        default="v1",
        description="Model version to use: 'v1', 'v2', or 'ab' for random A/B routing.",
        pattern="^(v1|v2|ab)$",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v.strip()


class BatchPredictRequest(BaseModel):
    """Batch review prediction request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of review texts (1–32 items).",
    )
    return_probabilities: bool = Field(default=False)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        cleaned = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"texts[{i}] must be a string")
            t = t.strip()
            if len(t) < 3:
                raise ValueError(f"texts[{i}] must be at least 3 characters")
            if len(t) > 2000:
                raise ValueError(f"texts[{i}] must not exceed 2000 characters")
            cleaned.append(t)
        return cleaned
