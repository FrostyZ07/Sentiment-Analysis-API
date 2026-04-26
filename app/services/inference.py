"""Core inference logic — tokenisation and model forward pass."""
import logging
import re
import time

import torch
import torch.nn.functional as F

from app.core.metrics import (
    PROMETHEUS_AVAILABLE,
    confidence_histogram,
    inference_duration,
    sentiment_counter,
)
from app.core.model import ModelBundle

logger = logging.getLogger(__name__)

MAX_LENGTH = 128


def clean_text(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _record_metrics(result: dict, bundle: ModelBundle, elapsed_ms: float) -> None:
    """Record Prometheus metrics if available."""
    if not PROMETHEUS_AVAILABLE:
        return
    sentiment_counter.labels(
        label=result["sentiment"],
        model_version=bundle.model_name,
    ).inc()
    confidence_histogram.observe(result["confidence"])
    inference_duration.observe(elapsed_ms / 1000)


def predict_single(
    text: str,
    bundle: ModelBundle,
    return_probabilities: bool = False,
) -> dict:
    """
    Run inference on a single text string.

    Args:
        text: Raw review text (will be cleaned internally).
        bundle: Loaded ModelBundle.
        return_probabilities: Whether to return per-class probabilities.

    Returns:
        dict with sentiment, label_id, confidence, probabilities, processing_time_ms.
    """
    start = time.perf_counter()

    cleaned = clean_text(text)

    inputs = bundle.tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bundle.model(**inputs)

    logits = outputs.logits  # shape: (1, num_labels)
    probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: (num_labels,)

    label_id = int(probs.argmax().item())
    confidence = float(probs[label_id].item())
    sentiment = bundle.labels[label_id]

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "sentiment": sentiment,
        "label_id": label_id,
        "confidence": round(confidence, 6),
        "probabilities": None,
        "processing_time_ms": round(elapsed_ms, 2),
        "model_version": bundle.model_name,
    }

    if return_probabilities:
        result["probabilities"] = {
            bundle.labels[i]: round(float(probs[i].item()), 6)
            for i in range(len(bundle.labels))
        }

    _record_metrics(result, bundle, elapsed_ms)

    return result


def predict_batch(
    texts: list[str],
    bundle: ModelBundle,
    return_probabilities: bool = False,
    batch_size: int = 16,
) -> list[dict]:
    """
    Run inference on a list of texts using micro-batching.

    Args:
        texts: List of raw review strings.
        bundle: Loaded ModelBundle.
        return_probabilities: Include per-class probs in output.
        batch_size: Internal micro-batch size for GPU efficiency.

    Returns:
        List of prediction dicts (same order as input).
    """
    cleaned_texts = [clean_text(t) for t in texts]
    all_results = []

    for i in range(0, len(cleaned_texts), batch_size):
        micro_batch = cleaned_texts[i : i + batch_size]

        inputs = bundle.tokenizer(
            micro_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bundle.model(**inputs)

        probs_batch = F.softmax(outputs.logits, dim=-1)  # (batch, num_labels)

        for j, probs in enumerate(probs_batch):
            label_id = int(probs.argmax().item())
            confidence = float(probs[label_id].item())
            sentiment = bundle.labels[label_id]

            item = {
                "sentiment": sentiment,
                "label_id": label_id,
                "confidence": round(confidence, 6),
                "probabilities": None,
            }
            if return_probabilities:
                item["probabilities"] = {
                    bundle.labels[k]: round(float(probs[k].item()), 6)
                    for k in range(len(bundle.labels))
                }
            all_results.append(item)

    return all_results
