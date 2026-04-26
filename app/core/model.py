"""Model loading and management."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Holds model, tokenizer and metadata together."""

    model: DistilBertForSequenceClassification
    tokenizer: DistilBertTokenizer
    model_name: str
    model_path: str
    device: torch.device
    labels: dict[int, str]


def load_model(model_path: str, version_tag: str = "v1") -> ModelBundle:
    """
    Load fine-tuned DistilBERT model from local path or HuggingFace Hub.

    Args:
        model_path: Local directory path or HF Hub model ID.
        version_tag: Label for this model version (e.g., "v1", "v2").

    Returns:
        ModelBundle with loaded model and tokenizer.

    Raises:
        RuntimeError: If model files are not found.
    """
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from '{model_path}' on {device}...")

    # Validate local path exists
    path = Path(model_path)
    if path.exists() and not path.is_dir():
        raise RuntimeError(
            f"MODEL_PATH '{model_path}' exists but is not a directory."
        )
    if path.exists() and not (path / "config.json").exists():
        raise RuntimeError(
            f"No config.json found in '{model_path}'. "
            "Is this a valid model directory?"
        )

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from '{model_path}': {e}"
        ) from e

    elapsed = time.perf_counter() - start
    labels = {int(k): v for k, v in model.config.id2label.items()}

    logger.info(
        f"Model loaded successfully in {elapsed:.2f}s "
        f"| version={version_tag} | labels={labels} | device={device}"
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=f"distilbert-sentiment-{version_tag}",
        model_path=model_path,
        device=device,
        labels=labels,
    )
