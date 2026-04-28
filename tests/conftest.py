"""Shared test fixtures with mocked model — no GPU required."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from app.core.model import ModelBundle


def make_mock_bundle(version="v1"):
    """Create a minimal ModelBundle mock that returns deterministic predictions."""
    bundle = MagicMock(spec=ModelBundle)
    bundle.model_name = f"distilbert-sentiment-{version}"
    bundle.model_path = f"./models/distilbert-sentiment-{version}"
    bundle.labels = {0: "negative", 1: "positive"}
    bundle.device = torch.device("cpu")

    # Mock tokenizer: dynamically sized for single or batch inputs
    def _mock_tokenize(text_or_texts, **kwargs):
        n = len(text_or_texts) if isinstance(text_or_texts, list) else 1
        return {
            "input_ids": torch.zeros(n, 128, dtype=torch.long),
            "attention_mask": torch.ones(n, 128, dtype=torch.long),
        }

    mock_tokenizer = MagicMock(side_effect=_mock_tokenize)
    bundle.tokenizer = mock_tokenizer

    # Mock model: returns batch-sized logits favoring "positive"
    def _mock_forward(**kwargs):
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.logits = torch.tensor([[0.1, 2.5]] * n)
        return out

    bundle.model = MagicMock(side_effect=_mock_forward)

    return bundle


def _create_test_app(bundle_v1=None, model_loaded=True):
    """Create a FastAPI app with no-op lifespan for testing."""
    from app.main import create_app

    @asynccontextmanager
    async def _noop_lifespan(app):
        app.state.bundle_v1 = bundle_v1
        app.state.bundle_v2 = None
        app.state.model_loaded = model_loaded
        yield

    app = create_app()
    app.router.lifespan_context = _noop_lifespan
    return app


@pytest.fixture
def mock_bundle():
    return make_mock_bundle("v1")


@pytest.fixture
def client(mock_bundle):
    """TestClient with mocked model loaded into app state."""
    app = _create_test_app(bundle_v1=mock_bundle, model_loaded=True)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model():
    """TestClient without model loaded — tests 503 responses."""
    app = _create_test_app(bundle_v1=None, model_loaded=False)
    with TestClient(app) as c:
        yield c
