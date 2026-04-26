"""Tests for single prediction endpoint."""

VALID_REVIEW = "This is a great product that I really enjoyed using."


def test_predict_returns_200_for_valid_input(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    data = response.json()
    assert "sentiment" in data
    assert "label_id" in data
    assert "confidence" in data
    assert "processing_time_ms" in data
    assert "model_version" in data
    assert data["sentiment"] in ("positive", "negative")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_with_probabilities(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW, "return_probabilities": True},
    )
    data = response.json()
    assert data["probabilities"] is not None
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
    # Probabilities should sum to ~1.0
    total = sum(data["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_predict_without_probabilities_by_default(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert response.json()["probabilities"] is None


def test_predict_422_for_text_too_short(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "hi"},
    )
    assert response.status_code == 422


def test_predict_422_for_empty_text(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "   "},
    )
    assert response.status_code == 422


def test_predict_422_for_text_too_long(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": "x" * 2001},
    )
    assert response.status_code == 422


def test_predict_422_for_missing_text(client):
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422


def test_predict_503_when_model_not_loaded(client_no_model):
    response = client_no_model.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert response.status_code == 503


def test_predict_html_stripped_from_input(client):
    """HTML in text should not crash inference."""
    response = client.post(
        "/api/v1/predict",
        json={"text": "<b>Amazing product!</b> Really loved it."},
    )
    assert response.status_code == 200


def test_predict_response_has_model_version(client):
    response = client.post(
        "/api/v1/predict",
        json={"text": VALID_REVIEW},
    )
    assert "distilbert-sentiment" in response.json()["model_version"]
