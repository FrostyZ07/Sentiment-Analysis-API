"""Tests for batch prediction endpoint."""

REVIEWS = [
    "Excellent product, very happy with my purchase.",
    "Complete garbage. Do not buy this.",
    "It is okay, nothing special.",
]


def test_batch_predict_returns_200(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    assert response.status_code == 200


def test_batch_predict_response_length_matches_input(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    data = response.json()
    assert data["total"] == len(REVIEWS)
    assert len(data["results"]) == len(REVIEWS)


def test_batch_predict_index_order_preserved(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    results = response.json()["results"]
    for i, r in enumerate(results):
        assert r["index"] == i


def test_batch_422_empty_list(client):
    response = client.post("/api/v1/batch", json={"texts": []})
    assert response.status_code == 422


def test_batch_422_more_than_32_items(client):
    texts = ["Some review text here."] * 33
    response = client.post("/api/v1/batch", json={"texts": texts})
    assert response.status_code == 422


def test_batch_422_item_too_short(client):
    response = client.post("/api/v1/batch", json={"texts": ["ok", "great product"]})
    assert response.status_code == 422


def test_batch_has_processing_time(client):
    response = client.post("/api/v1/batch", json={"texts": REVIEWS})
    assert "processing_time_ms" in response.json()
    assert response.json()["processing_time_ms"] > 0


def test_batch_503_no_model(client_no_model):
    response = client_no_model.post("/api/v1/batch", json={"texts": REVIEWS})
    assert response.status_code == 503
