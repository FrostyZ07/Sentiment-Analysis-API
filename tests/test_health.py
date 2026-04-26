"""Tests for health and readiness endpoints."""


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_health_has_request_id_header(client):
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_ready_returns_200_when_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_ready_returns_503_when_model_not_loaded(client_no_model):
    response = client_no_model.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["model_loaded"] is False
