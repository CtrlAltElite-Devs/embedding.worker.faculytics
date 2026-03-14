from fastapi.testclient import TestClient


def test_health_ready(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "LaBSE"


def test_health_unavailable(unloaded_client: TestClient) -> None:
    response = unloaded_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["model"] is None
