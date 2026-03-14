from datetime import datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from tests.conftest import make_embedding_request


def test_success_response_shape(client: TestClient) -> None:
    """Response must match the analysisResultSchema with status='completed'."""
    payload = make_embedding_request()
    response = client.post("/embeddings", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["jobId"] == payload["jobId"]
    assert body["version"] == "1.0"
    assert body["status"] == "completed"
    assert "result" in body
    assert "completedAt" in body


def test_embedding_dimension(client: TestClient) -> None:
    """Embedding must be a 768-dimensional float array."""
    payload = make_embedding_request()
    response = client.post("/embeddings", json=payload)
    body = response.json()

    embedding = body["result"]["embedding"]
    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(v, float) for v in embedding)


def test_model_name_in_result(client: TestClient) -> None:
    """Result must include modelName."""
    payload = make_embedding_request()
    response = client.post("/embeddings", json=payload)
    body = response.json()

    assert body["result"]["modelName"] == "LaBSE"


def test_job_id_echo(client: TestClient) -> None:
    """Response jobId must echo the request jobId."""
    job_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    payload = make_embedding_request(jobId=job_id)
    response = client.post("/embeddings", json=payload)
    body = response.json()

    assert body["jobId"] == job_id


def test_completed_at_is_iso_format(client: TestClient) -> None:
    """completedAt must be a valid ISO 8601 datetime."""
    payload = make_embedding_request()
    response = client.post("/embeddings", json=payload)
    body = response.json()

    completed_at = body["completedAt"]
    parsed = datetime.fromisoformat(completed_at)
    assert parsed.tzinfo is not None or "Z" in completed_at or "+" in completed_at


def test_error_response_on_failure(client: TestClient, mock_embedding_service: MagicMock) -> None:
    """Domain errors must return HTTP 200 with status='failed'."""
    mock_embedding_service.encode.side_effect = RuntimeError("Model crashed")

    payload = make_embedding_request()
    response = client.post("/embeddings", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "failed"
    assert body["jobId"] == payload["jobId"]
    assert "error" in body
    assert "Model crashed" in body["error"]
    assert "completedAt" in body
    assert "result" not in body


def test_validation_error_on_empty_text(client: TestClient) -> None:
    """Request with empty text must be rejected (Pydantic validation)."""
    payload = make_embedding_request(text="")
    response = client.post("/embeddings", json=payload)

    assert response.status_code == 422


def test_validation_error_on_missing_fields(client: TestClient) -> None:
    """Request missing required fields must be rejected."""
    response = client.post("/embeddings", json={"text": "hello"})

    assert response.status_code == 422
