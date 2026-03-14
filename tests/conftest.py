from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.embedding import EmbeddingService
from src.main import app


@pytest.fixture()
def mock_embedding_service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the global embedding_service with a mock that returns a 768-dim vector."""
    mock = MagicMock(spec=EmbeddingService)
    mock.is_ready = True
    mock.model_name = "LaBSE"
    mock.encode.return_value = [0.01] * 768

    monkeypatch.setattr("src.main.embedding_service", mock)
    return mock


@pytest.fixture()
def client(mock_embedding_service: MagicMock) -> TestClient:
    """Test client with model pre-loaded (mocked)."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def unloaded_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Test client where the model has NOT been loaded."""
    mock = MagicMock(spec=EmbeddingService)
    mock.is_ready = False
    mock.model_name = ""

    monkeypatch.setattr("src.main.embedding_service", mock)
    return TestClient(app, raise_server_exceptions=False)


def make_embedding_request(**overrides: object) -> dict:
    """Build a valid embedding request payload with optional overrides."""
    payload: dict = {
        "jobId": "550e8400-e29b-41d4-a716-446655440000",
        "version": "1.0",
        "type": "embedding",
        "text": "The professor explains concepts clearly.",
        "metadata": {
            "submissionId": "660e8400-e29b-41d4-a716-446655440001",
            "facultyId": "faculty-001",
            "versionId": "version-001",
        },
        "publishedAt": "2026-03-14T00:00:00.000Z",
    }
    payload.update(overrides)
    return payload
