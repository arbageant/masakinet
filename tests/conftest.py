"""Pytest fixtures (mock db, mock server test client)."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client for integration checks."""
    return TestClient(app)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for unit tests (uses in-memory storage)."""
    return QdrantClient(":memory:")


@pytest.fixture
def real_qdrant_client():
    """Real Qdrant client connecting to localhost Docker container.

    Skips the test if Qdrant is not reachable.
    """
    client = QdrantClient(host="localhost", port=6333)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant is not running — start with: docker compose -f docker/docker-compose.yml up -d")
    yield client
    client.close()
