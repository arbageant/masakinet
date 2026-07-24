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
