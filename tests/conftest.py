"""Pytest fixtures (mock db, mock server test client)."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client for integration checks."""
    return TestClient(app)


@pytest.fixture
def mock_qdrant_client():
    """Placeholder fixture for a mocked Qdrant client, to be filled in with
    a fake/in-memory client once src.db.client is implemented."""
    raise NotImplementedError
