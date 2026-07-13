"""Handles connection pooling to the Qdrant cluster.

Design ref: design-doc.md §3 (Vector Storage: Qdrant).
"""

from functools import lru_cache

from qdrant_client import QdrantClient

from src.config import settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a process-wide singleton Qdrant client."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
