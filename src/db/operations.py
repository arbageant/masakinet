"""Methods for creating collections, querying, & payloads.

Design ref: design-doc.md §2.1 — combines vector search (Cosine/Dot Product)
with metadata payload filters (e.g., creature type, level).
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from src.config import settings

def create_collection(
        client: QdrantClient,
        collection_name: str = settings.qdrant_collection_name,
        vector_size: int = 512, # configured for CLIP ViT-B/32
) -> None:
    """Create the Qdrant collection with image and text named vectors."""
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": VectorParams(size=vector_size, distance=Distance.COSINE),
                "text": VectorParams(size=vector_size, distance=Distance.COSINE),
            },
        )


def upsert_monster(
        client: QdrantClient,
        monster_id: int | str | uuid.UUID,
        vectors: dict[str, list[float]],
        payload: dict,
        collection_name: str = settings.qdrant_collection_name,
) -> None:
    """Insert or update a single monster's named embeddings + metadata payload."""
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=monster_id,
                vector=vectors,
                payload=payload,
            )
        ],
    )


def query_similar(
        client: QdrantClient,
        vector: list[float],
        vector_name: str = "image",
        filters: Filter | None = None,
        top_k: int = 10,
        collection_name: str = settings.qdrant_collection_name,
) -> list:
    """Query the collection for nearest neighbors against a named vector, optionally filtered by payload."""
    return client.query_points(
        collection_name=collection_name,
        query=vector,
        using=vector_name,
        query_filter=filters,
        limit=top_k,
    ).points
