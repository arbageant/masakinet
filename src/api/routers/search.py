"""Handles /search/text and /search/image endpoints.

Design ref: design-doc.md §2.1 (Multimodal Search Pipeline).
Text and image queries are embedded via src.services.embeddings and matched
against Qdrant via src.db.operations.
"""

from fastapi import APIRouter, Depends, File, Query, UploadFile
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.db.client import get_qdrant_client
from src.db.operations import query_similar
from src.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from src.services.embeddings import EmbeddingService

router = APIRouter(prefix="/search", tags=["search"])


def _build_filter(request: SearchRequest) -> Filter | None:
    conditions = []
    if request.primary_type is not None:
        conditions.append(
            FieldCondition(key="primary_type", match=MatchValue(value=request.primary_type))
        )
    if request.secondary_type is not None:
        conditions.append(
            FieldCondition(key="secondary_type", match=MatchValue(value=request.secondary_type))
        )
    if request.monster_source is not None:
        conditions.append(
            FieldCondition(key="monster_source", match=MatchValue(value=request.monster_source))
        )
    return Filter(must=conditions) if conditions else None


@router.post("/text", response_model=SearchResponse)
def search_by_text(
    body: SearchRequest,
    client: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    embedder = EmbeddingService()
    vector = embedder.embed_text(body.query)
    qdrant_filter = _build_filter(body)
    results = query_similar(
        client, vector, vector_name="text", filters=qdrant_filter, top_k=body.top_k
    )
    return SearchResponse(
        results=[
            SearchResultItem(
                monster_id=p.id if isinstance(p.id, str) else str(p.id),
                name=p.payload["name"],
                description=p.payload.get("description", ""),
                primary_type=p.payload["primary_type"],
                secondary_type=p.payload.get("secondary_type"),
                base_level=p.payload["base_level"],
                abilities=p.payload.get("abilities", []),
                monster_source=p.payload["monster_source"],
                score=p.score,
            )
            for p in results
        ]
    )


@router.post("/image", response_model=SearchResponse)
def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=100),
    client: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    image_bytes = file.read()
    embedder = EmbeddingService()
    vector = embedder.embed_image(image_bytes)
    results = query_similar(client, vector, vector_name="image", top_k=top_k)
    return SearchResponse(
        results=[
            SearchResultItem(
                monster_id=p.id if isinstance(p.id, str) else str(p.id),
                name=p.payload["name"],
                description=p.payload.get("description", ""),
                primary_type=p.payload["primary_type"],
                secondary_type=p.payload.get("secondary_type"),
                base_level=p.payload["base_level"],
                abilities=p.payload.get("abilities", []),
                monster_source=p.payload["monster_source"],
                score=p.score,
            )
            for p in results
        ]
    )
