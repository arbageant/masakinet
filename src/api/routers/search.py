"""Handles /search/text and /search/image endpoints.

Design ref: design-doc.md §2.1 (Multimodal Search Pipeline).
Text and image queries are embedded via src.services.embeddings and matched
against Qdrant via src.db.operations.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/text")
def search_by_text() -> None:
    """Embed a text query and return nearest-neighbor monster matches."""
    raise NotImplementedError


@router.post("/image")
def search_by_image() -> None:
    """Embed an uploaded image query and return nearest-neighbor monster matches."""
    raise NotImplementedError
