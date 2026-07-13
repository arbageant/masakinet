"""Handles /generate/text and /generate/image extensions.

Design ref: design-doc.md §2.2 (Generative Bidirectional Extension Pipeline).
Text-to-image goes through src.services.generator; image-to-text/metadata
goes through src.services.captioner.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/image")
def generate_image() -> None:
    """Generate pixel-art from a text prompt + structured tags."""
    raise NotImplementedError


@router.post("/text")
def generate_text() -> None:
    """Caption an uploaded image into description text + structured metadata."""
    raise NotImplementedError
