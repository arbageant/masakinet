"""Handles /generate/text and /generate/image extensions.

Design ref: design-doc.md §2.2 (Generative Bidirectional Extension Pipeline).
Text-to-image goes through src.services.generator; image-to-text/metadata
goes through src.services.captioner.
"""

from fastapi import APIRouter
from fastapi import Body, Response

from src.services.generator import GeneratorService

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/image")
def generate_image(
    prompt: str = Body(..., min_length=1),
    tags: list[str] | None = Body(default=None),
) -> Response:
    """Generate pixel-art PNG bytes from a text prompt + optional structured tags."""
    png = GeneratorService().generate(prompt, tags)
    return Response(content=png, media_type="image/png")


@router.post("/text")
def generate_text() -> None:
    """Caption an uploaded image into description text + structured metadata."""
    raise NotImplementedError
