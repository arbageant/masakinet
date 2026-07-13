"""Connects PaliGemma/Moondream for image-to-text.

Design ref: design-doc.md §2.2 — decodes pixel matrices into a natural
language description (~100 words) and a strict JSON schema of categorical
attributes (src.schemas.monster.MonsterMetadata).
"""

from src.schemas.monster import MonsterMetadata


class CaptionerService:
    """Wraps a Vision-Language Model for image-to-text/metadata captioning."""

    def caption(self, image_bytes: bytes) -> MonsterMetadata:
        """Generate a description and structured metadata for a creature image."""
        raise NotImplementedError
