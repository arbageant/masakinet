"""Handles CLIP/SigLIP text & image ONNX inference.

Design ref: design-doc.md §2.1, §3 — dual-encoder contrastive model
producing joint text/image embeddings for the vector index.
"""

from src.config import settings


class EmbeddingService:
    """Wraps a CLIP/SigLIP dual encoder for text and image embedding."""

    def __init__(self, model_name: str = settings.clip_model_name) -> None:
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        """Project a text string into the shared embedding space."""
        raise NotImplementedError

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Project raw image bytes into the shared embedding space."""
        raise NotImplementedError
