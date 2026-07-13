"""Manages Stable Diffusion + LORA pixel generation.

Design ref: design-doc.md §2.2 — maps text features to a discrete pixel
grid (e.g., 32x32 or 64x64) via a lightweight diffusion model or custom
pixel-art generator.
"""


class GeneratorService:
    """Wraps a diffusion (or custom) model for text-to-pixel-art generation."""

    def generate(self, prompt: str, tags: list[str] | None = None) -> bytes:
        """Generate a pixel-art image from a text prompt and optional structured tags."""
        raise NotImplementedError
