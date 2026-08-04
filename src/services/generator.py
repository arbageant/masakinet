"""Manages deterministic, CPU-only pixel-art generation for text prompts.

Design ref: design-doc.md §2.2 — maps text features (a prompt plus
optional structured tags) to a discrete pixel grid (32x32 by default) via
a custom lightweight pixel-art generator. No diffusion weights or GPU are
required; output is fully deterministic given a prompt and seed, which
keeps the model testable offline (see design-doc.md §5.1).
"""

import hashlib
import io
from typing import Final

import numpy as np
from PIL import Image

from src.config import settings


class GeneratorService:
    """Wraps a custom procedural pixel-art generator for text-to-image."""

    def __init__(
        self, grid_size: int = settings.generation_grid_size, seed: int | None = None
    ) -> None:
        self.grid_size = grid_size
        self.seed = seed

    def _build_seed(self, prompt: str, tags: list[str]) -> int:
        """Derive a stable integer seed from prompt, tags, and optional override."""
        material = "|".join([prompt, *tags])
        hashed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
        if self.seed is not None:
            composite = (hashed * 31 + self.seed) % (2**63)
            return composite
        return hashed

    @staticmethod
    def _select_palette(tags: list[str], seed: int) -> tuple[tuple[int, int, int], ...]:
        """Map type tags to a fixed palette; fall back to a seeded palette otherwise."""
        palettes: Final[dict[str, tuple[tuple[int, int, int], ...]]] = {
            "fire": ((34, 20, 20), (216, 62, 42), (255, 150, 41), (255, 224, 102)),
            "water": ((18, 30, 48), (48, 98, 190), (96, 156, 232), (200, 228, 255)),
            "electric": ((30, 28, 12), (230, 176, 32), (255, 226, 96), (255, 250, 200)),
            "grass": ((20, 36, 20), (74, 148, 62), (140, 198, 78), (220, 240, 180)),
            "ground": ((34, 28, 20), (148, 106, 70), (196, 148, 92), (240, 214, 160)),
            "psychic": ((34, 20, 34), (176, 74, 150), (226, 122, 192), (250, 210, 240)),
            "ice": ((24, 32, 38), (110, 176, 208), (182, 224, 244), (240, 252, 255)),
            "dark": ((22, 22, 26), (62, 62, 76), (110, 110, 132), (200, 200, 216)),
        }
        lowered = [tag.lower() for tag in tags]
        body = next((palettes[k] for k in palettes if k in " ".join(lowered)), None)
        if body is not None:
            return body

        rng = np.random.default_rng(seed)
        keep = rng.integers(0, 256, size=(3,)).tolist()
        warm = max(30, int((keep[0] + 40) % 256))
        cool = max(30, int((keep[1] + 40) % 256))
        bg = (int(keep[0]) // 8, int(keep[1]) // 8, int(keep[2]) // 8)
        body = (warm, int(keep[1]) % 256, cool)
        dark = tuple(max(0, c - 80) for c in body)
        light = tuple(min(255, c + 80) for c in body)
        return (bg, body, dark, light)

    def _render(self, prompt: str, tags: list[str], seed: int) -> Image.Image:
        """Render a symmetric blocky creature onto a pixel grid and return a PIL image."""
        n = self.grid_size
        bg, body, outline, accent = self._select_palette(tags, seed)
        rng = np.random.default_rng(seed)

        # Base RGB canvas filled with the background color.
        canvas = np.empty((n, n, 3), dtype=np.uint8)
        canvas[..., 0], canvas[..., 1], canvas[..., 2] = bg

        # Centered, slightly low silhouette; use seeded harmonics for an organic blob.
        cx = (n - 1) / 2.0
        cy = n * 0.56
        rx = n * 0.28
        ry = n * 0.36
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.06, 0.18)

        ys, xs = np.mgrid[0:n, 0:n].astype(float)
        nx = (xs - cx) / rx
        ny = (ys - cy) / ry

        # Symmetric perturbation: evaluate a signed radial field and mirror columns.
        field = nx**2 + ny**2 - 1 + amp * np.sin(3 * nx + phase)
        half = np.arange(n)
        mirrored = field[:, half] * 1.0
        if n % 2 == 0:
            right = half[len(half) // 2 :]
            mirrored[:, right] = mirrored[:, right[::-1]]
        field = np.minimum(field, mirrored)

        body_mask = field <= 0
        canvas[body_mask] = body

        # Thin outline where a body pixel borders empty space.
        outlined = np.zeros((n, n), dtype=bool)
        if body_mask.any():
            eroded = np.zeros_like(body_mask)
            eroded[1:-1, 1:-1] = (
                body_mask[1:-1, 1:-1]
                & body_mask[:-2, 1:-1]
                & body_mask[2:, 1:-1]
                & body_mask[1:-1, :-2]
                & body_mask[1:-1, 2:]
            )
            outlined = body_mask & ~eroded
        canvas[outlined] = outline

        # Mirrored eyes sitting on the upper body.
        eye_offsets = [-int(rx * 0.4), int(rx * 0.4)]
        eye_y = int(cy - ry * 0.25)
        eye_size = max(1, n // 16)
        for off in eye_offsets:
            ex = int(cx + off)
            canvas[
                eye_y : eye_y + eye_size, ex : ex + eye_size
            ] = accent

        return Image.fromarray(canvas, mode="RGB")

    def generate(self, prompt: str, tags: list[str] | None = None) -> bytes:
        """Generate a PNG-encoded pixel-art image from a text prompt and tags."""
        tag_list = list(tags or [])
        seed = self._build_seed(prompt, tag_list)
        image = self._render(prompt, tag_list, seed)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()