"""Deterministic pixel-art generator contract checks.

Design ref: design-doc.md §5.1 — deterministic data checks: inputs map to
expected output matrices (grid dimensions, value ranges) and remain
reproducible across identical prompts.
"""

import io
from collections import Counter

from PIL import Image

from src.services.generator import GeneratorService


def test_generate_returns_png_bytes():
    result = GeneratorService(grid_size=32).generate("a fire creature", ["fire"])

    assert isinstance(result, bytes)
    image = Image.open(io.BytesIO(result))
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_custom_grid_size():
    result = GeneratorService(grid_size=16).generate("a small creature")

    image = Image.open(io.BytesIO(result))
    assert image.size == (16, 16)


def test_pixel_values_in_byte_range():
    result = GeneratorService(grid_size=32).generate("a creature", ["water"])

    image = Image.open(io.BytesIO(result)).convert("RGB")
    pixels = [image.getpixel((x, y)) for y in range(image.height) for x in range(image.width)]
    image.close()
    assert pixels
    assert all(0 <= c <= 255 for pixel in pixels for c in pixel)
    assert all(0 <= c <= 255 for pixel in pixels for c in pixel)


def test_generation_is_deterministic():
    service = GeneratorService(grid_size=32)
    first = service.generate("a thunder monster", ["electric"])
    second = service.generate("a thunder monster", ["electric"])

    assert first == second


def test_distinct_prompts_differ():
    service = GeneratorService(grid_size=32)
    a = service.generate("a fire monster", ["fire"])
    b = service.generate("a water monster", ["water"])

    assert a != b


def test_tag_selects_palette():
    """Fire and water tags should produce visually distinct body palettes."""
    fire = GeneratorService(grid_size=32).generate("a monster", ["fire"])
    water = GeneratorService(grid_size=32).generate("a monster", ["water"])

    def dominant_rgb(png: bytes) -> tuple[int, int, int]:
        image = Image.open(io.BytesIO(png)).convert("RGB")
        pixels = [image.getpixel((x, y)) for y in range(image.height) for x in range(image.width)]
        image.close()
        return Counter(pixels).most_common(1)[0][0]

    assert dominant_rgb(fire) != dominant_rgb(water)


def test_seed_override_changes_output():
    base = GeneratorService(grid_size=32, seed=1).generate("a monster", ["fire"])
    other = GeneratorService(grid_size=32, seed=2).generate("a monster", ["fire"])

    assert base != other