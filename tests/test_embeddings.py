import io
import math
import pytest
from PIL import Image

from src.services.embeddings import EmbeddingService


@pytest.fixture(scope="module")
def service():
    """Instantiate the embedding service once for the test module."""
    return EmbeddingService()


@pytest.fixture
def sample_image_bytes():
    """Generate a simple red square image in memory as PNG bytes."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def calculate_dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


# --- Unit & Contract Tests ---

def test_embed_text_contract(service: EmbeddingService):
    text = "A fire-type Pokémon named Magmar."
    embedding = service.embed_text(text)

    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    assert len(embedding) > 0


def test_embed_image_contract(service: EmbeddingService, sample_image_bytes: bytes):
    embedding = service.embed_image(sample_image_bytes)

    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    assert len(embedding) > 0


def test_embedding_dimensions_match(service: EmbeddingService, sample_image_bytes: bytes):
    """Text and image embeddings must produce vectors of equal dimensionality."""
    text_emb = service.embed_text("sample text")
    img_emb = service.embed_image(sample_image_bytes)

    assert len(text_emb) == len(img_emb)


def test_l2_normalization(service: EmbeddingService, sample_image_bytes: bytes):
    """Ensure vectors are normalized to unit length (L2 norm ≈ 1.0)."""
    text_emb = service.embed_text("a red fire monster")
    img_emb = service.embed_image(sample_image_bytes)

    text_norm = math.sqrt(sum(x ** 2 for x in text_emb))
    img_norm = math.sqrt(sum(x ** 2 for x in img_emb))

    assert math.isclose(text_norm, 1.0, abs_tol=1e-4)
    assert math.isclose(img_norm, 1.0, abs_tol=1e-4)


# --- Integration & Semantic Sanity Test ---

def test_cross_modal_alignment(service: EmbeddingService, sample_image_bytes: bytes):
    """
    A solid red image should score higher against 'red square'
    than against 'blue circle'.
    """
    img_emb = service.embed_image(sample_image_bytes)

    pos_text_emb = service.embed_text("a solid red square")
    neg_text_emb = service.embed_text("a solid blue circle")

    pos_score = calculate_dot_product(img_emb, pos_text_emb)
    neg_score = calculate_dot_product(img_emb, neg_text_emb)

    assert pos_score > neg_score