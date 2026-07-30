"""Script to run batch embedding creation & DB seeding.

Design ref: design-doc.md §5.2 Weeks 3-4 — extract embeddings from raw
paired creature assets and load them into the vector DB (and eventually
an S3/GCS bucket for the raw source assets).

Usage:
    poetry run python -m src.pipeline.ingest
"""

import io

from PIL import Image

from src.db.client import get_qdrant_client
from src.db.operations import create_collection, upsert_monster
from src.pipeline.dataset import MonsterDataset
from src.services.embeddings import EmbeddingService


def run_ingest() -> None:
    """Batch-embed the raw dataset and upsert named vectors + payloads into Qdrant."""
    client = get_qdrant_client()
    embedder = EmbeddingService()
    dataset = MonsterDataset()

    create_collection(client)

    for image, metadata in dataset:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        image_vector = embedder.embed_image(image_bytes)

        caption = embedder.create_clip_caption(metadata.model_dump())
        text_vector = embedder.embed_text(caption)

        payload = metadata.model_dump()

        upsert_monster(
            client,
            metadata.monster_id,
            {"image": image_vector, "text": text_vector},
            payload,
        )

    print(f"Ingested {len(dataset)} monsters into Qdrant.")


if __name__ == "__main__":
    run_ingest()
