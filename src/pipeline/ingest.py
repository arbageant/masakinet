"""Script to run batch embedding creation & DB seeding.

Design ref: design-doc.md §5.2 Weeks 3-4 — extract embeddings from raw
paired creature assets and load them into the vector DB (and eventually
an S3/GCS bucket for the raw source assets).

Usage (once implemented):
    poetry run python -m src.pipeline.ingest
"""

from torch.utils.data import DataLoader

from src.pipeline.dataset import MonsterDataset


def run_ingest() -> None:
    """Batch-embed the raw dataset and upsert vectors + payloads into Qdrant."""
    dataset = MonsterDataset()
    _loader = DataLoader(dataset, batch_size=32)
    raise NotImplementedError


if __name__ == "__main__":
    run_ingest()
