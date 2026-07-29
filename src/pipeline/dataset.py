"""Pure PyTorch Dataset & DataLoader implementation.

Loads paired (image, metadata) samples from data/raw/ for offline embedding
extraction. Design ref: design-doc.md §5.1 (Deterministic Data Checks —
matrices must match expected shape, e.g. [Batch, 3, 224, 224]).
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.schemas.monster import MonsterMetadata


class MonsterDataset(Dataset):
    """Dataset of paired creature images and structured metadata."""

    def __init__(
        self,
        data_dir: Path | str = "data/raw",
        metadata_file: Path | str | None = None,
        default_level: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)

        if metadata_file is None:
            metadata_file = self.data_dir / "monster_metadata.json"
        metadata_path = Path(metadata_file)

        with open(metadata_path) as f:
            raw_records = json.load(f)

        self.records = [
            MonsterMetadata(
                monster_id=rec["name"],
                monster_source="pokemon",
                name=rec["name"].title(),
                description=(rec.get("biology") or rec.get("flavor_text", ""))[:1000],
                primary_type=rec["types"][0].capitalize(),
                secondary_type=rec["types"][1].capitalize()
                if len(rec["types"]) > 1
                else None,
                base_level=default_level,
                abilities=[],
            )
            for rec in raw_records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, MonsterMetadata]:
        record = self.records[index]
        image_path = self.data_dir / f"{record.name.lower()}.png"
        image = Image.open(image_path).convert("RGB")
        return image, record
