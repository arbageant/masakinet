"""Pure PyTorch Dataset & DataLoader implementation.

Loads paired (image, metadata) samples from data/raw/ for offline embedding
extraction. Design ref: design-doc.md §5.1 (Deterministic Data Checks —
matrices must match expected shape, e.g. [Batch, 3, 224, 224]).
"""

from pathlib import Path

from torch.utils.data import Dataset


class MonsterDataset(Dataset):
    """Dataset of paired creature images and structured metadata."""

    def __init__(self, data_dir: Path | str = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError
