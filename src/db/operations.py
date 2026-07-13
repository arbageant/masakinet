"""Methods for creating collections, querying, & payloads.

Design ref: design-doc.md §2.1 — combines vector search (Cosine/Dot Product)
with metadata payload filters (e.g., creature type, level).
"""

def create_collection() -> None:
    """Create the Qdrant collection for monster embeddings if it doesn't exist."""
    raise NotImplementedError


def upsert_monster(monster_id: str, vector: list[float], payload: dict) -> None:
    """Insert or update a single monster's embedding + metadata payload."""
    raise NotImplementedError


def query_similar(vector: list[float], filters: dict | None = None, top_k: int = 10) -> list:
    """Query the collection for nearest neighbors, optionally filtered by payload."""
    raise NotImplementedError
