from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query to search for")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    primary_type: Optional[str] = Field(default=None, description="Optional filter by primary type")
    secondary_type: Optional[str] = Field(default=None, description="Optional filter by secondary type")
    monster_source: Optional[str] = Field(default=None, description="Optional filter by source universe")


class SearchResultItem(BaseModel):
    monster_id: str
    name: str
    description: str
    primary_type: str
    secondary_type: Optional[str] = None
    base_level: int
    abilities: List[str]
    monster_source: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
