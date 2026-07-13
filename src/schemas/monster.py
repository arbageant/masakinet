"""Pydantic data contracts for creature assets.

Design ref: design-doc.md §4.1 (Data Modeling & Validation). Inputs are
validated against this contract prior to feature extraction to prevent
silent failures or downstream degradation in the vector space.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class MonsterMetadata(BaseModel):
    monster_id: str = Field(..., description="Unique alphanumeric identifier.")
    name: str = Field(..., description="Name of the creature asset.")
    description: str = Field(
        ...,
        max_length=1000,
        description="Detailed text (~100 words) describing behaviors, physical traits, and lore.",
    )
    primary_type: str = Field(..., description="Categorical type element (e.g., Fire, Water, Electric).")
    secondary_type: Optional[str] = None
    base_level: int = Field(..., ge=1, le=100)
    abilities: List[str]
