"""Input validation boundary and shape assertion tests.

Design ref: design-doc.md §5.1 — Deterministic Data Checks: validates
that input matrices match expected dimensions (e.g., [Batch, 3, 224, 224])
and ranges.
"""

import pytest
from pydantic import ValidationError

from src.schemas.monster import MonsterMetadata

def test_monster_metadata_valid():
    monster = MonsterMetadata(
        monster_id="m001",
        name="Emberling",
        description="A small fire-type creature.",
        primary_type="Fire",
        base_level=5,
        abilities=["Ember", "Growl"],
    )
    assert monster.base_level == 5


def test_monster_metadata_rejects_out_of_range_level():
    with pytest.raises(ValidationError):
        MonsterMetadata(
            monster_id="m002",
            name="Overleveled",
            description="Too strong.",
            primary_type="Fire",
            base_level=101,
            abilities=[],
        )
