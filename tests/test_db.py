"""Qdrant database tests

Design ref: design-doc.md 5.1
"""

import pytest

from src.db.operations import create_collection, upsert_monster, query_similar
from src.config import settings

@pytest.fixture(scope="module")
def collection_name():
    """unique collection name for test isolation"""
    return "test_monsters"

def test_create_collection(mock_qdrant_client, collection_name):
    """text collection creation"""
    create_collection(mock_qdrant_client, collection_name)

    collections = [c.name for c in mock_qdrant_client.get_collections().collections]
    assert collection_name in collections

def test_upsert_and_query(mock_qdrant_client, collection_name):
    """test upserting a monster, then querying that monster"""
    # create collection
    create_collection(mock_qdrant_client, collection_name, vector_size=4)

    # define test monster data (use integer IDs for in-memory Qdrant compatibility)
    monster_id = 1
    test_vector = [0.1, 0.2, 0.3, 0.4]
    test_payload = {
        "name": "testmander",
        "primary_type": "Fire",
        "base_level": 25,
    }

    # upsert the monster
    upsert_monster(
        mock_qdrant_client,
        monster_id,
        test_vector,
        test_payload,
        collection_name,
    )

    # query the monster
    results = query_similar(
        client=mock_qdrant_client,
        vector=test_vector,
        top_k=5,
        collection_name=collection_name
    )

    assert len(results) > 0
    assert results[0].id == monster_id

def test_query_with_filters(mock_qdrant_client, collection_name):
    """test querying with filters"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # create collection
    create_collection(mock_qdrant_client, collection_name, vector_size=4)

    # upsert the first test monster
    upsert_monster(
        mock_qdrant_client,
        1,
        [1.0, 0.0, 0.0, 0.0],
        {"name": "testmander", "primary_type": "Fire","base_level": 25},
        collection_name,
    )

    # upsert the second test monster
    upsert_monster(
        mock_qdrant_client,
        2,
        [0.0, 1.0, 0.0, 0.0],
        {"name": "testurtle", "primary_type": "Water","base_level": 25},
        collection_name,
    )

    # create a filter for fire type query
    fire_filter = Filter(
        must=[
            FieldCondition(
                key="primary_type",
                match=MatchValue(value="Fire"),
            )
        ]
    )

    # query with filter
    results = query_similar(
            client=mock_qdrant_client,
            vector=[1.0, 0.0, 0.0, 0.0],
            filters=fire_filter,
            top_k=5,
            collection_name=collection_name
    )

    assert len(results) == 1
    assert results[0].id == 1
