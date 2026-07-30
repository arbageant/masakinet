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
    test_payload = {
        "name": "testmander",
        "primary_type": "Fire",
        "base_level": 25,
    }

    # upsert the monster with named vectors
    upsert_monster(
        mock_qdrant_client,
        monster_id,
        {"image": [0.1, 0.2, 0.3, 0.4], "text": [0.5, 0.6, 0.7, 0.8]},
        test_payload,
        collection_name,
    )

    # query against the image vector
    results = query_similar(
        client=mock_qdrant_client,
        vector=[0.1, 0.2, 0.3, 0.4],
        vector_name="image",
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
        {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 0.0, 0.0, 1.0]},
        {"name": "testmander", "primary_type": "Fire","base_level": 25},
        collection_name,
    )

    # upsert the second test monster
    upsert_monster(
        mock_qdrant_client,
        2,
        {"image": [0.0, 1.0, 0.0, 0.0], "text": [0.0, 0.0, 1.0, 0.0]},
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

    # query with filter against image vector
    results = query_similar(
            client=mock_qdrant_client,
            vector=[1.0, 0.0, 0.0, 0.0],
            vector_name="image",
            filters=fire_filter,
            top_k=5,
            collection_name=collection_name
    )

    assert len(results) == 1
    assert results[0].id == 1


def test_query_text_vector(mock_qdrant_client, collection_name):
    """query against the named text vector independently of the image vector."""
    create_collection(mock_qdrant_client, collection_name, vector_size=4)

    upsert_monster(
        mock_qdrant_client, 1,
        {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 0.0, 0.0, 1.0]},
        {"name": "mon_a"}, collection_name,
    )
    upsert_monster(
        mock_qdrant_client, 2,
        {"image": [0.0, 1.0, 0.0, 0.0], "text": [0.0, 0.0, 1.0, 0.0]},
        {"name": "mon_b"}, collection_name,
    )

    # query with a vector close to mon_b's text vector
    results = query_similar(
        mock_qdrant_client, [0.0, 0.0, 0.9, 0.1],
        vector_name="text", top_k=1, collection_name=collection_name,
    )
    assert len(results) == 1
    assert results[0].id == 2


@pytest.mark.integration
class TestQdrantIntegration:
    """Tests against real Qdrant Docker container."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, real_qdrant_client):
        """Create a unique collection per test, delete after."""
        self.client = real_qdrant_client
        self.collection = "integration_test_monsters"
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        create_collection(self.client, self.collection, vector_size=4)
        yield
        self.client.delete_collection(self.collection)

    def test_create_and_list_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        assert self.collection in collections

    def test_upsert_and_query_monster(self):
        """Upsert a monster and query it back."""
        upsert_monster(
            self.client,
            1,
            {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 1.0, 0.0, 0.0]},
            {"name": "Pikachu", "primary_type": "Electric", "base_level": 25},
            self.collection,
        )
        results = query_similar(
            self.client,
            [1.0, 0.0, 0.0, 0.0],
            vector_name="image",
            top_k=1,
            collection_name=self.collection,
        )
        assert len(results) == 1
        assert results[0].id == 1

    def test_query_with_type_filter(self):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        upsert_monster(self.client, 1, {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 0.0, 0.0, 1.0]},
            {"name": "Firemon", "primary_type": "Fire"}, self.collection)
        upsert_monster(self.client, 2, {"image": [0.0, 1.0, 0.0, 0.0], "text": [0.0, 0.0, 1.0, 0.0]},
            {"name": "Watermon", "primary_type": "Water"}, self.collection)

        fire_filter = Filter(must=[
            FieldCondition(key="primary_type", match=MatchValue(value="Fire"))
        ])
        results = query_similar(
            self.client, [1.0, 0.0, 0.0, 0.0],
            vector_name="image", filters=fire_filter, top_k=10, collection_name=self.collection,
        )
        assert len(results) == 1
        assert results[0].payload["primary_type"] == "Fire"

    def test_upsert_updates_existing_point(self):
        """Upserting same ID replaces the payload."""
        upsert_monster(self.client, 1, {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 0.0, 0.0, 1.0]},
            {"name": "Old", "base_level": 1}, self.collection)
        upsert_monster(self.client, 1, {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 0.0, 0.0, 1.0]},
            {"name": "New", "base_level": 50}, self.collection)

        results = query_similar(
            self.client, [1.0, 0.0, 0.0, 0.0],
            vector_name="image", top_k=1, collection_name=self.collection,
        )
        assert results[0].payload["name"] == "New"
        assert results[0].payload["base_level"] == 50

    def test_query_text_vector(self):
        """query against the named text vector independently of the image vector."""
        upsert_monster(self.client, 1,
            {"image": [1.0, 0.0, 0.0, 0.0], "text": [0.0, 0.0, 0.0, 1.0]},
            {"name": "mon_a"}, self.collection)
        upsert_monster(self.client, 2,
            {"image": [0.0, 1.0, 0.0, 0.0], "text": [0.0, 0.0, 1.0, 0.0]},
            {"name": "mon_b"}, self.collection)

        results = query_similar(
            self.client, [0.0, 0.0, 0.9, 0.1],
            vector_name="text", top_k=1, collection_name=self.collection,
        )
        assert len(results) == 1
        assert results[0].id == 2
