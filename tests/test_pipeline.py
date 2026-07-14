"""Data generation and import tests
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pipeline.datascrape import PokeApiData, PokeApiScraper

TEST_LIST_PATH = Path(__file__).parent / "pokeapi_list_test.txt"


@pytest.mark.slow
def test_pokeapi_scraper_full_pipeline(tmp_path: Path):
    """Fetch Pokemon data from the real API and create a test dataset."""
    with PokeApiScraper(list_path=TEST_LIST_PATH) as scraper:
        records = scraper.fetch_all(save=False)

        assert len(records) == 3

        for rec in records:
            assert isinstance(rec, PokeApiData)
            assert rec.name
            assert isinstance(rec.types, list) and len(rec.types) > 0
            assert isinstance(rec.flavor_text, str) and len(rec.flavor_text) > 0
            assert rec.image_url.startswith("http")

        metadata_path = tmp_path / "pokeapi_metadata.json"
        saved = scraper.save_metadata(records, out_path=metadata_path)

        assert saved == metadata_path
        assert metadata_path.exists()

        data = json.loads(metadata_path.read_text())
        assert len(data) == 3
        assert {r["name"] for r in data} == {"pikachu", "bulbasaur", "charmander"}

