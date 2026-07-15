"""Data crawler for PokeAPI.

Fetches Pokemon data from pokeapi.co using the PokeApiScraper class
and stores it locally in data/raw/.

Usage:
    poetry run python data/data_crawler.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.datascrape import PokeApiScraper


def main() -> None:
    list_path = Path(__file__).parent / "raw" / "pokeapi_list.txt"
    out_dir = Path(__file__).parent / "raw"

    with PokeApiScraper(list_path=list_path) as scraper:
        records = scraper.fetch_all(save=True)
        scraper.save_metadata(records, out_dir / "pokeapi_metadata.json")


if __name__ == "__main__":
    main()
