"""Data crawler for PokeAPI and Bulbapedia.

Fetches Pokemon data from pokeapi.co and bulbapedia.bulbagarden.net
using the PokeApiScraper and BulbapediaScraper classes,
and stores it locally in data/raw/.

Usage:
    poetry run python data/data_crawler.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.datascrape import BulbapediaScraper, PokeApiScraper


def main() -> None:
    list_path = Path(__file__).parent / "raw" / "pokeapi_list.txt"
    out_dir = Path(__file__).parent / "raw"

    with PokeApiScraper(list_path=list_path) as pokeapi_scraper:
        records = pokeapi_scraper.fetch_all(save=True)

    if not records:
        print("No Pokemon were fetched successfully; nothing to save.")
        return

    with BulbapediaScraper(list_path=list_path) as bulbapedia_scraper:
        biology_data = bulbapedia_scraper.fetch_all()

    for record in records:
        record.biology = biology_data.get(record.name)

    with PokeApiScraper(list_path=list_path) as pokeapi_scraper:
        pokeapi_scraper.save_metadata(records, out_dir / "pokeapi_metadata.json")

    print(f"Saved metadata for {len(records)} Pokemon to {out_dir / 'pokeapi_metadata.json'}.")


if __name__ == "__main__":
    main()
