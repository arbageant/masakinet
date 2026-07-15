"""Data scraping and download utilities for external APIs.

Fetches raw asset data from third-party sources and stores it locally
in data/raw/ for downstream embedding and ingestion.

Usage:
    poetry run python -m src.pipeline.datascrape
"""

import json
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class PokeApiData(BaseModel):
    """Structured output for a single Pokemon fetched from PokeAPI."""

    name: str
    types: list[str]
    flavor_text: str = Field(..., description="Most recent English flavor_text.")
    image_url: str
    image_path: Optional[Path] = None
    biology: Optional[str] = Field(None, description="Biology section text from Bulbapedia.")


class PokeApiScraper:
    """Fetch Pokemon data from pokeapi.co and save locally."""

    BASE_URL = "https://pokeapi.co/api/v2"

    def __init__(self, list_path: Path | str = "data/raw/pokeapi_list.txt") -> None:
        self.list_path = Path(list_path)
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PokeApiScraper":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _read_names(self) -> list[str]:
        """Read Pokemon names from the list file, one per line."""
        return [
            line.strip()
            for line in self.list_path.read_text().splitlines()
            if line.strip()
        ]

    def _get_pokemon(self, name: str) -> dict:
        resp = self._client.get(f"{self.BASE_URL}/pokemon/{name}")
        resp.raise_for_status()
        return resp.json()

    def _get_species(self, name: str) -> dict:
        resp = self._client.get(f"{self.BASE_URL}/pokemon-species/{name}")
        resp.raise_for_status()
        return resp.json()

    def _pick_image_url(self, pokemon: dict) -> str:
        sprites = pokemon.get("sprites", {})
        official = sprites.get("other", {}).get("official-artwork", {})
        if official.get("front_default"):
            return official["front_default"]
        return sprites.get("front_default", "")

    def _pick_flavor_text(self, species: dict) -> str:
        """Return the most recent English flavor_text entry."""
        entries = species.get("flavor_text_entries", [])
        for entry in reversed(entries):
            if entry.get("language", {}).get("name") == "en":
                return entry["flavor_text"].replace("\n", " ").replace("\f", " ")
        return ""

    def _parse_types(self, pokemon: dict) -> list[str]:
        return [t["type"]["name"] for t in pokemon.get("types", [])]

    def fetch_one(self, name: str) -> PokeApiData:
        """Fetch all fields for a single Pokemon by name."""
        pokemon = self._get_pokemon(name)
        species = self._get_species(name)

        return PokeApiData(
            name=pokemon["name"],
            types=self._parse_types(pokemon),
            flavor_text=self._pick_flavor_text(species),
            image_url=self._pick_image_url(pokemon),
        )

    def download_image(self, data: PokeApiData, out_dir: Path | str = "data/raw") -> Path:
        """Download the Pokemon image and return the local path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{data.name}.png"
        dest = out_dir / filename

        if dest.exists():
            return dest

        resp = self._client.get(data.image_url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest

    def fetch_all(self, save: bool = True) -> list[PokeApiData]:
        """Fetch every Pokemon in the list file."""
        names = self._read_names()
        results: list[PokeApiData] = []

        for name in names:
            data = self.fetch_one(name)
            if save:
                data.image_path = self.download_image(data)
            results.append(data)

        return results

    def save_metadata(self, records: list[PokeApiData], out_path: Path | str = "data/raw/pokeapi_metadata.json") -> Path:
        """Persist scraped metadata as JSON."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = [r.model_dump(mode="json") for r in records]
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path


class BulbapediaScraper:
    """Fetch Pokemon biology text from Bulbapedia."""

    BASE_URL = "https://bulbapedia.bulbagarden.net/wiki"

    def __init__(self, list_path: Path | str = "data/raw/pokeapi_list.txt", delay: float = 1.0) -> None:
        self.list_path = Path(list_path)
        self.delay = delay
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BulbapediaScraper":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _read_names(self) -> list[str]:
        """Read Pokemon names from the list file, one per line."""
        return [
            line.strip()
            for line in self.list_path.read_text().splitlines()
            if line.strip()
        ]

    def _build_url(self, name: str) -> str:
        """Build Bulbapedia URL for a Pokemon name."""
        capitalized = name.capitalize()
        encoded = quote(f"{capitalized}_(Pokémon)")
        return f"{self.BASE_URL}/{encoded}"

    def _fetch_page(self, name: str) -> Optional[str]:
        """Fetch the HTML content of a Pokemon's Bulbapedia page."""
        url = self._build_url(name)
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError:
            return None

    def _extract_biology(self, html: str) -> Optional[str]:
        """Extract text from the Biology section of the HTML."""
        soup = BeautifulSoup(html, "html.parser")

        biology_heading = soup.find("span", id="Biology")
        if not biology_heading:
            return None

        h2 = biology_heading.parent
        if not h2:
            return None

        paragraphs = []
        for element in h2.find_next_siblings():
            if element.name == "h2":
                break
            if element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    paragraphs.append(text)

        return "\n\n".join(paragraphs) if paragraphs else None

    def fetch_one(self, name: str) -> Optional[str]:
        """Fetch biology text for a single Pokemon by name."""
        html = self._fetch_page(name)
        if html is None:
            return None
        return self._extract_biology(html)

    def fetch_all(self) -> dict[str, Optional[str]]:
        """Fetch biology text for every Pokemon in the list file."""
        names = self._read_names()
        results: dict[str, Optional[str]] = {}

        for i, name in enumerate(names):
            if i > 0:
                time.sleep(self.delay)
            results[name] = self.fetch_one(name)

        return results
