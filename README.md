# masakinet

Multimodal search & generation platform for pixel-art creature assets (Pokémon / Digimon / TemTem-style). See [`design-doc.md`](./design-doc.md) for the full architecture and roadmap.

## Capabilities

1. **Multimodal Information Retrieval** — text-to-image, image-to-text, and metadata-filtered search over a creature index using joint vector embeddings (CLIP/SigLIP + Qdrant).
2. **Generative Synthesis** — text-to-pixel-art generation and image-to-structured-metadata captioning.

## Project layout

```
src/
├── api/         FastAPI app, routers (search, generate)
├── db/          Qdrant client & collection operations
├── pipeline/    Offline dataset loading & embedding ingestion
├── schemas/     Pydantic data contracts
└── services/    ML inference engines (embeddings, captioning, generation)
tests/           pytest suite (data, model, API tiers)
docker/          Dockerfile & docker-compose for local Qdrant + app
data/            raw/ and processed/ assets (git-ignored)
```

## Getting started

> Implementation is not yet in place — this is scaffolding only. See `design-doc.md` §5.2 for the roadmap.

```bash
# Install dependencies
poetry install

# Run the API locally (once implemented)
poetry run uvicorn src.api.main:app --reload

# Run the local Qdrant instance
docker compose -f docker/docker-compose.yml up -d

# Run tests
poetry run pytest
```

## Development

- Formatting: `black`
- Linting: `flake8`
- Type checking: `mypy`
- Pre-commit hooks: `pre-commit install`

## Configuration

Application settings are managed via `pydantic-settings` in `src/config.py`. Copy `.env.example` to `.env` and fill in values as needed (Qdrant host/port, model paths, etc.) once defined.
