"""Application configuration, managed via pydantic-settings.

Values are sourced from environment variables / a local .env file. Extend
this as new components (vector DB, model registry, object storage) come
online.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    app_name: str = "masakinet"
    environment: str = "development"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "monsters"

    # --- Model paths / identifiers (placeholders) ---
    clip_model_name: str = "openai/clip-vit-base-patch32"


settings = Settings()
