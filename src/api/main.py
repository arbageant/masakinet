"""FastAPI application initialization & middleware.

Entrypoint referenced by the Docker CMD and local uvicorn runs:
    uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI

from src.api.routers import generate, search
from src.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(search.router)
app.include_router(generate.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic liveness probe."""
    return {"status": "ok"}
