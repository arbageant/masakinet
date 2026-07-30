# Multi-stage production container build.
# Build from the repo root: `docker build -f docker/Dockerfile .`
# See design-doc.md §4.2 for rationale.
# TODO: fix python versions to >=3.12

# --- Stage 1: Build & Compilation Dependencies ---
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes -f requirements.txt --output requirements.txt
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# --- Stage 2: Final Lean Runtime Asset ---
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
COPY src/ /app/src
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
