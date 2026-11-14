# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first (cached layer). `local` extra = fastembed for key-free embeddings.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra local --no-install-project

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra local

RUN useradd --create-home --uid 1000 nl2sql && mkdir -p /app/data && chown -R nl2sql /app/data
USER nl2sql

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_CACHE_DIR=/app/data/models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
