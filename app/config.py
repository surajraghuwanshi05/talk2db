"""
Application settings, loaded from environment variables / `.env`.

Every setting has a default that works for local development, so a fresh clone
runs with no `.env` at all. See `.env.example` for documentation of each value.

LLM providers are deliberately NOT configured here in normal use — they live in
the metadata DB and are managed from the UI (plan §15). The `LLM_*` values are an
optional bootstrap so a headless/CLI setup can work before the UI exists.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_ENV: Literal["dev", "prod", "test"] = "dev"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    APP_DB_URL: str = "sqlite:///./data/nl2sql_app.db"
    AUTO_MIGRATE: bool = True  # run Alembic migrations to head on startup

    # Fernet key for secrets at rest. Empty in dev → a key is derived and a
    # warning is logged (see app/utils/crypto.py). Required in prod.
    SECRET_KEY: str = ""

    # Comma-separated admin API keys. Empty → auth disabled (dev only).
    API_KEYS: str = ""

    # ── LLM bootstrap (optional) ─────────────────────────────────────────────
    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""
    LLM_API_KEY: str = ""
    LLM_API_BASE: str = ""

    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_BASE: str = ""
    # Where local (fastembed) models are downloaded/cached. Persist it in docker.
    MODEL_CACHE_DIR: str = "./data/models"

    # LiteLLM call behaviour
    LLM_TIMEOUT_S: float = 120.0
    LLM_NUM_RETRIES: int = 2

    # ── Vector store ─────────────────────────────────────────────────────────
    VECTOR_BACKEND: Literal["memory", "qdrant"] = "memory"
    QDRANT_URL: str = "http://localhost:6333"

    # ── Query safety ─────────────────────────────────────────────────────────
    SQL_QUERY_TIMEOUT_S: int = Field(30, ge=1, le=600)
    SQL_MAX_ROWS: int = Field(500, ge=1, le=100_000)

    @field_validator("LLM_PROVIDER", "EMBEDDING_PROVIDER")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def is_prod(self) -> bool:
        return self.APP_ENV == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
