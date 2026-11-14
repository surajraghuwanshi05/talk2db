"""
FastAPI app factory.

    uv run uvicorn app.main:app --reload

Startup (lifespan):
  1. configure logging
  2. refuse unsafe prod config (no SECRET_KEY / no API_KEYS)
  3. migrate the metadata DB to head (AUTO_MIGRATE, on by default)
  4. seed LLM config from env if nothing is configured yet
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.v1 import health
from app.config import Settings, get_settings
from app.db.app_db import get_engine, session_factory
from app.db.migrate import upgrade_to_head
from app.llm.bootstrap import bootstrap_from_env
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def _check_prod_config(settings: Settings) -> None:
    if not settings.is_prod:
        return
    problems = []
    if not settings.SECRET_KEY:
        problems.append("SECRET_KEY is empty")
    if not settings.api_keys:
        problems.append("API_KEYS is empty (the API would be unauthenticated)")
    if problems:
        raise RuntimeError("Refusing to start in prod: " + "; ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)
    _check_prod_config(settings)

    if settings.AUTO_MIGRATE:
        upgrade_to_head(settings.APP_DB_URL)
    get_engine()

    with session_factory()() as session:
        bootstrap_from_env(session, settings)

    logger.info("talk2db started", version=__version__, env=settings.APP_ENV)
    yield
    logger.info("talk2db stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="talk2db",
        version=__version__,
        description="Ask your database questions in plain English — bring your own LLM.",
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/api/v1")
    return app


app = create_app()
