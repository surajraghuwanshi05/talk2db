"""Run Alembic migrations programmatically (used at startup and by tests)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(db_url: str | None = None) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    # Don't let alembic.ini's [loggers] reconfigure the app's logging mid-startup.
    cfg.attributes["configure_logger"] = False
    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def upgrade_to_head(db_url: str | None = None) -> None:
    command.upgrade(alembic_config(db_url), "head")
