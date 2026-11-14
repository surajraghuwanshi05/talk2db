"""
Shared fixtures.

Every test gets:
  • a fresh temp SQLite metadata DB, migrated with the REAL Alembic migrations
    (so migrations are tested on every run, not just the ORM models)
  • isolated settings — no developer .env leaks into tests
  • no real network: LLM calls are monkeypatched in the tests that need them
"""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet

# Set BEFORE app modules read settings. Point env_file at nothing so a developer's
# .env can never change test behaviour.
os.environ.update(
    {
        "APP_ENV": "test",
        "SECRET_KEY": Fernet.generate_key().decode(),
        "API_KEYS": "",
        "LLM_PROVIDER": "",
        "LLM_MODEL": "",
        "EMBEDDING_PROVIDER": "none",
    }
)

from app.config import Settings, get_settings  # noqa: E402
from app.db import app_db  # noqa: E402
from app.db.migrate import upgrade_to_head  # noqa: E402
from app.utils import crypto  # noqa: E402

Settings.model_config["env_file"] = None


def _clear_caches() -> None:
    get_settings.cache_clear()
    crypto._fernet.cache_clear()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("APP_DB_URL", url)
    _clear_caches()
    upgrade_to_head(url)
    engine = app_db.make_engine(url)
    app_db.set_engine(engine)
    yield
    engine.dispose()
    _clear_caches()


@pytest.fixture
def session():
    with app_db.session_factory()() as s:
        yield s


@pytest.fixture
def set_env(monkeypatch):
    """set_env(LLM_PROVIDER="ollama", ...) — sets env vars and refreshes settings."""

    def _set(**kv):
        for k, v in kv.items():
            monkeypatch.setenv(k, v)
        _clear_caches()

    return _set
