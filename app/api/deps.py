"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import get_settings
from app.db.app_db import get_session

__all__ = ["get_session", "require_api_key"]


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Admin API-key check. Disabled when API_KEYS is empty (dev convenience).

    In prod an empty API_KEYS is refused at startup (see app/main.py), so this
    can't silently leave a deployed instance open.
    """
    keys = get_settings().api_keys
    if not keys:
        return
    if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")
