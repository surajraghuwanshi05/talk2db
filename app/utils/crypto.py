"""
Encryption for secrets at rest (LLM API keys, datasource connection URLs).

Uses Fernet (AES-128-CBC + HMAC) keyed by settings.SECRET_KEY.

  • prod: SECRET_KEY is required — startup fails without it.
  • dev/test: if SECRET_KEY is empty, a fixed dev key is derived and a warning is
    logged. Fine on your laptop; never use it for real keys you care about.

`mask_secret()` is what the API shows instead of a key: "sk-…abcd".
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEV_SEED = b"nl2sql-insecure-dev-key-do-not-use-in-prod"


class SecretKeyMissingError(RuntimeError):
    pass


class DecryptionError(RuntimeError):
    pass


@lru_cache
def _fernet() -> Fernet:
    settings = get_settings()
    key = settings.SECRET_KEY.strip()
    if not key:
        if settings.is_prod:
            raise SecretKeyMissingError(
                "SECRET_KEY must be set in production. Generate one with: "
                'python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            )
        logger.warning("SECRET_KEY not set — using an insecure dev key")
        key = base64.urlsafe_b64encode(hashlib.sha256(_DEV_SEED).digest()).decode()
    return Fernet(key.encode())


def encrypt(plaintext: str | None) -> str | None:
    if not plaintext:
        return None
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str | None) -> str | None:
    if not token:
        return None
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as e:
        raise DecryptionError(
            "Could not decrypt a stored secret — SECRET_KEY changed? Re-enter the key."
        ) from e


def mask_secret(plaintext: str | None) -> str | None:
    """'sk-proj-abc123wxyz' -> 'sk-…wxyz'. Never reveals more than 4 trailing chars."""
    if not plaintext:
        return None
    if len(plaintext) <= 8:
        return "…" + plaintext[-2:]
    prefix = plaintext[:3] if plaintext[:3].isascii() else ""
    return f"{prefix}…{plaintext[-4:]}"
