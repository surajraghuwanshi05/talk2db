"""
Embeddings — local (fastembed) or any LiteLLM-supported provider.

    vectors = await embed_texts(["Table artist ...", "Table album ..."])
    if vectors is None:   # no embedding model configured / available
        ... fall back to BM25-only retrieval (graceful degradation)

Returns None — never raises — when embeddings are unavailable, because retrieval
must keep working without them (plan §15.1). Real call failures are logged.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import litellm
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.app_db import session_factory
from app.llm.registry import Role
from app.llm.resolver import ResolvedModel, resolve_model
from app.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 64


@lru_cache(maxsize=4)
def _fastembed_model(model_name: str) -> Any:
    from fastembed import TextEmbedding  # optional dependency: `uv sync --extra local`

    cache_dir = get_settings().MODEL_CACHE_DIR
    logger.info(
        "Loading local embedding model (first use downloads it)",
        model=model_name,
        cache_dir=cache_dir,
    )
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir)


def _fastembed_sync(model_name: str, texts: list[str]) -> list[list[float]]:
    model = _fastembed_model(model_name)
    return [vec.tolist() for vec in model.embed(texts, batch_size=BATCH_SIZE)]


async def _embed_with(resolved: ResolvedModel, texts: list[str]) -> list[list[float]]:
    if resolved.provider_type == "fastembed":
        return await asyncio.to_thread(_fastembed_sync, resolved.model_name, texts)

    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = await litellm.aembedding(
            model=resolved.litellm_model, input=batch, **resolved.call_kwargs
        )
        out.extend(item["embedding"] for item in resp.data)
    return out


def _resolve(session: Session | None) -> ResolvedModel | None:
    if session is not None:
        return resolve_model(session, Role.EMBEDDING)
    with session_factory()() as s:
        return resolve_model(s, Role.EMBEDDING)


async def embed_texts(
    texts: list[str], *, session: Session | None = None
) -> list[list[float]] | None:
    """Embed `texts` with the configured embedding model, or None if unavailable."""
    if not texts:
        return []
    resolved = _resolve(session)
    if resolved is None:
        return None
    try:
        return await _embed_with(resolved, texts)
    except ImportError:
        logger.warning("fastembed not installed — run `uv sync --extra local`; using BM25-only")
        return None
    except Exception as e:
        logger.warning(
            "Embedding failed; using BM25-only",
            model=resolved.label,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
        return None


async def embed_text(text: str, *, session: Session | None = None) -> list[float] | None:
    vecs = await embed_texts([text], session=session)
    return vecs[0] if vecs else None


async def embed_with_model(resolved: ResolvedModel, texts: list[str]) -> list[list[float]]:
    """Embed with a specific model, raising on failure (used by health checks)."""
    return await _embed_with(resolved, texts)
