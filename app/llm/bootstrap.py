"""
Seed LLM configuration from environment variables — for headless / CLI / eval use
before the Settings UI exists, and for docker-compose "just works" setups.

Idempotent and conservative:
  • Chat: only if NO chat model is assigned yet and LLM_PROVIDER + LLM_MODEL are set.
    Creates a provider + model and assigns it to every chat role.
  • Embedding: only if the embedding role is unassigned and EMBEDDING_PROVIDER is
    not "none". Default is local fastembed, so a fresh clone gets embeddings with
    no key.

Anything configured through the UI is never overwritten.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import LLMModel, LLMProvider, ModelAssignment
from app.llm.registry import CHAT_ROLES, Role, UnknownProviderError, get_spec
from app.utils.crypto import encrypt
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_or_create_provider(
    session: Session, provider_type: str, api_key: str, api_base: str
) -> LLMProvider:
    name = f"{get_spec(provider_type).label} (from .env)"
    p = session.scalars(select(LLMProvider).where(LLMProvider.display_name == name)).first()
    if p is None:
        default_base = next(
            (f.default for f in get_spec(provider_type).fields if f.target == "api_base"), None
        )
        p = LLMProvider(
            provider_type=provider_type,
            display_name=name,
            api_key_encrypted=encrypt(api_key) if api_key else None,
            api_base=api_base or default_base,
        )
        session.add(p)
        session.flush()
    return p


def _get_or_create_model(session: Session, provider: LLMProvider, name: str, kind: str) -> LLMModel:
    m = session.scalars(
        select(LLMModel).where(
            LLMModel.provider_id == provider.id,
            LLMModel.model_name == name,
            LLMModel.kind == kind,
        )
    ).first()
    if m is None:
        m = LLMModel(provider_id=provider.id, model_name=name, kind=kind)
        session.add(m)
        session.flush()
    return m


def bootstrap_from_env(session: Session, settings: Settings) -> dict[str, str]:
    """Returns {role: 'seeded …'} for whatever it seeded (empty if nothing)."""
    seeded: dict[str, str] = {}

    # ── chat ─────────────────────────────────────────────────────────────────
    has_chat = session.get(ModelAssignment, Role.SQL_GENERATION.value) is not None
    if not has_chat and settings.LLM_PROVIDER and settings.LLM_MODEL:
        try:
            provider = _get_or_create_provider(
                session, settings.LLM_PROVIDER, settings.LLM_API_KEY, settings.LLM_API_BASE
            )
            model = _get_or_create_model(session, provider, settings.LLM_MODEL, "chat")
            for role in CHAT_ROLES:
                if session.get(ModelAssignment, role.value) is None:
                    session.add(ModelAssignment(role=role.value, model_id=model.id))
                    seeded[role.value] = f"{settings.LLM_PROVIDER}:{settings.LLM_MODEL}"
        except UnknownProviderError as e:
            logger.error("LLM bootstrap skipped", error=str(e))

    # ── embedding ────────────────────────────────────────────────────────────
    emb_provider = settings.EMBEDDING_PROVIDER
    has_emb = session.get(ModelAssignment, Role.EMBEDDING.value) is not None
    if not has_emb and emb_provider and emb_provider != "none" and settings.EMBEDDING_MODEL:
        try:
            spec = get_spec(emb_provider)
            if "embedding" not in spec.kinds:
                raise UnknownProviderError(f"{emb_provider} does not provide embeddings")
            provider = _get_or_create_provider(
                session, emb_provider, settings.EMBEDDING_API_KEY, settings.EMBEDDING_API_BASE
            )
            model = _get_or_create_model(session, provider, settings.EMBEDDING_MODEL, "embedding")
            session.add(ModelAssignment(role=Role.EMBEDDING.value, model_id=model.id))
            seeded[Role.EMBEDDING.value] = f"{emb_provider}:{settings.EMBEDDING_MODEL}"
        except UnknownProviderError as e:
            logger.error("Embedding bootstrap skipped", error=str(e))

    session.commit()
    if seeded:
        logger.info("Seeded LLM configuration from environment", seeded=seeded)
    return seeded
