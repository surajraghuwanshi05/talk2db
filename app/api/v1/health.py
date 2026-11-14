"""GET /api/v1/health — liveness + what's configured (never any secrets)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import __version__
from app.db.app_db import get_session, ping
from app.llm.registry import Role
from app.llm.resolver import NoModelConfiguredError, resolve_model

router = APIRouter(tags=["health"])


class RoleStatus(BaseModel):
    role: str
    model: str | None
    local: bool | None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    version: str
    app_db: bool
    llm_configured: bool
    roles: list[RoleStatus]
    hints: list[str]


@router.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    db_ok = ping()
    roles: list[RoleStatus] = []
    hints: list[str] = []

    if db_ok:
        for role in Role:
            try:
                r = resolve_model(session, role)
                roles.append(
                    RoleStatus(
                        role=role.value,
                        model=r.label if r else None,
                        local=r.is_local if r else None,
                    )
                )
            except NoModelConfiguredError:
                roles.append(RoleStatus(role=role.value, model=None, local=None))
            except Exception as e:  # e.g. undecryptable key after SECRET_KEY change
                roles.append(RoleStatus(role=role.value, model=None, local=None))
                hints.append(f"{role.value}: {type(e).__name__}: {e}")

    llm_ok = any(r.role == Role.SQL_GENERATION.value and r.model for r in roles)
    if not llm_ok:
        hints.append(
            "No chat model configured: set LLM_PROVIDER/LLM_MODEL in .env "
            "or add one in Settings → Models."
        )
    emb = next((r for r in roles if r.role == Role.EMBEDDING.value), None)
    if emb and not emb.model:
        hints.append("No embedding model: table retrieval will use BM25 keywords only.")

    return HealthResponse(
        status="ok" if db_ok and llm_ok else "degraded",
        version=__version__,
        app_db=db_ok,
        llm_configured=llm_ok,
        roles=roles,
        hints=hints,
    )
