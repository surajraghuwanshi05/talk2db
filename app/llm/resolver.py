"""
Resolve a pipeline role (or an explicit model id) to a ready-to-call model.

    resolved = resolve_model(session, Role.SQL_GENERATION)
    litellm.acompletion(model=resolved.litellm_model, **resolved.call_kwargs, ...)

Fallback rules (plan §15.2):
  • chat role without an assignment → use the sql_generation model
  • embedding role without an assignment → None (pipeline degrades to BM25-only)
  • explicit model_id (per-request override from the chat dropdown) wins
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.models import LLMModel, ModelAssignment
from app.llm.registry import Role, build_call_kwargs, get_spec, litellm_model_string
from app.utils.crypto import decrypt


class NoModelConfiguredError(RuntimeError):
    """No model is assigned for this role — the user must add one in Settings → Models."""


@dataclass
class ResolvedModel:
    model_id: int
    provider_type: str
    provider_name: str
    model_name: str
    kind: str
    litellm_model: str
    is_local: bool
    context_window: int | None = None
    temperature_default: float = 0.0
    call_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)  # holds secrets

    @property
    def label(self) -> str:
        return f"{self.provider_name} · {self.model_name}"


def _to_resolved(m: LLMModel) -> ResolvedModel:
    p = m.provider
    spec = get_spec(p.provider_type)
    return ResolvedModel(
        model_id=m.id,
        provider_type=p.provider_type,
        provider_name=p.display_name,
        model_name=m.model_name,
        kind=m.kind,
        litellm_model=litellm_model_string(p.provider_type, m.model_name, m.kind),
        is_local=spec.local,
        context_window=m.context_window,
        temperature_default=m.temperature_default,
        call_kwargs=build_call_kwargs(
            p.provider_type,
            api_key=decrypt(p.api_key_encrypted),
            api_base=p.api_base,
            extra=p.extra,
        ),
    )


def _load_model(session: Session, model_id: int) -> LLMModel | None:
    stmt = select(LLMModel).options(joinedload(LLMModel.provider)).where(LLMModel.id == model_id)
    m = session.scalars(stmt).first()
    if m is None or not m.is_enabled or not m.provider.is_enabled:
        return None
    return m


def _assigned(session: Session, role: Role) -> LLMModel | None:
    a = session.get(ModelAssignment, role.value)
    return _load_model(session, a.model_id) if a else None


def resolve_model(
    session: Session, role: Role, model_id: int | None = None
) -> ResolvedModel | None:
    """Return the model for `role`, or None for an unconfigured EMBEDDING role.

    Raises NoModelConfiguredError for an unconfigured chat role.
    """
    if model_id is not None:
        m = _load_model(session, model_id)
        if m is None:
            raise NoModelConfiguredError(f"Model {model_id} not found or disabled")
        if m.kind != role.kind:
            raise NoModelConfiguredError(
                f"Model {m.model_name} is a {m.kind} model; role {role} needs {role.kind}"
            )
        return _to_resolved(m)

    m = _assigned(session, role)
    if m is None and role.kind == "chat" and role is not Role.SQL_GENERATION:
        m = _assigned(session, Role.SQL_GENERATION)

    if m is None:
        if role is Role.EMBEDDING:
            return None
        raise NoModelConfiguredError(
            "No LLM configured. Add a provider in Settings → Models "
            "(or set LLM_PROVIDER / LLM_MODEL in .env)."
        )
    return _to_resolved(m)
