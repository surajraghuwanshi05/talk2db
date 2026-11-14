"""Tiny helpers to create LLM config rows in tests."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import LLMModel, LLMProvider, ModelAssignment
from app.utils.crypto import encrypt


def make_model(
    session: Session,
    *,
    provider_type: str = "openai",
    model_name: str = "gpt-test",
    kind: str = "chat",
    api_key: str | None = "sk-test-1234567890",
    api_base: str | None = None,
    extra: dict | None = None,
    assign: list[str] | None = None,
    display_name: str | None = None,
) -> LLMModel:
    p = LLMProvider(
        provider_type=provider_type,
        display_name=display_name or f"{provider_type}-{model_name}-{kind}",
        api_key_encrypted=encrypt(api_key),
        api_base=api_base,
        extra=extra or {},
    )
    session.add(p)
    session.flush()
    m = LLMModel(provider_id=p.id, model_name=model_name, kind=kind)
    session.add(m)
    session.flush()
    for role in assign or []:
        session.add(ModelAssignment(role=role, model_id=m.id))
    session.commit()
    return m
