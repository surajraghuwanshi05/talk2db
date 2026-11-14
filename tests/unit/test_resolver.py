import pytest

from app.llm.registry import Role
from app.llm.resolver import NoModelConfiguredError, resolve_model
from tests.factories import make_model


def test_unconfigured_chat_role_raises(session):
    with pytest.raises(NoModelConfiguredError):
        resolve_model(session, Role.SQL_GENERATION)


def test_unconfigured_embedding_returns_none(session):
    assert resolve_model(session, Role.EMBEDDING) is None


def test_assigned_model_resolves_with_decrypted_key(session):
    make_model(
        session,
        provider_type="anthropic",
        model_name="claude-sonnet-5-5",
        api_key="sk-ant-xyz",
        assign=["sql_generation"],
    )
    r = resolve_model(session, Role.SQL_GENERATION)
    assert r.litellm_model == "anthropic/claude-sonnet-5-5"
    assert r.call_kwargs == {"api_key": "sk-ant-xyz"}
    assert r.is_local is False


def test_api_key_never_in_repr(session):
    make_model(session, api_key="sk-supersecret", assign=["sql_generation"])
    r = resolve_model(session, Role.SQL_GENERATION)
    assert "sk-supersecret" not in repr(r)


def test_other_chat_roles_fall_back_to_sql_generation(session):
    m = make_model(session, assign=["sql_generation"])
    r = resolve_model(session, Role.ANSWER_FORMATTING)
    assert r.model_id == m.id


def test_explicit_role_assignment_beats_fallback(session):
    make_model(session, model_name="big", assign=["sql_generation"])
    small = make_model(session, model_name="small", assign=["answer_formatting"])
    assert resolve_model(session, Role.ANSWER_FORMATTING).model_id == small.id


def test_model_id_override(session):
    make_model(session, model_name="default", assign=["sql_generation"])
    other = make_model(
        session,
        provider_type="ollama",
        model_name="llama3.1:8b",
        api_key=None,
        api_base="http://localhost:11434",
    )
    r = resolve_model(session, Role.SQL_GENERATION, model_id=other.id)
    assert r.litellm_model == "ollama_chat/llama3.1:8b"
    assert r.is_local is True


def test_override_with_wrong_kind_rejected(session):
    emb = make_model(
        session,
        provider_type="fastembed",
        model_name="BAAI/bge-small-en-v1.5",
        kind="embedding",
        api_key=None,
    )
    with pytest.raises(NoModelConfiguredError, match="embedding model"):
        resolve_model(session, Role.SQL_GENERATION, model_id=emb.id)


def test_disabled_model_is_not_used(session):
    m = make_model(session, assign=["sql_generation"])
    m.is_enabled = False
    session.commit()
    with pytest.raises(NoModelConfiguredError):
        resolve_model(session, Role.SQL_GENERATION)
