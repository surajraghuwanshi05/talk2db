from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import LLMProvider, ModelAssignment
from app.llm.bootstrap import bootstrap_from_env
from app.llm.registry import CHAT_ROLES, Role
from app.llm.resolver import resolve_model
from tests.factories import make_model


def _count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def test_nothing_set_seeds_nothing(session):
    assert bootstrap_from_env(session, get_settings()) == {}
    assert _count(session, LLMProvider) == 0


def test_seeds_chat_and_embedding_from_env(session, set_env):
    set_env(
        LLM_PROVIDER="ollama",
        LLM_MODEL="qwen2.5-coder:7b",
        EMBEDDING_PROVIDER="fastembed",
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
    )
    seeded = bootstrap_from_env(session, get_settings())

    assert set(seeded) == {r.value for r in CHAT_ROLES} | {Role.EMBEDDING.value}
    chat = resolve_model(session, Role.SQL_GENERATION)
    assert chat.litellm_model == "ollama_chat/qwen2.5-coder:7b"
    # Ollama's default base URL from the registry is filled in
    assert chat.call_kwargs["api_base"] == "http://localhost:11434"
    assert resolve_model(session, Role.EMBEDDING).provider_type == "fastembed"


def test_idempotent(session, set_env):
    set_env(LLM_PROVIDER="openai", LLM_MODEL="gpt-4.1-mini", LLM_API_KEY="sk-x")
    bootstrap_from_env(session, get_settings())
    assert bootstrap_from_env(session, get_settings()) == {}
    assert _count(session, LLMProvider) == 1


def test_never_overrides_ui_configuration(session, set_env):
    ui = make_model(session, model_name="from-ui", assign=["sql_generation"])
    set_env(LLM_PROVIDER="ollama", LLM_MODEL="qwen2.5-coder:7b")
    bootstrap_from_env(session, get_settings())
    assert session.get(ModelAssignment, "sql_generation").model_id == ui.id


def test_unknown_provider_is_logged_not_fatal(session, set_env):
    set_env(LLM_PROVIDER="not-a-provider", LLM_MODEL="x")
    assert bootstrap_from_env(session, get_settings()) == {}


def test_embedding_none_disables(session, set_env):
    set_env(EMBEDDING_PROVIDER="none")
    bootstrap_from_env(session, get_settings())
    assert resolve_model(session, Role.EMBEDDING) is None
