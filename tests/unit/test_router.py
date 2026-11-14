from types import SimpleNamespace

import pytest

from app.llm import embeddings, router
from app.llm.registry import Role
from app.llm.router import LLMCallError, generate_chat, strip_reasoning
from tests.factories import make_model


def _fake_completion(content: str, captured: dict):
    async def _acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        )

    return _acompletion


def test_strip_reasoning():
    assert strip_reasoning("<think>\nhmm\n</think>\nSELECT 1") == "SELECT 1"
    assert strip_reasoning("SELECT 1") == "SELECT 1"


async def test_generate_chat_uses_configured_model(session, monkeypatch):
    make_model(
        session,
        provider_type="anthropic",
        model_name="claude-haiku-5-5",
        api_key="sk-ant-1",
        assign=["sql_generation"],
    )
    captured: dict = {}
    monkeypatch.setattr(
        router.litellm, "acompletion", _fake_completion("<think>x</think>SELECT 1", captured)
    )

    resp = await generate_chat([{"role": "user", "content": "hi"}], session=session)

    assert resp.text == "SELECT 1"
    assert (resp.tokens_in, resp.tokens_out) == (11, 3)
    assert captured["model"] == "anthropic/claude-haiku-5-5"
    assert captured["api_key"] == "sk-ant-1"
    assert captured["temperature"] == 0.0


async def test_provider_error_becomes_llm_call_error(session, monkeypatch):
    make_model(session, assign=["sql_generation"])

    async def boom(**_):
        raise ConnectionError("refused")

    monkeypatch.setattr(router.litellm, "acompletion", boom)
    with pytest.raises(LLMCallError, match="ConnectionError"):
        await generate_chat([{"role": "user", "content": "hi"}], session=session)


async def test_embed_texts_none_when_unconfigured(session):
    assert await embeddings.embed_texts(["a"], session=session) is None


async def test_embed_texts_degrades_on_failure(session, monkeypatch):
    make_model(
        session,
        provider_type="openai",
        model_name="text-embedding-3-small",
        kind="embedding",
        assign=[Role.EMBEDDING.value],
    )

    async def boom(**_):
        raise TimeoutError()

    monkeypatch.setattr(embeddings.litellm, "aembedding", boom)
    assert await embeddings.embed_texts(["a"], session=session) is None


async def test_embed_texts_litellm_batches(session, monkeypatch):
    make_model(
        session,
        provider_type="openai",
        model_name="text-embedding-3-small",
        kind="embedding",
        assign=[Role.EMBEDDING.value],
    )
    calls = []

    async def fake(**kw):
        calls.append(len(kw["input"]))
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2]} for _ in kw["input"]])

    monkeypatch.setattr(embeddings.litellm, "aembedding", fake)
    out = await embeddings.embed_texts([f"t{i}" for i in range(70)], session=session)
    assert len(out) == 70 and calls == [64, 6]
