import pytest

from app.llm.registry import (
    PROVIDERS,
    Role,
    UnknownProviderError,
    build_call_kwargs,
    litellm_model_string,
    public_registry,
)


@pytest.mark.parametrize(
    ("provider", "model", "kind", "expected"),
    [
        ("anthropic", "claude-sonnet-5-5", "chat", "anthropic/claude-sonnet-5-5"),
        ("openai", "gpt-4.1-mini", "chat", "openai/gpt-4.1-mini"),
        ("ollama", "qwen2.5-coder:7b", "chat", "ollama_chat/qwen2.5-coder:7b"),
        ("ollama", "nomic-embed-text", "embedding", "ollama/nomic-embed-text"),
        (
            "openrouter",
            "meta-llama/llama-3.3-70b-instruct",
            "chat",
            "openrouter/meta-llama/llama-3.3-70b-instruct",
        ),
        ("fastembed", "BAAI/bge-small-en-v1.5", "embedding", "BAAI/bge-small-en-v1.5"),
    ],
)
def test_litellm_model_string(provider, model, kind, expected):
    assert litellm_model_string(provider, model, kind) == expected


def test_prefix_is_not_doubled():
    assert litellm_model_string("ollama", "ollama_chat/llama3.1:8b") == "ollama_chat/llama3.1:8b"


def test_unknown_provider():
    with pytest.raises(UnknownProviderError):
        litellm_model_string("nope", "x")


def test_bedrock_secret_maps_to_aws_param_and_extra_is_filtered():
    kw = build_call_kwargs(
        "bedrock",
        api_key="SECRET",
        api_base=None,
        extra={"aws_access_key_id": "AKIA", "aws_region_name": "eu-west-1", "evil": "x"},
    )
    assert kw == {
        "aws_secret_access_key": "SECRET",
        "aws_access_key_id": "AKIA",
        "aws_region_name": "eu-west-1",
    }


def test_api_base_trailing_slash_stripped():
    kw = build_call_kwargs("ollama", api_key=None, api_base="http://localhost:11434/", extra={})
    assert kw == {"api_base": "http://localhost:11434"}


def test_public_registry_hides_internals_and_lists_all():
    reg = public_registry()
    assert {p["type"] for p in reg} == set(PROVIDERS)
    for p in reg:
        assert "litellm_prefix" not in p and "api_key_param" not in p


def test_every_provider_has_a_kind_and_cloud_ones_need_a_key():
    for spec in PROVIDERS.values():
        assert spec.kinds
        if not spec.local:
            assert any(f.secret and f.required for f in spec.fields), spec.type


def test_role_kinds():
    assert Role.EMBEDDING.kind == "embedding"
    assert Role.SQL_GENERATION.kind == "chat"
