"""
Provider registry — the single place that knows about LLM vendors.

Each ProviderSpec says:
  • what the UI form must ask for (`fields`)
  • how to turn (provider, model_name) into a LiteLLM call (`litellm_prefix`, `api_key_param`)
  • which models to SUGGEST (hints only — the user may type any model name)

Adding a provider = adding one entry here. The rest of the app never branches on
vendor names. GET /api/v1/llm/registry (Phase 2) serialises this for the frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal


class Role(StrEnum):
    """Pipeline roles a model can be assigned to (plan §15.2)."""

    SQL_GENERATION = "sql_generation"
    QUESTION_REWRITE = "question_rewrite"
    ANSWER_FORMATTING = "answer_formatting"
    DESCRIPTION_DRAFT = "description_draft"
    EMBEDDING = "embedding"

    @property
    def kind(self) -> Literal["chat", "embedding"]:
        return "embedding" if self is Role.EMBEDDING else "chat"


CHAT_ROLES = [r for r in Role if r.kind == "chat"]


@dataclass(frozen=True)
class FieldSpec:
    """One input on the 'Add provider' form."""

    name: str  # key the API expects
    label: str
    target: Literal["api_key", "api_base", "extra"]  # where it's stored on LLMProvider
    required: bool = False
    secret: bool = False  # rendered as a password input; never echoed back
    default: str | None = None
    placeholder: str | None = None
    help: str | None = None


@dataclass(frozen=True)
class ProviderSpec:
    type: str
    label: str
    local: bool  # True → data never leaves the user's machine
    litellm_prefix: str  # "anthropic/" → "anthropic/<model>"; "" = none
    kinds: tuple[str, ...] = ("chat",)
    fields: tuple[FieldSpec, ...] = ()
    suggested_chat_models: tuple[str, ...] = ()
    suggested_embedding_models: tuple[str, ...] = ()
    embedding_prefix: str | None = None  # when embeddings need a different prefix
    api_key_param: str = "api_key"  # LiteLLM kwarg the stored secret maps to
    supports_discovery: bool = False  # can list models live (Phase 2)
    docs_url: str | None = None

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("api_key_param")
        d.pop("litellm_prefix")
        d.pop("embedding_prefix")
        return d


_API_KEY = FieldSpec("api_key", "API key", "api_key", required=True, secret=True)
_API_KEY_OPTIONAL = FieldSpec(
    "api_key", "API key", "api_key", secret=True, help="Only if your server requires one."
)

PROVIDERS: dict[str, ProviderSpec] = {
    spec.type: spec
    for spec in [
        # ── Local ────────────────────────────────────────────────────────────
        ProviderSpec(
            type="ollama",
            label="Ollama (local)",
            local=True,
            litellm_prefix="ollama_chat/",
            embedding_prefix="ollama/",
            kinds=("chat", "embedding"),
            fields=(
                FieldSpec(
                    "api_base",
                    "Base URL",
                    "api_base",
                    required=True,
                    default="http://localhost:11434",
                ),
            ),
            suggested_chat_models=(
                "qwen2.5-coder:7b",
                "qwen2.5-coder:14b",
                "qwen2.5-coder:32b",
                "llama3.1:8b",
                "sqlcoder:7b",
            ),
            suggested_embedding_models=("nomic-embed-text", "mxbai-embed-large"),
            supports_discovery=True,
            docs_url="https://ollama.com/library",
        ),
        ProviderSpec(
            type="fastembed",
            label="FastEmbed (local embeddings, no key)",
            local=True,
            litellm_prefix="",
            kinds=("embedding",),
            suggested_embedding_models=("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"),
            docs_url="https://qdrant.github.io/fastembed/",
        ),
        ProviderSpec(
            type="openai_compatible",
            label="OpenAI-compatible server (LM Studio, vLLM, llama.cpp, Together, ...)",
            local=True,  # usually local; the UI notes it may be remote
            litellm_prefix="openai/",
            kinds=("chat", "embedding"),
            fields=(
                FieldSpec(
                    "api_base",
                    "Base URL",
                    "api_base",
                    required=True,
                    placeholder="http://localhost:1234/v1",
                ),
                _API_KEY_OPTIONAL,
            ),
            supports_discovery=True,
        ),
        # ── Cloud ────────────────────────────────────────────────────────────
        ProviderSpec(
            type="openai",
            label="OpenAI",
            local=False,
            litellm_prefix="openai/",
            kinds=("chat", "embedding"),
            fields=(_API_KEY,),
            suggested_chat_models=("gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"),
            suggested_embedding_models=("text-embedding-3-small", "text-embedding-3-large"),
            supports_discovery=True,
            docs_url="https://platform.openai.com/docs/models",
        ),
        ProviderSpec(
            type="anthropic",
            label="Anthropic (Claude)",
            local=False,
            litellm_prefix="anthropic/",
            fields=(_API_KEY,),
            suggested_chat_models=("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"),
            docs_url="https://docs.anthropic.com/en/docs/about-claude/models",
        ),
        ProviderSpec(
            type="gemini",
            label="Google Gemini",
            local=False,
            litellm_prefix="gemini/",
            kinds=("chat", "embedding"),
            fields=(_API_KEY,),
            suggested_chat_models=("gemini-2.5-pro", "gemini-2.5-flash"),
            suggested_embedding_models=("text-embedding-004",),
            docs_url="https://ai.google.dev/gemini-api/docs/models",
        ),
        ProviderSpec(
            type="groq",
            label="Groq",
            local=False,
            litellm_prefix="groq/",
            fields=(_API_KEY,),
            suggested_chat_models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
            supports_discovery=True,
            docs_url="https://console.groq.com/docs/models",
        ),
        ProviderSpec(
            type="mistral",
            label="Mistral",
            local=False,
            litellm_prefix="mistral/",
            kinds=("chat", "embedding"),
            fields=(_API_KEY,),
            suggested_chat_models=("mistral-large-latest", "codestral-latest"),
            suggested_embedding_models=("mistral-embed",),
            docs_url="https://docs.mistral.ai/getting-started/models/",
        ),
        ProviderSpec(
            type="deepseek",
            label="DeepSeek",
            local=False,
            litellm_prefix="deepseek/",
            fields=(_API_KEY,),
            suggested_chat_models=("deepseek-chat", "deepseek-reasoner"),
        ),
        ProviderSpec(
            type="openrouter",
            label="OpenRouter (one key, many models)",
            local=False,
            litellm_prefix="openrouter/",
            fields=(_API_KEY,),
            suggested_chat_models=(
                "openai/gpt-4.1-mini",
                "anthropic/claude-sonnet-5-5",
                "meta-llama/llama-3.3-70b-instruct",
            ),
            supports_discovery=True,
            docs_url="https://openrouter.ai/models",
        ),
        ProviderSpec(
            type="azure",
            label="Azure OpenAI",
            local=False,
            litellm_prefix="azure/",
            kinds=("chat", "embedding"),
            fields=(
                _API_KEY,
                FieldSpec(
                    "api_base",
                    "Endpoint",
                    "api_base",
                    required=True,
                    placeholder="https://<resource>.openai.azure.com",
                ),
                FieldSpec(
                    "api_version", "API version", "extra", required=True, default="2024-10-21"
                ),
            ),
            # model_name for Azure = your DEPLOYMENT name
        ),
        ProviderSpec(
            type="bedrock",
            label="AWS Bedrock",
            local=False,
            litellm_prefix="bedrock/",
            kinds=("chat", "embedding"),
            fields=(
                FieldSpec("aws_access_key_id", "Access key ID", "extra", required=True),
                FieldSpec("api_key", "Secret access key", "api_key", required=True, secret=True),
                FieldSpec("aws_region_name", "Region", "extra", required=True, default="us-east-1"),
            ),
            api_key_param="aws_secret_access_key",
            suggested_embedding_models=("amazon.titan-embed-text-v2:0",),
        ),
    ]
}


class UnknownProviderError(ValueError):
    pass


def get_spec(provider_type: str) -> ProviderSpec:
    try:
        return PROVIDERS[provider_type]
    except KeyError as e:
        raise UnknownProviderError(
            f"Unknown provider type '{provider_type}'. Supported: {', '.join(PROVIDERS)}"
        ) from e


def litellm_model_string(provider_type: str, model_name: str, kind: str = "chat") -> str:
    """('anthropic', 'claude-sonnet-5-5') → 'anthropic/claude-sonnet-5-5'.

    Idempotent: a model name that already carries the prefix is left alone, so a
    user who types 'ollama_chat/llama3.1:8b' doesn't get it doubled.
    """
    spec = get_spec(provider_type)
    prefix = (
        spec.embedding_prefix
        if (kind == "embedding" and spec.embedding_prefix)
        else spec.litellm_prefix
    )
    if not prefix or model_name.startswith(prefix):
        return model_name
    return prefix + model_name


def build_call_kwargs(
    provider_type: str,
    *,
    api_key: str | None,
    api_base: str | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Credentials/endpoint kwargs for litellm.acompletion / aembedding."""
    spec = get_spec(provider_type)
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs[spec.api_key_param] = api_key
    if api_base:
        kwargs["api_base"] = api_base.rstrip("/")
    allowed_extra = {f.name for f in spec.fields if f.target == "extra"}
    for k, v in (extra or {}).items():
        if k in allowed_extra and v not in (None, ""):
            kwargs[k] = v
    return kwargs


def public_registry() -> list[dict[str, Any]]:
    """What the frontend needs to render the 'Add provider' form."""
    return [spec.to_public() for spec in PROVIDERS.values()]
