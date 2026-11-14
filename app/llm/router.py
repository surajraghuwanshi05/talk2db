"""
generate_chat() — the ONE function the pipeline uses to talk to an LLM.

Provider-agnostic: the model comes from the user's configuration (resolver.py),
the call goes through LiteLLM.

Output contract is PLAIN TEXT (plan §15.4) so small local models work as well as
big cloud ones. Reasoning models' <think>…</think> blocks are stripped here, once,
so no stage has to care which model produced the text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import litellm
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.app_db import session_factory
from app.llm.registry import Role
from app.llm.resolver import ResolvedModel, resolve_model
from app.utils.logger import get_logger

logger = get_logger(__name__)

# LiteLLM is chatty by default and can phone home for telemetry; keep it quiet.
litellm.telemetry = False
litellm.suppress_debug_info = True

# LangSmith tracing is optional — a no-op decorator when it isn't installed.
try:  # pragma: no cover - depends on optional package
    from langsmith import traceable
except ImportError:  # pragma: no cover

    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMCallError(RuntimeError):
    """The provider call failed. Message is safe to show (no secrets)."""


@dataclass
class LLMResponse:
    text: str
    model: str  # "<provider name> · <model>"
    model_id: int
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


def strip_reasoning(text: str) -> str:
    """Drop <think>…</think> blocks (DeepSeek-R1, Qwen3, ... reasoning output)."""
    return _THINK_RE.sub("", text).strip()


async def _call(
    resolved: ResolvedModel, messages: list[dict], temperature: float, max_tokens: int | None
) -> LLMResponse:
    s = get_settings()
    start = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=resolved.litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=s.LLM_TIMEOUT_S,
            num_retries=s.LLM_NUM_RETRIES,
            **resolved.call_kwargs,
        )
    except Exception as e:
        logger.warning("LLM call failed", model=resolved.label, error=type(e).__name__)
        raise LLMCallError(f"{resolved.label}: {type(e).__name__}: {str(e)[:300]}") from e

    latency = (time.perf_counter() - start) * 1000
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    out = LLMResponse(
        text=strip_reasoning(content),
        model=resolved.label,
        model_id=resolved.model_id,
        tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
        tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        latency_ms=round(latency, 1),
    )
    logger.info(
        "LLM call",
        model=out.model,
        tokens_in=out.tokens_in,
        tokens_out=out.tokens_out,
        latency_ms=out.latency_ms,
    )
    return out


@traceable(name="generate_chat", run_type="llm")
async def generate_chat(
    messages: list[dict],
    *,
    role: Role = Role.SQL_GENERATION,
    model_id: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    session: Session | None = None,
) -> LLMResponse:
    """Run a chat completion with the model configured for `role`.

    `model_id` overrides the role assignment for this one call (chat dropdown).
    Raises NoModelConfiguredError if nothing is configured, LLMCallError on failure.
    """
    if session is not None:
        resolved = resolve_model(session, role, model_id)
    else:
        with session_factory()() as s:
            resolved = resolve_model(s, role, model_id)
    assert resolved is not None  # chat roles raise instead of returning None
    temp = resolved.temperature_default if temperature is None else temperature
    return await _call(resolved, messages, temp, max_tokens)


async def call_resolved(
    resolved: ResolvedModel,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Call an already-resolved model (used by health checks)."""
    return await _call(resolved, messages, temperature, max_tokens)
