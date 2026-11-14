"""
"Test connection" for a configured model — a tiny real call.

Chat: ask for the word OK with a 5-token budget. Embedding: embed "ping" and
report the vector size (needed to size the vector store).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.llm.embeddings import embed_with_model
from app.llm.resolver import ResolvedModel
from app.llm.router import call_resolved


@dataclass
class ModelTestResult:
    ok: bool
    model: str
    kind: str
    latency_ms: float
    detail: str | None = None  # chat: reply snippet; embedding: "dim=384"
    error: str | None = None
    embedding_dim: int | None = None


async def test_model(resolved: ResolvedModel) -> ModelTestResult:
    start = time.perf_counter()
    try:
        if resolved.kind == "embedding":
            vecs = await embed_with_model(resolved, ["ping"])
            dim = len(vecs[0]) if vecs else 0
            return ModelTestResult(
                ok=dim > 0,
                model=resolved.label,
                kind="embedding",
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                detail=f"dim={dim}",
                embedding_dim=dim,
            )
        resp = await call_resolved(
            resolved,
            [{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=16,
        )
        return ModelTestResult(
            ok=True,
            model=resolved.label,
            kind="chat",
            latency_ms=resp.latency_ms,
            detail=resp.text[:50],
        )
    except Exception as e:
        return ModelTestResult(
            ok=False,
            model=resolved.label,
            kind=resolved.kind,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error=f"{type(e).__name__}: {str(e)[:300]}",
        )
