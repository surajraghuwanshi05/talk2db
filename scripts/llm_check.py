"""
Check every configured model with a tiny real call.

    uv run python -m scripts.llm_check

Runs migrations + env bootstrap first, so it works on a fresh clone with just a
.env. Prints one line per role: OK/FAIL, latency, and the reply or error.
"""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.db.app_db import session_factory
from app.db.migrate import upgrade_to_head
from app.llm.bootstrap import bootstrap_from_env
from app.llm.health import test_model
from app.llm.registry import Role
from app.llm.resolver import NoModelConfiguredError, resolve_model
from app.utils.logger import configure_logging


async def main() -> int:
    # Windows consoles default to cp1252 and choke on "→"/"·" in model labels.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    configure_logging("WARNING")
    upgrade_to_head(settings.APP_DB_URL)

    failures = 0
    with session_factory()() as session:
        bootstrap_from_env(session, settings)
        tested: dict[int, str] = {}
        for role in Role:
            try:
                resolved = resolve_model(session, role)
            except NoModelConfiguredError as e:
                print(f"  {role.value:<20} — not configured ({e})")
                failures += role is Role.SQL_GENERATION
                continue
            if resolved is None:
                print(f"  {role.value:<20} — not configured (BM25-only retrieval)")
                continue
            if resolved.model_id in tested:
                print(f"  {role.value:<20} = same model as {tested[resolved.model_id]}")
                continue
            tested[resolved.model_id] = role.value
            r = await test_model(resolved)
            status = "OK  " if r.ok else "FAIL"
            where = "local" if resolved.is_local else "cloud"
            print(
                f"{status} {role.value:<20} {r.model} [{where}] {r.latency_ms:.0f}ms "
                f"→ {r.detail or r.error}"
            )
            failures += not r.ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
