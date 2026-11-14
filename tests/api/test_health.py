from fastapi.testclient import TestClient

from app.main import create_app


def _client():
    return TestClient(create_app())


def test_health_degraded_without_llm():
    with _client() as c:
        body = c.get("/api/v1/health").json()
    assert body["status"] == "degraded"
    assert body["app_db"] is True
    assert body["llm_configured"] is False
    assert any("No chat model" in h for h in body["hints"])


def test_health_ok_after_env_bootstrap(set_env):
    set_env(LLM_PROVIDER="groq", LLM_MODEL="llama-3.1-8b-instant", LLM_API_KEY="gsk_secret_value")
    with _client() as c:
        r = c.get("/api/v1/health")
    body = r.json()
    assert body["status"] == "ok"
    sql_role = next(x for x in body["roles"] if x["role"] == "sql_generation")
    assert sql_role["model"].endswith("llama-3.1-8b-instant")
    assert sql_role["local"] is False
    assert "gsk_secret_value" not in r.text  # secrets never leave the server


def test_prod_refuses_to_start_without_secrets(set_env):
    set_env(APP_ENV="prod", SECRET_KEY="", API_KEYS="")
    try:
        with _client():
            raise AssertionError("should not start")
    except RuntimeError as e:
        assert "SECRET_KEY" in str(e)
