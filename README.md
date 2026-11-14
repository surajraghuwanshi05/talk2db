# talk2db

Ask your database questions in plain English. Get back the SQL, the rows, and an answer —
with every step of the pipeline visible.

- **Bring your own LLM** — OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, OpenRouter, Azure,
  Bedrock, any OpenAI-compatible server, or **fully local with Ollama** (no API key).
- **Safe by construction** — read-only connections, AST-validated `SELECT`-only SQL, timeouts, row caps.
- **Transparent** — see which tables were picked, the exact schema the model saw, the SQL, and every retry.

> **Status:** early development. The foundation (API, config, bring-your-own-LLM layer,
> metadata DB, Docker) is done; the query pipeline is next. See [Roadmap](#roadmap).

## How it works

```
question
  → pick relevant tables      keyword (BM25) + semantic (embeddings) search over table "cards"
  → expand along foreign keys add the join/bridge tables the query will need
  → build schema context      CREATE-TABLE-like text: columns, types, enums, relationships
  → generate SQL              your configured LLM — or abstain if the schema can't answer
  → validate                  parsed as SQL: single SELECT only, known tables/columns, LIMIT added
  → self-correct              on a validation/database error, retry with the error as feedback
  → execute                   read-only connection, statement timeout, row cap
  → answer                    rows + plain-English summary + the full trace of every step
```

---

## Quickstart (local Python)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```bash
git clone https://github.com/surajraghuwanshi05/talk2db.git && cd talk2db
uv sync --all-extras                      # creates .venv, installs everything
cp .env.example .env                      # optional — defaults work for local dev
uv run python -m scripts.download_sample_dbs   # Chinook sample database
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs (API) or check http://localhost:8000/api/v1/health.

### Pick an LLM

Set these in `.env` (a Settings → Models UI arrives in Phase 2):

```bash
# Fully local (install Ollama, then: ollama pull qwen2.5-coder:7b)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5-coder:7b

# …or any cloud provider, e.g.
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5-5
LLM_API_KEY=sk-ant-...
```

Then verify with a real call:

```bash
uv run python -m scripts.llm_check
```

Embeddings default to **local fastembed** (no key; the ~70 MB model downloads on first use).

## Quickstart (Docker)

```bash
uv run python -m scripts.download_sample_dbs
docker compose --profile local up -d      # API + Postgres + Chinook + Ollama
```

Without `--profile local`, set a cloud provider in `.env` first. All ports bind to `127.0.0.1`.

| Service | Port | Purpose |
|---|---|---|
| api | 8000 | talk2db API |
| app-db | 5432 | metadata (datasources, models, history) |
| chinook-db | 5433 | sample DB — read-only user `nl2sql_reader` / `reader` |
| ollama | 11434 | local LLM (`local` profile) |
| qdrant | 6333 | vector store (`qdrant` profile) |

## Development

```bash
uv run pytest                 # tests (no network, no API keys needed)
uv run ruff check . && uv run ruff format .
uv run pre-commit install     # lint + secret scan on every commit
```

## Project structure

```
app/
  main.py            FastAPI app (startup: migrations, LLM bootstrap)
  config.py          settings from env / .env
  api/v1/            HTTP endpoints
  db/                metadata DB: ORM models, session, migrations runner
  llm/               bring-your-own-model layer
    registry.py        supported providers + the fields each needs
    resolver.py        pipeline role → configured model + credentials
    router.py          generate_chat() — every LLM call goes through here (LiteLLM)
    embeddings.py      local fastembed or any provider; None → keyword-only retrieval
  utils/             logging, secret encryption
migrations/          Alembic migrations for the metadata DB
scripts/             llm_check (test your models), download_sample_dbs
sample_dbs/          Postgres init for the Chinook sample (data is downloaded, not committed)
tests/               pytest — no network or API keys needed
```

## Roadmap

- [x] Foundation: FastAPI, config, metadata DB + migrations, bring-your-own-LLM layer, Docker, tests
- [ ] Query pipeline: schema introspection, table retrieval, SQL generation, validation, self-correction, execution
- [ ] Multiple datasources, Settings → Models API, query history & feedback, accuracy evaluation
- [ ] Few-shot examples, follow-up questions, streaming, web UI
- [ ] Large schemas (Qdrant), business-glossary RAG, plugins (row-level security, PII masking)

## Security

- Generate a `SECRET_KEY` before storing real API keys — they're encrypted with it at rest.
- Set `API_KEYS` for anything beyond your own machine. `APP_ENV=prod` refuses to start without both.
- Point talk2db at databases with a **read-only** user.

## License

MIT — see [LICENSE](LICENSE). The Chinook sample database is © Luis Rocha, under its own
[license](https://github.com/lerocha/chinook-database/blob/master/LICENSE.md).
