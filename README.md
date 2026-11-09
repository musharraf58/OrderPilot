# OrderPilot

OrderPilot is an AI support agent built with FastAPI and Python. It will answer
customer questions about orders by calling an existing e-commerce backend through
LLM tool calling. It is a portfolio/learning project that runs entirely on free tiers.

## Run locally with Docker Compose

```bash
cp .env.example .env        # optional; compose has working defaults
docker compose up --build
```

Then:

- `curl localhost:8000/health` returns `{"status":"ok"}`
- `curl localhost:8000/health/ready` checks the database (503 if unreachable)

If port 5432 is already in use on your machine, run `DB_PORT=5433 docker compose up --build`.

## Run the tests

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests use a stubbed DB session, so no database is needed.

## Roadmap

- [x] Phase 0: Project foundation (config, logging, health checks, Docker, CI)
- [ ] Phase 1: TBD
- [ ] Phase 2: TBD
- [ ] Phase 3: TBD
- [ ] Phase 4: TBD
- [ ] Phase 5: TBD
- [ ] Phase 6: TBD
- [ ] Phase 7: TBD
- [ ] Phase 8: TBD
