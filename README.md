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

## Tool design

The tools are the only way the future agent touches the store, so they are built to be
safe even if the model misbehaves.

- **Identity never comes from the model.** Every tool call receives a `UserContext`
  (`customer_id`, `store_token`) as a separate argument. Our code builds it from the
  signed-in user's token, and `customer_id` is read from the store's `/auth/me`, never
  from user or model input. It is not part of any tool's schema, so the model cannot
  see, choose or change whose data a tool acts on. The token is a `SecretStr`, so it stays
  out of logs and reprs.
- **The model's arguments are validated and closed.** Each tool has a Pydantic input model
  that rejects unknown fields (so a made-up `customer_id` argument is an error) and bad
  types.
- **Ownership is checked in our code.** The store's orders carry no owner field, so
  `get_order` and `cancel_order` first confirm the id appears in the caller's own
  `GET /orders` list. Foreign and nonexistent ids give the same `not_found` result, so
  nothing leaks about other customers' orders. The store remains the final authority and
  enforces ownership on its side as well.
- **Tools never raise.** Every tool returns `{ok, data, error: {code, message}}` with stable
  error codes (`not_found`, `not_cancellable`, `store_unavailable`, `invalid_arguments`, ...).
- **Reads and writes are marked** (`kind`), so the agent loop can add confirmation for
  writes. Only `cancel_order` writes, and only `pending` orders can be cancelled.
- **Outputs are compact**, with only the fields the model needs, to save tokens.

Try the tools against a real store with `uv run python scripts/try_tools.py` (needs
`STORE_API_URL` and either `STORE_TOKEN` or `STORE_EMAIL`/`STORE_PASSWORD`; see
`.env.example`). It is not run by pytest or CI.

## Roadmap

- [x] Phase 0: Project foundation (config, logging, health checks, Docker, CI)
- [x] Phase 1: Tool layer (store client, typed tools, registry)
- [ ] Phase 2: TBD
- [ ] Phase 3: TBD
- [ ] Phase 4: TBD
- [ ] Phase 5: TBD
- [ ] Phase 6: TBD
- [ ] Phase 7: TBD
- [ ] Phase 8: TBD
