# Smart Agents

Give your agents a durable record of mistakes, obscure discoveries, and the fixes that worked. Retrieve those lessons before similar work, with source IDs that can be checked and cited.

Smart Agents brings the operational Corpus mechanism into one repository: capture and classification, PostgreSQL/pgvector storage, read-only MCP retrieval, relevance ranking, local reranker training, weekly scheduling, feedback, consolidation, and deployment.

## How it works

1. Capture a correction or verified discovery, including enough context to recognize the problem later.
2. Validate, deduplicate and embed it into your private `learnings` table.
3. Before unfamiliar work, risky actions or retrying a failure, your agent calls `corpus_search`.
4. Vector search retrieves candidates. A small cross-encoder reranks them; weighted diversity ranking is the fallback.
5. The agent receives actual lessons and row IDs, checks their applicability, applies them and cites the IDs.
6. Weekly training can improve the cross-encoder using labeled task/lesson examples. Evaluation gates control promotion; feedback exposes mistakes and stale advice.

The database stores the lessons. The trained model learns **which stored lessons fit a task**. It does not replace the database or invent new facts. Training is separate from ingestion.

A memory summary can omit an unusual discovery. The explicit lesson-recording path preserves it without depending on summary memory. This system helps reduce repeated mistakes; capture coverage and agent compliance must still be measured.

## Start here

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and PostgreSQL with pgvector. Docker is an option for an isolated development database.

```sh
uv sync --extra dev
cp .env.example .env
# Edit .env for an explicitly selected development database and providers.
docker compose up -d db
# .env.example uses synthetic local-development credentials only.
uv run --env-file .env python -m smart_agents.install_schema --dsn-env SMART_AGENTS_TEST_DSN
uv run --env-file .env uvicorn corpus_api.main:app --host 127.0.0.1 --port 8210
```

The API can start without a classifier; administrative writes stay unavailable until configured. Retrieval requires the configured embedding service. Install `--extra rerank` and select a downloaded model to enable the cross-encoder. [Setup](docs/setup.md) covers provider configuration, source capture and local model installation.

Connect an agent with this stdio MCP entry (use absolute paths):

```json
{"mcpServers":{"corpus":{"command":"/absolute/path/smart-agents/.venv/bin/smart-agents-mcp","env":{"CORPUS_URL":"http://127.0.0.1:8210"}}}}
```

Install the [agent instructions](examples/agent-instructions.md) in the host's applicable instruction file. MCP availability and successful source capture are separate checks.

## Verify

```sh
~/.local/bin/woodpecker-cli exec --local --backend-engine local --repo-path "$PWD" .woodpecker.yaml
# Then run any checks outside the pipeline coverage:
make test
# Optional real database plumbing with synthetic evidence:
SMART_AGENTS_TEST_DSN=postgresql://postgres:synthetic-test-only@127.0.0.1:55439/smart_agents make integration
# Explicit model tests require installed weights and may load the local cross-encoder:
SMART_AGENTS_MODEL_TESTS=1 uv run pytest tests/test_cross_encoder.py
```

Unit tests use deterministic providers. Database tests prove persistence and API plumbing; they do not establish semantic quality. Training, model promotion, deployment and host capture each need their own runtime evidence.

## Documentation

- [Architecture and component map](docs/architecture.md)
- [Installation and local model setup](docs/setup.md)
- [Configuration](docs/configuration.md)
- [Capture and source coverage](docs/capture.md), [host coverage matrix](docs/coverage.md)
- [Retrieval and API/MCP reference](docs/retrieval.md)
- [Weekly training and evaluation](docs/training.md)
- [Operations and runtime cutover](docs/operations.md), [cutover checklist](docs/cutover.md)
- [References and source provenance](docs/references.md)
- [Security](SECURITY.md), [contributing](CONTRIBUTING.md)

MIT licensed. Private lesson contents, transcripts, credentials, health functionality and model weights are excluded. The cross-encoder's weights have their own upstream license.
