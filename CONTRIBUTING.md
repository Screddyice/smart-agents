# Contributing

Use a dedicated synthetic corpus. Run `uv sync --extra dev`, `make test`, `make check` and `make build`. Optional model tests need the rerank extra and separately downloaded weights; database integration needs the explicitly selected isolated DSN.

Keep packages focused and preserve source provenance, row IDs, deliberate invocation, read-only MCP discovery and tested fallback ordering. Add regression cases for obscure mistakes, source-settlement failures, no-answer tasks and dataset leakage. Do not add a local generative-model fallback or a schedule that starts during installation.

Document provider/configuration changes, new data access and deployment effects. Keep code publication, successful checks and verified runtime behavior distinct in pull requests. Never attach private datasets, logs, checkpoints or populated environment files.
