# Smart Agents

The operational lessons system lives in this repository. Preserve API/MCP row IDs, explicit invocation, source provenance, retryable ingestion and immutable training outputs. The MCP is read-only. Classifiers must be tool-less; general local Qwen, private health modules and private datasets are outside scope.

Read docs/architecture.md, docs/coverage.md and docs/cutover.md before changing runtime paths. Check current live ownership rather than relying on historical audit notes. Do not claim training, promotion, deployment and working capture from the same evidence.

For Shawn-owned work, read ~/.config/local-ci/POLICY.md and prefer local Woodpecker, then documented repository checks, then GitHub checks. This repo uses `woodpecker-cli exec --local --backend-engine local --repo-path "$PWD" .woodpecker.yaml`; its image fields are local shell executables. Commands: make test, make integration (explicit isolated SMART_AGENTS_TEST_DSN), make check, make build; Node bridge: node --test integrations/node/bridge.test.mjs. Model tests require explicit SMART_AGENTS_MODEL_TESTS=1 and installed weights. No inference or schedules run on package installation.
