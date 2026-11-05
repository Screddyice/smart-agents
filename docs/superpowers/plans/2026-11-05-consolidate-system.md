# Smart Agents Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make one public repository own the operational lessons learning loop and its integrations.

**Architecture:** Extract the existing tested Python retrieval, classifier, ingestion, evaluation and reranker components, preserving compatibility package names where consumers depend on them. Add portable configuration, trusted administrative feedback, immutable training orchestration and a read-only MCP. Keep records and weights private.

**Tech Stack:** Python 3.11+, FastAPI, psycopg/pgvector, MCP stdio, PyTorch/Transformers for optional local reranking/training.

**Spec:** ../specs/2026-11-05-smart-agents-design.md

## Global Constraints

No private records, raw transcripts, credentials, weights or prior Git history in public files. No local Qwen invocation, health-runtime changes or overwritten checkpoints. Keep explicit agent invocation, MCP tool name corpus_search and lesson row IDs. Preserve running startup paths until replacements pass checks. Source cutover does not authorize new broad data ingestion or model-provider changes.

## Review Focus

- Provider failure must remain retryable and must not settle a source cursor.
- Duplicate or paraphrased lessons and related tasks must not leak across dataset partitions.
- Reranker failures and no-answer cases must preserve baseline behavior.
- A lesson or feedback payload must not execute instructions or rewrite evidence without review.
- New repository ownership must include ingestion, training and consumers, not only the API.

### Task 1: Extract portable packages and existing behavior tests

**Files:** corpus_api/, learning_classifier/, learning_corpus_pipeline/, smart_agents/, tests/, pyproject.toml, sql/schema.sql, docs/provenance.md.
**Interfaces:** Preserve corpus_api.main:app and corpus-ingest; namespace evaluation/reranker modules under smart_agents.

- [ ] Copy selected original tests first, run and observe missing implementation.
- [ ] Extract code from allowlisted files only, record source hashes, replace retired/private assumptions with explicit configuration.
- [ ] Run pytest; keep tests independent of private fixtures and local inference.
- [ ] Commit tested extraction.

### Task 2: MCP, trusted recording and feedback

**Files:** smart_agents/mcp.py, smart_agents/admin.py, tests/test_mcp.py, tests/test_feedback.py, examples/agent-instructions.md.
**Interfaces:** corpus_search(task, limit) -> MCP text with row IDs; admin record posts /corrections; admin feedback posts /feedback; MCP exposes no mutations.

- [ ] Write and run failing tests for row formatting, validation, timeouts, read-only discovery, feedback validation and immutable feedback insertion.
- [ ] Implement configuration-driven adapters and administrative endpoints.
- [ ] Verify synthetic ingestion -> retrieval -> MCP using isolated Postgres when available; verify no-answer and correction-feedback behavior.
- [ ] Commit passing integration.

### Task 3: Weekly local reranker lifecycle

**Files:** smart_agents/training.py, smart_agents/weekly.py, deploy/launchd/, deploy/systemd/, tests/test_training.py.
**Interfaces:** smart-agents weekly --dry-run prepares an immutable version; actual training is explicit and uses the cross-encoder, not a general-purpose local Qwen model.

- [ ] Write failing tests for grouped data partitions, append-only artifacts, promotion gates, locks and failed run state.
- [ ] Extract trainer and dataset builders; implement resource/configuration-driven orchestration, versioned promotion, threshold and hash lineage, rollback metadata.
- [ ] Verify deterministic tests and metadata-only readiness checks without running local inference or training.
- [ ] Commit training/scheduling facilities.

### Task 4: Public documentation, clean setup and release

**Files:** README.md, LICENSE, SECURITY.md, docs/{architecture,setup,configuration,capture,retrieval,training,operations,references,coverage}.md, compose.yaml, .github/workflows/test.yml.
**Interfaces:** make test; isolated public quickstart; explicit installation commands for operator services.

- [ ] Write full setup and subsystem documentation, provenance and third-party notices; use synthetic examples.
- [ ] Check links, package build, source compile, complete pytest suite, staged privacy/secret scan and fresh install.
- [ ] Review the complete branch with a fresh reviewer; fix supported findings and retest.
- [ ] Create public Screddyice/smart-agents, publish verified head and check anonymous accessibility.

### Task 5: Runtime ownership and references

**Files:** deploy/install-runtime.sh, docs/cutover.md and a private per-component cutover receipt.
**Interfaces:** pinned repository revision for API, ingestion, trainer, scheduler and MCP consumers; old targets retained for rollback.

- [ ] Inventory exact live service/provider configuration and last failure before mutation.
- [ ] Prepare replacement paths and run isolated health/MCP checks; do not activate inactive ingestion or training merely to move source ownership.
- [ ] Cut over existing consumers and services without changing database records or provider identity; verify exact loaded paths, model artifact and task result.
- [ ] Update narrow owning references. Report inactive/restoration-dependent facilities honestly and retain evidence in work state.

## Execution record

The user explicitly directed consolidation into one repository after approving the design direction. Execute natively in the new, isolated repository; do not stop for an additional workflow-selection question. Task completion is tracked in ClaudeMem and the execution ledger.
