# Configuration

| Variable | Purpose |
|---|---|
| `RDS_URL` or `LEARNING_CORPUS_RDS_URL` | Private Postgres connection string; legacy name does not require AWS |
| `CORPUS_URL` | Direct API base URL, loopback by default |
| `CORPUS_ADMIN_TOKEN` | Required bearer token for administrative writes and training export |
| `CORPUS_READ_TOKEN` | Optional bearer header to a read gateway |
| `CORPUS_GATEWAY_URL` | Optional existing JSON-RPC gateway URL for remote MCP routing |
| `CORPUS_GATEWAY_TOKEN_ENV` | Name of the environment variable holding gateway bearer token |
| `CORPUS_GATEWAY_ENV_FILE` | Explicit private env file fallback for that token; never sourced as shell code |
| `CORPUS_GATEWAY_TOOL` | Upstream gateway tool name, default `mcp.corpus.corpus_search` |
| `CORPUS_TIMEOUT_S` | Direct retrieval HTTP timeout, default 30 seconds |
| `OLLAMA_HOST` | Embedding service address; no generative model is started |
| `LEARNING_CORPUS_EMBED_MODEL` | Compatible embedding model, default `nomic-embed-text` |
| `LEARNING_CLASSIFIER_BACKEND` | Explicit `compatible`/`jev`; disabled by default |
| `SMART_AGENTS_CLASSIFIER_URL`, `SMART_AGENTS_CLASSIFIER_MODEL`, `SMART_AGENTS_CLASSIFIER_KEY` | Explicit tool-less chat-completions provider configuration |
| `SMART_AGENTS_JEV_BIN` | Installed typed JEV stdio adapter, default `~/.local/bin/jev-mcp` |
| `SMART_AGENTS_LABEL_BUDGET_USD` | Stop further labeling calls after the recorded budget is reached; provider hard spending limits remain authoritative |
| `CMEM_PRO_TOKEN`, `CMEM_MCP_URL` | Distilled memory source credentials and endpoint |
| `SMART_AGENTS_MIN_FREE_BYTES` | Minimum free space before local training, default 2 GiB |
| `SMART_AGENTS_STATE_DIR` | Private cursor/training state directory |
| `SMART_AGENTS_DATA_DIR` | Private dataset/evaluation root |
| `SMART_AGENTS_ORG`, `SMART_AGENTS_INSTANCE` | Explicit scope/provenance of new writes |
| `SMART_AGENTS_TRIPLES` | Private training triples file for scheduled weekly runs |
| `SMART_AGENTS_REVISION` | Exact deployed source revision reported in health/manifests |
| `RERANK_ENABLED` | `1` enables the cross-encoder; `0` selects MMR |
| `RERANK_MODEL` | Local stock or promoted model directory |
| `RERANK_MIN_SCORE` | Abstention threshold calibrated with the specific model |
| `RERANK_POOL`, `RERANK_THREADS`, `RERANK_BUDGET_S` | Candidate cap, CPU threads and scoring budget |
| `RERANK_DEDUPE_COS` | Similarity cutoff for duplicate suppression, default 0.90 |

Do not commit populated environment files. Empty path settings should be omitted rather than set to an empty string. The checked-in `.env.example` is for development only. Configure the same state path after a code move to retain ingestion cursors. Retain legacy organization/source vocabularies when using an existing constrained database; changing labels is a migration, not a path update.
