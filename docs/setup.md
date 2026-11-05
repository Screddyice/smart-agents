# Setup

## Base system

Use Python 3.11 or newer. `uv sync --extra dev` installs from the checked-in lockfile into `.venv`. No model download, training job, agent hook or schedule runs during installation. For a wheel installation, use `pip install smart-agents-corpus` only after a package release exists; the Git repository is the current distribution source.

Choose a dedicated operational database, enable pgvector, then apply `sql/schema.sql` only to an empty database. The helper `uv run --env-file .env python -m smart_agents.install_schema` refuses an existing `learnings` table. Existing deployments need a reviewed additive migration, not a reset. Transaction-pooler connections disable psycopg prepared statements.

Run the API bound to loopback. Place remote access behind an authenticated gateway or a private tunnel. Configure `CORPUS_ADMIN_TOKEN` privately for administrative commands. The MCP exposes retrieval only.

## Embeddings and classification

The compatible embedding provider is Ollama's `/api/embeddings` with `nomic-embed-text` and 768 dimensions. Configure its address with `OLLAMA_HOST`; a remote private service can be used. Installing or starting a general local language model is unnecessary. A different embedding space requires a planned schema/re-embedding migration, including retrieval threshold evaluation.

Classification is opt-in. Select `LEARNING_CLASSIFIER_BACKEND=compatible` for an explicitly configured tool-less chat-completions endpoint, or `jev` for typed category judgments of already-distilled items. Configure the approved provider/account before enabling ingestion. The compatible endpoint receives no tools and JEV returns typed probabilities only. Native agent subprocesses are not classification backends. Unknown/disabled backends raise an error instead of falling back to Qwen. Historical local-classifier code is not an approved fallback. No provider credentials are supplied by this repository.

## Local relevance model

This is a small PyTorch cross-encoder, not the separate Qwen/MLX health model. Install:

```sh
uv sync --extra dev --extra rerank
uv run python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("cross-encoder/ms-marco-MiniLM-L6-v2", revision="233902d25c440f23af6f7d6e94d2946bac0bee0a"))'
```

The command downloads weights into the Hugging Face cache. Point `RERANK_MODEL` at the returned directory and set `RERANK_ENABLED=1`. Keep the weights private/outside Git. The upstream model metadata identifies Apache-2.0; see [references](references.md). The fixed revision prevents a moving download from silently changing your base. For training, pass this local directory as `--base` too.

Serving defaults to two CPU threads, a pool of 12 candidates, length 384 and a 1.5-second reranking budget. Training uses Apple MPS when available, otherwise CPU. Resource needs depend on platform and batch size; start with the documented batch size 16 on a dedicated window. No CUDA or other accelerator is required by the code. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` allow operation after the cache is populated; missing weights then fail rather than download.

Training and inference should run in separate resource windows. The weekly process has a lock to prevent duplicate training runs. It never starts or stops another app or local model. Coordinate other GPU users before opting in.

## Capture and weekly scheduling

Set `CMEM_PRO_TOKEN` privately and test `corpus-ingest --dry-run --limit 3` first. A dry run fetches/groups but does not classify or write. Validate source scope and actual new-row persistence before scheduling a bounded ingestion pass. See [capture](capture.md).

Prepare reviewed private triples and run `smart-agents weekly --triples /private/path/triples.jsonl --dry-run`. Inspect the new versioned manifest. An actual training run removes `--dry-run`; it may take substantial compute and invokes the cross-encoder. No Qwen model is involved. Scheduler templates are opt-in and require substituted absolute paths. See [training](training.md).
