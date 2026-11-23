# Capture obscure lessons

A downstream corpus cannot recover evidence that was never captured. The cmem adapter reads distilled observations and summaries, enriches full summary fields, and excludes raw user prompts. It is not a raw-transcript collector. Verify each host/profile's actual capture separately.

Use the explicit recording command when an unusual discovery might disappear in a summary:

```sh
smart-agents record /private/path/lesson.json
```

The JSON includes `what_happened`, `correction`, `agent_id`, optional `project`, `source_session_id` and `source` (`shawn`, `self`, `observed`). Describe the trigger, failed assumption, fix and validation. Keep evidence minimal; do not include passwords, keys, customer data or unrelated personal content. A synthetic example is in `examples/lesson.json`.

The authenticated `/corrections` path checks significance, exact duplicates and semantic near-duplicates before inserting. A self-discovered reproducible constraint can be significant without an explicit human correction. Backend unavailability must be retried from the retained evidence, not treated as a settled rejection. Persisted IDs are returned on success.

For cmem, run:

```sh
corpus-ingest --dry-run --limit 3
# Only after verifying source scope, classifier identity and selected database:
corpus-ingest --limit 3
# Continuous polling is opt-in:
corpus-ingest --poll 900
```

Keep one source item per classifier chunk. Checkpoint only settled outcomes. Backend failures remain retryable. Deduplication must not silently discard several distinct lessons in one category. Failed embeddings must leave the item retryable rather than permanently invisible to search.

Feedback uses `smart-agents feedback /private/path/feedback.json`. Required fields: `lesson_id`, `task`, `outcome`, `evidence`; optional source/replacement references retain provenance. Outcomes are `applied_successfully`, `repeated_mistake`, `irrelevant`, `incorrect`, `superseded`. Feedback is stored pending review. It does not rewrite lesson text or declare a generated claim correct. Resolve it with `smart-agents review-feedback /private/review.json`, naming `feedback_id`, `status` (`accepted`/`rejected`), `reviewer` and `note`. An already-reviewed record cannot be silently overwritten.

For consolidation, `python -m smart_agents.consolidation.learnings` reports potential duplicates. An explicit apply pass marks supersession; it does not delete old rows. Review conditions carefully: two similar lessons may depend on different obscure constraints. Preserve cited row IDs and evidence. Use feedback to create regression cases after human/operator review.

Uncertain judgments, malformed model responses and requests with uncertain billing are preserved as private `pending-review/*.json` source-reference records. They are not automatically sent to the provider again. Review those source IDs in the authorized memory feed, reconstruct only the needed evidence and use the explicit recording command after review. Keep the queue record and resulting lesson ID in the private operator receipt. No historical sweep is needed.

## Paired CMEM replica and read-only recall

Use this optional flow when an authorized Claude-Mem worker already syncs observations to a private replica. The publisher reads SQLite in read-only mode and checks the originating device's paths against explicit POSIX workspace roots. One configuration can cover multiple projects under those roots. Relative file paths alone are insufficient. The publisher filters text and paths locally, then submits a bounded excerpt to JEV. Use roots approved for both the configured providers and the recall audience. Narrow the scope further with repeatable `--exclude-term` arguments.

Set the private Corpus DSN, explicit `SMART_AGENTS_ORG` and `SMART_AGENTS_INSTANCE` labels, and the existing JEV and embedding provider settings. Use the same paths and namespace for all runs on one replica.

```sh
corpus-ingest-cmem-replica \
  --source-db /private/cmem/claude-mem.db \
  --publication-db /private/smart-agents/cmem-publication.sqlite3 \
  --replica-id operator-assigned-replica \
  --workspace-root /workspace/approved-projects \
  --poll 900
```

The first run establishes a fresh-only watermark. Later batches scan up to 1,024 rows and evaluate up to five eligible observations per pass. Omit `--poll` for a single pass. For paired workers that pull on context injection, pass the worker's local port and an authorized project via `--refresh-port` and `--refresh-project`. This sends a loopback HEAD request before each scan. The operator manages pairing, scheduling and private host configuration.

JEV approval requires a probability of at least 0.80 and a receipt. The publisher records the receipt before writing the exact approved excerpt as an embedded Corpus lesson. Interrupted or uncertain judgments go to the private `needs_review` table without resubmission. A failed Corpus write or publication can resume from the saved approval. Recall becomes available after the writer reads back the matching embedded lesson and commits its publication record.

```sh
smart-agents-cmem-mcp \
  --source-db /private/cmem/claude-mem.db \
  --publication-db /private/smart-agents/cmem-publication.sqlite3
```

The stdio server exposes only `recall_cmem_memory` and opens both databases read-only. It checks the live observation against the published excerpt and digest and returns the CMEM observation ID and the Corpus lesson ID. Give the connector process read access only and route remote clients through your existing authenticated gateway. Keep the publication database and gateway bindings private. Use a separate publication database per authorized recall audience. For a live check, capture a new in-scope observation, confirm its Corpus lesson ID and ask each authenticated client about the fact using a natural-language query.
