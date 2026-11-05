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
