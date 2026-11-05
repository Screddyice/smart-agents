# Operations

Run API and ingestion as separate services with separate least-privilege credentials. Keep retrieval reads independent of the administrative token. The model/data/state directories are private and external to the checkout. Use a pinned source revision and pinned model revision/hash.

Record four independent readiness states: capture, database/retrieval, training, and deployed model. Monitor source cursor age, retry/backend errors, missing embeddings, empty results, reranker fallback, weekly attempt/completion, rejected candidates and repeated-mistake feedback. A successful health request is not an end-to-end test.

If ingestion stops, first inspect the exact unit's enabled/active state, configured source, classifier CLI availability/authentication, model identity, database connectivity and last settled cursor. A disabled old unit may be deliberate. Do not blindly replay all history or activate a new provider. Run a bounded dry run, then an explicitly scoped insertion check. Preserve failed evidence for replay.

If retrieval fails, check embedding connectivity/dimensions, database role and prepared-statement settings. If only reranking fails, confirm model cache, permissions, hash, thread limit and budget. The API should still return the weighted-MMR path where appropriate. Inspect `ranker` for the actual request.

Back up private records, feedback, cursors, dataset snapshots, manifests and weights with access controls. Verify restore into an isolated destination. Do not replace an existing populated database with the public synthetic schema. Disk cleanup must be a separately scoped decision; no automatic cleanup of old checkpoints is included.

For model rollout, `smart_agents.deploy` stages an immutable release, preserves the prior model configuration and requires explicit health/canary verification. It does not accept arbitrary scripts from lesson text. See [cutover](cutover.md) before changing existing consumers.
