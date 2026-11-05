# Coverage matrix

| Surface | Included mechanism | Evidence needed |
|---|---|---|
| Claude/Codex/Hermes and other hosts | Same read-only MCP tool and instruction snippet | Each consuming host lists and calls `corpus_search` with returned row IDs |
| cmem source | Observation/summary reader with pagination and full summary enrichment | A new attributable observation becomes a private corpus row |
| Explicit discoveries | Trusted `record` command/API | Stored ID, then retrieval from a differently worded relevant task |
| Task outcomes | Trusted feedback command/API | Feedback row linked to a known lesson, pending review |
| Raw prompts/transcripts | Excluded from automatic ingestion | No claim of complete session coverage |
| Weekly lessons training | Local cross-encoder pipeline and scheduler templates | Successful run, passing release gate, exact deployed artifact and live retrieval evidence |
| Private health runtime | Excluded | Separate owning system |

For every real installation record host, client/profile, source identity, source watermark, last successful insertion, classifier/embedding identities, last successful training run, current model hash and last MCP canary. Do not put private receipts in Git. A configured MCP server, installed package, enabled schedule, existing model directory or healthy HTTP endpoint proves only that layer.

The extraction audit found an existing retrieval API configured for a trained champion, an inactive ingestion unit and no loaded weekly lessons-reranker job. A separate installed weekly job handled health-pattern training. These are migration findings, not readiness claims for a fresh installation or guarantees about every current host.
