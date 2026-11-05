# Runtime ownership checklist

Public source publication and live runtime migration are different outcomes. Keep a private receipt for every component.

| Component | New owner | Verify |
|---|---|---|
| Retrieval API | pinned Smart Agents checkout / `corpus_api.main:app` | process/module path, revision in health, database connectivity, relevant and no-answer tasks |
| Direct MCP | `.venv/bin/smart-agents-mcp` | consuming host's tools/list and authenticated tools/call |
| Gateway adapter | `smart_agents.gateway` | exact existing gateway/account and a returned lesson ID |
| SRC OS bridge | Smart Agents adapter launched by existing connector | connector tool discovery, row IDs and no write tools |
| Ingestion | new `corpus-ingest` executable | same approved source/provider/database/cursor, bounded source-to-row check |
| Dataset building | `smart_agents.rerank.triples` | source/evaluation exclusions and private output path |
| Local trainer | `smart_agents.train_rerank` | pinned base, immutable outputs and isolated validation/release |
| Weekly scheduler | new repository's executable | resolved paths, timezone, no duplicate old jobs, completion marker |
| Model deployment | `smart_agents.deploy` | artifact hash plus threshold, actual canary and rollback |
| Instructions/docs | applicable host instruction files | links to this repo and explicit invocation/capture guidance |

Before cutover record current paths, revisions, service state, model artifact/threshold, environment-file locations and rollback targets. Never put secret values in a receipt. Prepare the new checkout and dependencies, then run an isolated API/MCP canary before changing startup paths. Keep the current provider/account/database identities. Do not migrate private records to a public destination.

After a path-only change, an inactive service stays inactive. Restoring capture or a schedule needs source/provider readiness, not only a symlink. Preserve the old files for rollback. Do not erase old repositories or private training history. Documentation changes can point to the new source without deleting historical implementation files that other worktrees still use.
