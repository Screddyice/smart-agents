# Architecture

| Stage | Code | Output | Failure behavior |
|---|---|---|---|
| Distilled source capture | `learning_corpus_pipeline/cmem_source.py` | Attributable observation/summary items | Source failure remains retryable |
| Classification | `learning_classifier/` | Structured correction, mistake, inefficiency, strategy pivot or durable fact | No implicit local generative fallback |
| Persistence | `learning_corpus_pipeline/writer_rds.py`, `corpus_api/main.py` | Private pgvector lesson rows | Exact/semantic dedupe; no cursor settlement after backend/write failure |
| Retrieval | `corpus_api/{main,rank,rerank}.py` | Relevant lessons with IDs | Weighted MMR if reranking fails; abstain below relevance threshold |
| Agent integration | `smart_agents/mcp.py`, `smart_agents/gateway.py` | Read-only `corpus_search` | Dependency failure is an error, not a fabricated answer |
| Explicit recording/feedback | `smart_agents/{admin,feedback}.py` | New lesson or pending feedback | Separate authenticated administrative path |
| Dataset building | `smart_agents/rerank/{triples,labeler,dataset}.py` | Private labeled task/candidate triples | Reviewed labels, exclusions and provenance |
| Local training | `smart_agents/{training,train_rerank,weekly}.py` | Immutable run/checkpoints and evaluation manifest | Keep prior versions on failure |
| Consolidation | `smart_agents/consolidation/learnings.py` | Reviewable supersession suggestions | No deletion; explicit apply only |
| Deployment | `smart_agents/deploy.py`, `deploy/` | Pinned code/model/configuration | Verify exact release; preserve rollback |

The storage schema retains legacy `shawn_feedback` and `source='shawn'` names for compatibility. They represent explicit human feedback; `self` and `observed` cover agent discoveries and evidenced patterns. The database, embedding dimensions and provider identity must match across producers and consumers.

This is an operational lessons system, distinct from personal health analysis or a chat model. A trained relevance model ranks actual database rows. The original classifier API and package entry points are retained for existing integrations; new orchestration lives under `smart_agents`.
