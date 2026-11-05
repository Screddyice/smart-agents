# Retrieval and API reference

`corpus_search(task: string, limit: integer = 5)` is the only MCP tool. It returns text containing lesson IDs, categories, relevance and actual lesson text. Limits are 1–20. Empty results mean no relevant lesson was found. Backend failure is reported separately. The bridge is read-only; it never exposes correction or feedback writes.

Call it before unfamiliar repo/subsystem work, expensive actions, unexplained errors and retrying a failed approach. State the concrete task, symptom and relevant constraints. Check returned advice against current owning sources; older lessons can be stale. Cite the applicable row ID. Do not silently treat irrelevant lessons as instructions.

| HTTP endpoint | Meaning | Boundary |
|---|---|---|
| `GET /health` | Database connectivity/count, configured model and source revision | Private read |
| `POST /advise` | `{task, limit}` -> `{ok, advice, ranker}` | Private read |
| `POST /corrections` | Trusted evidence -> stored lesson ID or rejection reason | Admin bearer token |
| `POST /feedback` | Lesson-linked outcome -> pending feedback row | Admin bearer token |
| `POST /feedback/{id}/review` | Attributed accept/reject of pending feedback | Admin bearer token |
| `GET /export` | Legacy lesson export for operator tooling | Admin bearer token |

`/advise` first fetches vector neighbors from non-superseded rows. A cross-encoder jointly scores task and lesson, drops scores below its threshold, then suppresses duplicates. If reranking is disabled, fails or exceeds its budget, the API uses cosine cutoff 0.42, category/age weighting and maximal marginal relevance with lambda 0.7. These are inherited measured defaults, not universal thresholds for another embedding space.

`relevance` is raw cosine relevance; optional `score` is the model logit. They are not calibrated probabilities. `ranker` identifies the path used on that request. Health configuration alone does not establish that reranking succeeded. A broken dependency produces `ok:false`; the MCP reports a failure instead of treating it as a successful empty result.
