# Smart Agents: Corpus extraction and public release

Date: 2026-11-05
Status: extraction direction approved; expanded learning-loop audit

## Purpose

Publish the operational Corpus lessons mechanism as `Screddyice/smart-agents`,
with its documentation and references, so another person can run it without
access to Shawn's personal systems. The public repository should contain the
mechanism, not the contents of Shawn's lesson database or private health store.

## Approach

Recommended: extract the existing implementation into a repository with a fresh
history, adapt configuration and deployment examples for independent operators,
and preserve provenance in a source manifest. Preserve API and MCP behavior.
Use an MIT license as the proposed default for Shawn-owned code; retain any
third-party notices and check dependency licenses before release.

A fork of the private repository would carry unrelated health functionality and
private history. A rewrite would add behavior changes to an extraction and make
parity harder to establish. Neither is recommended.

## Verified source inventory

| Component | Source observed on this machine or server | Treatment |
|---|---|---|
| Agent-facing MCP | `SRC/nebos-consult/apps/api/tools/corpus-mcp-server.mjs`, origin `Screddyice/src-os` | Extract bridge, keep `corpus_search` and lesson row identifiers; correct outdated database comments |
| Local planning adapter | `~/.local/lib/corpus/corpus_search_mcp.py` | Preserve protocol behavior; remove hardcoded personal gateway and credential-discovery assumptions in public configuration |
| Retrieval and corrections API | SRC server `/opt/corpus/app/{main,gate,rank,rerank,train_export}.py` | Capture code only; preserve gating, deduplication, ranking, diversity, reranking and abstention behavior |
| Ingestion and classification | SRC server `/home/hermes/.local/corpus-ingest/{pipeline,classifier}` | Extract required packages and tests; identify active versus retired entrypoints and provider assumptions |
| Schema, evaluations and older design references | Existing `shawn-corpus` history/worktree `learning-corpus/`, relevant API tests and design docs | Select code and schema explicitly; reconcile with live implementation; replace private evaluation records with synthetic fixtures |
| Lessons reranker training and promotion | Historical worktree `learning-corpus/rerank/`, `learning-corpus/trainer/scripts/train_rerank.py`, `shawn_corpus/rerank/`, `shawn_corpus/eval/`, `deploy/ship-rerank-champion.sh` | Include dataset construction, labeling, local training, checkpoint selection, threshold calibration, promotion, shipping and rollback; modernize unsafe overwrite/deletion assumptions |
| Consolidation and correction lifecycle | Historical worktree `shawn_corpus/consolidation/` and associated tests/schema | Audit extraction boundaries, review queue, promotion/demotion and supersession; distinguish operational lesson handling from private observation/health paths |
| Health-specific runtime, excluded | Mac `~/.local/lib/shawn_corpus_runtime/` | Inspection shows health-pattern contracts, health weekly cases and a health adapter; do not mistake it for operational-lesson training or publish it as part of this extraction |
| Deployment examples | `SRC/src-box-config/systemd/`, installed Mac LaunchAgents | Publish portable opt-in templates; omit private paths, addresses and credentials |
| Database reference | Workspace `docs/reference/corpus-database.md` | Rewrite as a portable database runbook; do not publish private database history or unrelated health details |

Observed service state: the API is configured as a loopback FastAPI service on
port 8210. The user-scoped ingestion unit was inactive when inspected. Installed
files alone do not establish that ingestion currently works. The extraction
must test ingestion independently and report this live-state limitation.

The running API has `RERANK_ENABLED=1`, `RERANK_MODEL=/opt/corpus/models/champion`
and `RERANK_MIN_SCORE=-5.0`. A model artifact exists there. This establishes
configuration and artifact presence, not proof that every request uses it:
`/advise` may fall back to MMR if scoring fails. The historical champion manifest
records a trained run on 2026-09-03. Its recorded metrics are historical results,
not a current evaluation. The former weekly lessons-reranker LaunchAgent is
neither present in the installed LaunchAgents directory nor loaded under its
known label. The installed weekly fine-tune launcher instead calls the separate
health-pattern runtime. Do not claim that operational lessons currently train
weekly, or that every memory source currently reaches the corpus.

## Full learning loop and what training does

1. Capture evidence of a correction, failed approach, obscure environment
   constraint, inefficient sequence or durable new discovery. Preserve source
   identity, time, project and context needed to recognize the trigger later.
2. Extract a reusable lesson describing the trigger, failure, corrective action
   and a way to verify it. Classify significance and scope, check duplicates,
   embed it and write an attributable row into the private `learnings` database.
3. Before unfamiliar work, risky actions or retrying an unexplained failure, the
   agent deliberately calls `corpus_search(task)`. Vector search finds candidate
   database rows. A cross-encoder scores task/lesson pairs; thresholds and
   diversity filtering select the returned rows. If reranking fails, weighted
   MMR is the fallback. Keep raw relevance and learned scores distinguishable.
4. Return lesson text and row IDs. The agent checks applicability against live
   evidence, applies the relevant lesson and cites its ID. Retrieved text does
   not override instructions or authorize actions. Expired, superseded or
   inapplicable advice must be reportable and correctable.
5. Build labeled task/candidate/relevant-ID training examples from the lesson
   corpus, including hard negatives and tasks that should return no lesson.
   Fine-tune a small local cross-encoder, evaluate candidate checkpoints, choose
   the best eligible one and calibrate its abstention threshold.
6. Promote only if the candidate clears release gates. Ship an immutable model
   version with its threshold and manifest; verify the deployed artifact and
   actual `/advise` ranker behavior, then compare live retrieval against the
   previous release and roll back on regressions.

The database is the authoritative lesson store. Fine-tuning produces model
weights that improve selection of stored lessons; it is not a second source of
facts and does not write newly invented findings into the database. Classifier
or extraction model outputs become rows only through the ingestion checks.
Generating lesson prose from model weights is not the operational retrieval
contract. The historical reranker trainer uses the small
`cross-encoder/ms-marco-MiniLM-L-6-v2` family with PyTorch/Transformers, using
Apple MPS when available and CPU otherwise. This is distinct from the Qwen/MLX
health-pattern service and from the manually invoked general Qwen models.

## Coverage beyond ordinary memory summaries

The observed ingestion source is cmem's distilled learning feed. If an obscure
failure is absent from that feed, downstream retrieval and training cannot
recover it. Do not describe the current pipeline as lossless raw-session capture.

Include a source-coverage matrix for each supported host: what is captured,
which profile/account is read, which fields survive summarization, how progress
is checkpointed, how errors are retried, how privacy filters work, and what is
not supported. Host adapters for Claude, Codex and other agents need independent
evidence; sharing one MCP endpoint is not proof of shared capture.

Provide an explicit trusted lesson-recording path for discoveries that summary
memory omits, backed by the correction ingestion API. Do not invent an automatic
raw-transcript ingestion hook. Record minimal relevant evidence and the causal
lesson, excluding credentials and unrelated client/personal data. Document how
an agent records and verifies a lesson after resolving an unfamiliar failure.

Keep rejection and retry outcomes observable. A temporary classifier or database
failure must remain retryable and must not advance the settled cursor as if the
lesson had been intentionally rejected. Distinguish duplicate, low-significance,
unparseable and backend-failure cases. Provide bounded replay and a review queue
for uncertain or conflicting lessons; avoid silent permanent rejection of
rare, useful discoveries. Preserve source provenance through consolidation,
supersession and rollback. Do not silently delete original lesson evidence.

Define a private feedback record with `lesson_id`, task/source reference,
timestamp, outcome (`applied_successfully`, `repeated_mistake`, `irrelevant`,
`incorrect`, `superseded`), supporting evidence and any replacement lesson ID.
Submit feedback through a trusted administrative command/API, not a write tool
on the read-only MCP. Route it to the review queue; reviewed outcomes inform
corrections, supersession and new evaluation cases. Model feedback alone cannot
silently rewrite a lesson or label a training example as correct.

Rare-lesson acceptance must cover a unique one-off discovery: source evidence is
captured, the reusable constraint is preserved, a later differently worded task
retrieves its row, and the agent follows the corrective action in a synthetic
task. Include look-alike tasks where the lesson must not apply, contradictory
old lessons, repeated ingestion and supersession. Test capture, retrieval,
applicability and resulting behavior independently. These tests support a
measurable reduction in repeated mistakes; they cannot guarantee that an agent
will never repeat one.

## Local model setup and weekly training

Include a complete setup path for the operational cross-encoder: supported
Python/platform versions, pinned dependencies, model source and revision,
upstream model license, download/cache locations, hardware and resource limits,
offline behavior, isolated output directories and inference configuration.
Describe the separate embedding and classification providers and dimensions;
changing an embedding model requires a deliberate re-embedding migration.
Do not inherit the historical hardcoded Qwen gate as a permitted general agent
fallback. Provider routing must follow the host's current policy.

Include an opt-in weekly scheduler template and a matching manual command for:
snapshot/export, incremental example building, approved labeling provider,
dataset validation, training, evaluation, promotion and optional deployment.
Record the timezone, last attempt, last successful run, corpus watermark, growth
gate, base revision, dataset hashes, trained artifact hash, threshold, metrics
and deployed version. A scheduled skip, failed run or local promotion must not
be reported as a newly deployed model. Keep notification integrations opt-in.

Preserve versioned datasets, checkpoints and prior champions. Handle locks,
overlapping inference/training, insufficient data, exhausted provider budgets,
interrupted jobs, disk pressure and restore. Do not resume or train from an
adapter unless configuration establishes its provenance and intended mode.
Separate training, validation/checkpoint selection and held-out release tests;
the historical use of frozen evaluation metrics to select checkpoints must not
be represented as an independent held-out quality estimate. Include hard
negatives, no-answer cases and rare-lesson/key-term recall guardrails alongside
precision, hit rate, duplication and latency. Compare with the deployed champion
and an untrained/fallback baseline. Store all model and dataset lineage privately.
Group shared source sessions, duplicate/paraphrased lessons and generated task
variants before splitting so that related examples cannot leak across training,
validation and held-out release sets. Include an explicit leakage audit.

Restoring weekly operational training and active capture is part of full-system
readiness, not something the existing installation currently proves. First
resolve why the observed ingestion unit is inactive and reconcile newer host
policy with historical provider defaults. Do not blindly re-enable it. The
public code must include these facilities; installation and live scheduling
remain explicit operator choices. This audit changes documentation only and
does not start inference, training or a schedule.

## Proposed repository contents

API and storage schema; ingestion and classifier packages; read-only MCP
adapter; operational cross-encoder training and promotion; deployment templates; synthetic
examples and tests; README; architecture; installation; configuration reference;
API/MCP reference; ingestion and retrieval runbooks; evaluation methodology;
training runbook; troubleshooting; security/data boundaries; source provenance;
contributing guidance; license; CI.

The public quickstart should use an isolated database and synthetic lesson.
Keep existing endpoint contracts where practical. Document required services
and optional model providers explicitly. Do not enable local inference or
schedules as a side effect of installation.

## Data and trust boundary

Do not copy `.env` files, secret values, database dumps, raw transcripts, actual
lesson rows, personal evaluation datasets, health records, adapters, model
weights or checkpoints. Use an allowlist of source files and inspect every
public artifact. A fresh Git history avoids exposing deleted private material.

The MCP tool remains read-only. Correction ingestion is a separate trusted
administrative path and must not be exposed anonymously. Lesson text is
untrusted context, never authority to override host instructions. Document
authentication or network isolation for the API and isolation between corpora.

Public release must not retain automatic general-purpose local Qwen invocation.
Operational-reranker training is included, with explicit opt-in installation.
Private health analysis/training is outside this repository. This work must not
start inference, change installed models or settings, or overwrite training data.

## Extraction and runtime migration

First make the new repository independently runnable and verified, then publish
it publicly under the requested name. Public release is authorized by the user.
Record the exact source revisions and deployment file hashes used.

Complete the move in two verified phases: public extraction, then runtime
cutover to a pinned Smart Agents revision. Prepare and test the replacement
before changing any running service. Preserve database records, existing
configuration and rollback paths; no DNS change or data deletion is needed.
Compare configuration, verify health and synthetic retrieval through existing
consumers, then establish the new repository as the source of runtime code.
Do not activate the currently inactive ingestion unit merely as a source-path
cutover. Diagnose it and prepare the approved provider configuration first.
Report publication, capture readiness, training readiness and runtime cutover
independently; publication alone does not complete the requested move.

Track cutover per producer and consumer: local planning adapter, SRC OS bridge,
API unit, ingestion package/unit, training command, weekly scheduler and model
deployment script. Record the pinned Smart Agents revision each executes and
the exact pre-cutover rollback target. Updated documentation or one migrated
API is insufficient while another component still executes old repository code.

Update Corpus documentation and owning references to identify Smart Agents as
the source once published, using narrowly scoped changes in the owning repos.
Do not rewrite unrelated machine policies or shared team memory.

## Verification and acceptance

1. Fresh installation succeeds using only public files and documented dependencies.
2. A synthetic lesson passes ingestion, persists in an isolated test database,
   and is retrieved through the actual MCP protocol with its row ID. Distinguish
   deterministic fake-provider plumbing checks from real semantic checks. Use an
   already approved remote embedding/classification provider if available; do not
   start local inference. If no permitted provider is available, report semantic
   end-to-end validation as pending rather than claiming parity.
3. Duplicate and insignificant inputs are handled correctly; irrelevant tasks
   produce no advice; ranking and optional reranking retain their tested behavior.
4. Failure paths, input limits, timeouts and the read-only tool contract pass tests.
5. Training has synthetic export, dataset-split, checkpoint-selection, scheduler,
   promotion and rollback validation without starting local inference or writing
   into existing data or checkpoints. A real successful weekly training and
   deployment cycle requires separate runtime evidence; report pending until
   an authorized run completes and its artifact is consumed by retrieval.
6. All documentation links and setup commands are checked; source-file ownership,
   inherited licenses and dependency notices are verified before applying MIT;
   staged content and history pass a privacy/secret scan. Quarantine any source
   whose redistribution rights cannot be established until resolved.
7. GitHub reports the exact requested repository as public, with the verified
   commit accessible without authentication.
8. Changes to existing references are reviewed separately; runtime and publication
   evidence are reported independently.
9. A coverage matrix and source-to-database-to-MCP test establish the rare-lesson
   path, including explicit capture when ordinary memory summaries omit it.
10. Observation of a healthy endpoint is insufficient: verify capture freshness,
    retry state, data lineage, loaded model identity, fallback status and actual
    task results. Monitor failed ingestion, stale corpus, missed weekly runs,
    rejected candidates and repeated-mistake reports with bounded retention.

## Prior lessons applied

Corpus #6580: agent retrieval is explicitly invoked, not an automatic retrieval hook.
Corpus #7077: preserve and verify the existing hybrid ranking rather than replacing
it with a single similarity lookup.
Corpus #6589: inspect training mode, growth gates, active artifacts and local
caches together; do not infer the trained model from the scheduler name.
Corpus #6533: preserve the incoming fallback ordering when reranking fails.
Corpus #120: measure key-term recall and watch for style-over-content regressions.

## Remaining decision

The user approved the extraction direction and requested this fuller system
audit. Use the expanded design to make the implementation plan and carry it
through verification and publication; distinguish restored capabilities from
historical or currently inactive mechanisms. MIT remains the proposed license.
The earlier email to AJ remains pending the public project link and sender choice.
