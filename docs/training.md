# Weekly training and release

The trained component is a relevance scorer. Ingestion writes lessons to Postgres; training reads reviewed task/candidate labels and writes model weights. A trained model is not a new source of findings.

## Prepare examples

`python -m smart_agents.rerank.triples` can build task/candidate/relevant-ID examples from the selected private corpus using typed JEV relevance judgments. Configure credentials/model and source scope first. Labeled output stays outside the repository. It excludes rows from known evaluation pools. Labels require review; generated labels are not ground truth merely because a model emitted them.

Each triple has `task_id`, `task`, `candidates` (`id`, `lesson`), `relevant_ids`; preserve `source_session_id` and reviewed `lesson_group_ids` for related/paraphrase evidence. Include difficult negatives and no-answer tasks. Raw private examples and old evaluation datasets are not shipped publicly.

## Prepare, train, evaluate

```sh
smart-agents weekly --triples /private/triples.jsonl --dry-run
# Explicit local compute, after reviewing the prepared partitions:
smart-agents weekly --triples /private/triples.jsonl --base /private/model-cache/pinned-base
```

Every attempt creates a new run directory with an immutable source snapshot, source hash, prepared partitions, status and any failure marker. Grouping joins related source sessions, task variants, candidate lesson IDs and reviewed paraphrase groups before splitting. A connected dataset with too few independent groups must be rebuilt; the code refuses to manufacture a passing split. Both validation and release need answerable and no-answer tasks.

Training uses two epochs, seed 0, AdamW at 1e-5 with weight decay 0.01, batch size 16 and a maximum of eight negatives per positive. Each epoch's weights are retained. Validation AUC selects the best checkpoint. The held-out release set does not select checkpoints or tune thresholds. The candidate's abstention threshold is calibrated on validation examples, then release metrics compare it with the current champion (or the pinned stock base when no champion exists).

Promotion requires no regression in precision@3, hit@3 or no-answer accuracy and an improvement in at least one. Invalid/nonfinite/incomplete metrics fail closed. Retain rare-lesson/key-term regression examples and measure them separately when reviewing release quality. Additional feedback cases must not leak into training equivalents.

`release.json` records model identity, hash, threshold, source hash and comparison. A passing candidate updates a small private champion pointer; old runs and weights remain untouched. A tied or losing candidate retains the incumbent. Local promotion does not deploy the model. Use the deployment command and verify actual served ranker and canary results. Retain rollback records and previous weights.

## Weekly scheduling

The macOS LaunchAgent and Linux user-service/timer templates are in `deploy/`. Substitute absolute paths and timezone deliberately. They call `smart-agents weekly`; they do not reinstall a model, collect unrelated transcripts, run a Qwen council, or modify other apps. No schedule is installed automatically.

Schedule reviewed-example preparation separately, or explicitly pass `--build-count N` for bounded incremental JEV labeling before the training window. Automatic labels still need a reviewed dataset before promotion; strict provenance/rare-case gates can refuse generated data. Configure the provider hard spending cap; the builder stops subsequent requests at its recorded USD budget and never retries uncertain calls. A scheduled run over unchanged triples is skipped after a completed run; it is not evidence of learning new lessons. A free-space gate protects versioned outputs. Read each run's source hash, groups, completion/failure markers, gate outcome and deployment receipt. A missing completion marker means incomplete, not success. The process lock prevents overlapping weekly attempts; coordinate separate inference/GPU users manually.

## Verify safely

The default test suite exercises data grouping, immutability, metric gates, refusal paths and scheduler/command contracts without running training. Explicit model tests load the upstream cross-encoder. Only an actual completed private training/evaluation/deployment cycle establishes that weekly learning works for a particular installation. Do not advertise historical evaluation numbers as current results.

Reviewed feedback is resolved with `smart-agents review-feedback /private/review.json` (`feedback_id`, `status`, `reviewer`, `note`). Accepted feedback remains linked evidence for curating future regression cases; it is not automatically treated as a new fact or injected into training.
