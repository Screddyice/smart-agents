-- Apply only to an explicitly selected empty operational corpus database.
-- 768 dimensions matches nomic-embed-text. Do not alter an existing corpus's dimensions implicitly.
create extension if not exists vector;
create table if not exists learnings (
 id bigserial primary key,
 org text not null,
 instance text not null,
 agent_id text not null,
 project text,
 occurred_at timestamptz not null,
 ingested_at timestamptz not null default now(),
 category text not null check(category in ('correction','new_info','strategy_pivot','inefficiency','mistake')),
 what_happened text not null,
 shawn_feedback text,
 lesson text not null,
 tags text[],
 embedding vector(768),
 source_session_id text,
 source_entry_hash text,
 source text not null default 'shawn' check(source in ('shawn','self','observed')),
 superseded_by bigint references learnings(id),
 constraint learnings_dedupe unique(instance,agent_id,source_entry_hash,category)
);
create index if not exists learnings_embedding_hnsw on learnings using hnsw(embedding vector_cosine_ops);
create index if not exists learnings_source on learnings(source_session_id);
create table if not exists lesson_feedback (
 id bigserial primary key,
 lesson_id bigint not null references learnings(id),
 occurred_at timestamptz not null default now(),
 task text not null,
 outcome text not null check(outcome in ('applied_successfully','repeated_mistake','irrelevant','incorrect','superseded')),
 evidence text not null,
 source_reference text,
 replacement_lesson_id bigint references learnings(id),
 review_status text not null default 'pending' check(review_status in ('pending','accepted','rejected'))
);
-- Reviewed outcomes remain attributable; they do not automatically rewrite lessons.
alter table lesson_feedback add column if not exists reviewer text;
alter table lesson_feedback add column if not exists review_note text;
alter table lesson_feedback add column if not exists reviewed_at timestamptz;
