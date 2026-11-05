"""corpus-api -- ingestion and advisory surface for Shawn's Corpus.

Two jobs:
  POST /corrections  capture a correction, gate it, store it if significant
  POST /advise       what has Shawn corrected about this kind of task?

The advisory endpoint is what Hermes calls before it acts. It must never make
Hermes worse, so every failure path returns empty advice rather than an error.
"""
from __future__ import annotations

import logging
import os

import psycopg
from fastapi import FastAPI, HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from corpus_api.gate import (
    DUPE_MAX_COSINE_DISTANCE,
    content_hash,
    embed,
    is_significant,
)
from corpus_api.rank import apply_weights, mmr
from corpus_api.rerank import Reranker, settings_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("corpus-api")

DSN = os.environ.get("RDS_URL") or os.environ.get("LEARNING_CORPUS_RDS_URL")
app = FastAPI(title="Smart Agents Corpus", version="0.1.0")


class Correction(BaseModel):
    what_happened: str = Field(..., min_length=1, max_length=8000)
    correction: str = Field(..., min_length=1, max_length=8000)
    agent_id: str = "unknown"
    # NOTE: org/source/category are all CHECK-constrained in the DB; see
    # validators below. Invalid values are coerced, never passed through.
    project: str | None = None
    # org/instance/category are NOT NULL in `learnings`. Existing data uses
    # "NEB" (437 rows) and "cross" (41); default to cross-org since a
    # correction about how Shawn wants work done rarely belongs to one org.
    org: str = "cross"
    # learnings_source_check allows only these three.
    source: str = "shawn"
    source_session_id: str | None = None


class AdviceRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=12000)
    agent_id: str | None = None
    limit: int = 5


# How hard /advise trades relevance for variety. 1.0 is the pre-MMR ordering
# (pure cosine); lower values spend result slots on distinct ideas. Chosen from
# the sweep in learning-corpus/eval/BASELINE.md: with rank.SIMILARITY_FLOOR in
# place, 0.7 removes every near-duplicate pair from the top 3 at no precision cost.
MMR_LAMBDA = 0.7

# Largest cosine distance a row may have and still count as advice. The old
# value, 0.6, let three engineering lessons through for "how do I tune a
# guitar": the nearest neighbours of almost any embedding cleared it. Measured
# on the eval set (BASELINE.md, 2026-09-03): relevant rows sit at 0.171-0.367,
# and the best match for a task the corpus has never covered sits at
# 0.477-0.559. 0.42 is equidistant from both edges and scores no-answer
# accuracy 1.0 with precision unchanged.
MAX_DISTANCE = 0.42

# Lazily loads the model on the first call. RERANK_ENABLED=0 in /opt/corpus/.env
# turns the stage off without a redeploy; any failure inside returns None and
# the Phase 1 ordering below takes over.
RERANKER = Reranker(settings_from_env())


@app.on_event("startup")
def _warm_reranker() -> None:
    # About 1.3 s on src. Done here so no request ever pays the load.
    RERANKER.warm()

VALID_ORGS = ("NEB", "Cliqk", "TRC", "personal", "cross")
VALID_SOURCES = ("shawn", "self", "observed")


def _conn() -> psycopg.Connection:
    # prepare_threshold=None: the Supabase pooler (port 6543, transaction mode)
    # rejects psycopg's auto-prepared statements with "prepared statement does
    # not exist" once a connection is reused. Harmless on a direct Postgres.
    if not DSN:
        raise RuntimeError("Corpus database is not configured")
    return psycopg.connect(DSN, row_factory=dict_row, prepare_threshold=None)


@app.get("/health")
def health() -> dict:
    try:
        with _conn() as conn:
            n = conn.execute("select count(*) as n from learnings").fetchone()["n"]
        return {"ok": True, "learnings": n, "reranker_enabled": RERANKER.settings.enabled, "reranker_model": RERANKER.settings.model, "reranker_loaded": RERANKER._scorer is not None, "reranker_sha256": getattr(RERANKER._scorer, "model_sha256", None), "reranker_min_score": RERANKER.settings.min_score, "revision": os.environ.get("SMART_AGENTS_REVISION", "unknown")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "Corpus dependency unavailable"}


@app.post("/corrections")
async def ingest(item: Correction) -> dict:
    """Gate a correction, then store it only if it clears every check."""
    digest = content_hash(item.what_happened, item.correction)

    with _conn() as conn:
        seen = conn.execute(
            "select id from learnings where source_entry_hash = %s", (digest,)
        ).fetchone()
    if seen:
        return {"stored": False, "reason": "exact duplicate", "id": seen["id"]}

    keep, reason, category = await is_significant(item.what_happened, item.correction)
    if not keep and reason.startswith("gate unavailable"):
        raise HTTPException(status_code=503, detail="Classification unavailable or pending review; retain evidence and retry after resolution")
    if not keep:
        log.info("rejected: %s", reason)
        return {"stored": False, "reason": reason}

    vector = await embed(f"{item.what_happened}\n{item.correction}")

    with _conn() as conn:
        # Semantic near-duplicate check. Done after the gate because it costs an
        # index scan and most candidates die at the gate.
        near = conn.execute(
            """
            select id, lesson, embedding <=> %s::vector as dist
            from learnings
            where embedding is not null
            order by embedding <=> %s::vector
            limit 1
            """,
            (str(vector), str(vector)),
        ).fetchone()
        if near and near["dist"] is not None and near["dist"] < DUPE_MAX_COSINE_DISTANCE:
            return {
                "stored": False,
                "reason": f"near-duplicate of #{near['id']} (dist {near['dist']:.4f})",
            }

        row = conn.execute(
            """
            insert into learnings
              (org, instance, agent_id, project, occurred_at, ingested_at,
               category, what_happened, lesson, embedding, source,
               source_session_id, source_entry_hash)
            values (%s, %s, %s, %s, now(), now(), %s, %s, %s, %s::vector, %s, %s, %s)
            returning id
            """,
            (
                item.org if item.org in VALID_ORGS else "cross",
                os.environ.get("SMART_AGENTS_INSTANCE", "smart-agents"), item.agent_id, item.project,
                category, item.what_happened, item.correction, str(vector),
                item.source if item.source in VALID_SOURCES else "shawn",
                item.source_session_id, digest,
            ),
        ).fetchone()
        conn.commit()

    log.info("stored #%s (%s): %s", row["id"], category, reason)
    return {"stored": True, "id": row["id"], "category": category, "reason": reason}


@app.post("/advise")
async def advise(req: AdviceRequest) -> dict:
    """Retrieve prior corrections relevant to a task.

    Hermes calls this before acting. Any failure returns empty advice: a broken
    corpus must never block or degrade the agent it is meant to improve.
    """
    try:
        vector = await embed(req.task)
        wanted = max(1, min(req.limit, 20))
        with _conn() as conn:
            # Over-fetch: MMR needs a pool to diversify from, and the corpus
            # holds many rows expressing one idea. 6x is enough that a run of
            # near-duplicates at the top does not starve the selection.
            rows = conn.execute(
                """
                select id, category, lesson, what_happened, agent_id, occurred_at,
                       embedding::text as emb,
                       embedding <=> %s::vector as dist
                from learnings
                where embedding is not null and superseded_by is null
                order by embedding <=> %s::vector
                limit %s
                """,
                (str(vector), str(vector), wanted * 6),
            ).fetchall()

        # `raw_dist` keeps the true cosine distance for the response's
        # `relevance` field; `dist` is what the MMR fallback sorts on and gets
        # adjusted by category and age below.
        fetched = [
            {
                "id": r["id"],
                "category": r["category"],
                "lesson": r["lesson"],
                "occurred_at": r["occurred_at"],
                "dist": float(r["dist"]),
                "raw_dist": float(r["dist"]),
                "embedding": [float(x) for x in r["emb"].strip("[]").split(",")],
            }
            for r in rows
            if r["dist"] is not None
        ]

        # The cross-encoder sees every fetched row: its own score threshold is
        # the abstention rule on that path, and on eval set v2 the cosine cutoff
        # in front of it cost answers (hit@3 0.683 -> 0.594, BASELINE.md). The
        # cutoff stays on the MMR fallback, which has no other way to say no.
        # Rows arrive in cosine order, so the reranker's pool cap takes the
        # closest ones. None means the stage is off or failed.
        ranked = RERANKER.rerank(req.task, fetched, k=wanted)
        if ranked is None:
            ranker = "mmr"
            candidates = [c for c in fetched if c["dist"] < MAX_DISTANCE]
            ranked = mmr(apply_weights(candidates), lambda_=MMR_LAMBDA, k=wanted)
        else:
            ranker = "cross-encoder"

        advice = []
        for c in ranked:
            row = {
                "id": c["id"],
                "category": c["category"],
                "lesson": c["lesson"],
                "relevance": round(1 - c["raw_dist"], 3),
                "occurred_at": c["occurred_at"].date().isoformat() if c.get("occurred_at") else None,
            }
            if "score" in c:
                row["score"] = round(c["score"], 3)
            advice.append(row)
        return {"ok": True, "advice": advice, "ranker": ranker}
    except Exception as exc:  # noqa: BLE001
        log.warning("advise failed, returning empty: %s", exc)
        return {"ok": False, "advice": [], "error": "Corpus dependency unavailable"}


@app.get("/export")
def export(since_id: int = 0, limit: int = 10000) -> dict:
    """Training export. Consumed by the weekly Modal fine-tune job."""
    with _conn() as conn:
        rows = conn.execute(
            """
            select id, category, what_happened, lesson, agent_id, project, occurred_at
            from learnings
            where id > %s and lesson is not null and length(trim(lesson)) > 0
            order by id
            limit %s
            """,
            (since_id, limit),
        ).fetchall()
    return {
        "count": len(rows),
        "max_id": max((r["id"] for r in rows), default=since_id),
        "rows": [
            {
                "id": r["id"],
                "category": r["category"],
                "instruction": (r["what_happened"] or "").strip(),
                "response": (r["lesson"] or "").strip(),
            }
            for r in rows
        ],
    }

# Administrative mutation paths are authenticated separately from MCP reads.
import secrets
from fastapi import Request
from fastapi.responses import JSONResponse
from smart_agents.feedback import Feedback


@app.middleware('http')
async def administrative_boundary(request: Request, call_next):
    if request.url.path == '/export' or (request.method not in ('GET', 'HEAD', 'OPTIONS') and request.url.path != '/advise'):
        token = os.environ.get('CORPUS_ADMIN_TOKEN')
        if not token:
            return JSONResponse({'detail':'Administrative writes are not configured'},status_code=503)
        supplied = request.headers.get('authorization','')
        if not secrets.compare_digest(supplied, 'Bearer '+token):
            return JSONResponse({'detail':'Unauthorized'},status_code=401)
    return await call_next(request)


@app.post('/feedback')
def feedback(item: Feedback):
    with _conn() as conn:
        row = conn.execute(
            '''insert into lesson_feedback
               (lesson_id,task,outcome,evidence,source_reference,replacement_lesson_id)
               values (%s,%s,%s,%s,%s,%s) returning id''',
            (item.lesson_id,item.task,item.outcome,item.evidence,item.source_reference,item.replacement_lesson_id)
        ).fetchone()
        conn.commit()
    return {'stored':True,'id':row['id'],'review_status':'pending'}

from smart_agents.feedback import FeedbackReview


@app.post('/feedback/{feedback_id}/review')
def review_feedback(feedback_id: int,item: FeedbackReview):
    with _conn() as conn:
        row=conn.execute('''update lesson_feedback set review_status=%s,reviewer=%s,review_note=%s,reviewed_at=now()
                            where id=%s and review_status='pending' returning id''',
                         (item.status,item.reviewer,item.note,feedback_id)).fetchone()
        if not row:raise HTTPException(status_code=409,detail='Feedback missing or already reviewed')
        conn.commit()
    return {'id':row['id'],'review_status':item.status}
